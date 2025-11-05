import difflib
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from PIL import Image, ImageEnhance

from src.layout_analysis.image_layout_analysis import ImageLayoutAnalyzer
from src.preprocessing.iamge_preprocessing import ImagePreprocessor


class ImageTextRecognizer:

    def __init__(self,  trocr_model="trocr_german_handwritten" ,
                        folder_path = "../../input/",
                        image_name="page_04A",
                        image_type="png",
                        lp_score_thresh=0.5,
                        to_run_dla=False,
                        to_run_ocr=False,
                        form_type=None):

        self.trocr_model = trocr_model
        self.trocr_model_path = f"../../models/{trocr_model}"
        self.folder_path = folder_path
        self.image_name = image_name
        self.image_type = image_type
        self.lp_score_thresh = lp_score_thresh
        self.to_run_dla = to_run_dla
        self.to_run_ocr = to_run_ocr
        self.image_path = f"{folder_path}{self.image_name}.{self.image_type}"
        self.layout_analyzer = None
        self.image_pp = None
        self.image_trocr = None
        self.lines_ocr_results = None
        self.words_ocr_results = None
        self.cell_map = None
        self.optimized_ocr_results = None
        self.form_content = None
        self.htr_results = None
        self.form_type = form_type
        self.output_path = f"../../output/ocr/ocr_{self.image_name}.json"
        self.image_processor= ImagePreprocessor(real_image=False)

    def layout_analyze(self):
        self.layout_analyzer = ImageLayoutAnalyzer(image_name=self.image_name,
                                                   image_type=self.image_type,
                                                   to_show_results=True,
                                                   write_label_text = True,
                                                   score_thresh = self.lp_score_thresh,
                                                   run_dla=self.to_run_dla,
                                                   form_type=self.form_type,
                                                   folder_path=self.folder_path,
                                                   )
        self.layout_analyzer.analyze()
        self.image_path = self.layout_analyzer.preprocessed_image_path
        self.image_pp = self.layout_analyzer.image_pp
        self.form_content = self.layout_analyzer.str_form_data.get("form_content", [])
        self.form_type = self.layout_analyzer.str_form_data.get("form_type", [])

    def run_ocr(self, margin=5):
        """
        Run OCR using the DocTR predictor and process the result.
        """
        if not os.path.exists(self.output_path) or self.to_run_ocr:

            # Load OCR model
            model = ocr_predictor(pretrained=True)

            # Load image or PDF
            if self.image_path.endswith(".png") or self.image_path.endswith(".jpg"):
                doc = DocumentFile.from_images(self.image_path)
            else:
                doc = DocumentFile.from_pdf(self.image_path)

            # Run OCR and layout analysis
            results = model(doc)

            # results.show()

            self.visualize_ocr_results(results)

            self.extract_ocr_results(results)

            self.optimize_ocr_results(margin=margin)

            self.run_htr()

            self.save_ocr_results()

        else:
            self.optimized_ocr_results = json.load(open(self.output_path))
            self.visualize_optimized_ocr_results()


    def visualize_ocr_results(self, result):
        """
        Display OCR results with bounding boxes and recognized text using OpenCV and Matplotlib.
        """
        self.image_pp  = cv2.imread(self.image_path)
        image = self.image_pp.copy()
        h, w, _ = image.shape

        for block in result.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    x1 = int(x_min * w)
                    y1 = int(y_min * h)
                    x2 = int(x_max * w)
                    y2 = int(y_max * h)

                    # Draw bounding box
                    cv2.rectangle(image , (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Show and save visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
        plt.title("OCR Output")
        plt.axis("off")
        plt.show()

    def extract_ocr_results(self, result, line_level=False):
        """
        Save OCR results in structured JSON format.
        """
        self.lines_ocr_results = []
        self.words_ocr_results = []
        h, w, _ = self.image_pp .shape

        for page_num, page in enumerate(result.pages):
            for block in page.blocks:
                for line_index, line in enumerate(block.lines):
                    words_list = []
                    x_coords = []
                    y_coords = []
                    for word_index, word in enumerate(line.words):
                        (x_min, y_min), (x_max, y_max) = word.geometry

                        x_min = int(x_min * w)
                        y_min = int(y_min * h)
                        x_max = int(x_max * w)
                        y_max = int(y_max * h)

                        word_entry = {
                            "line": line_index+1,
                            "texts": word.value,
                            "coordinates": [x_min, y_min, x_max, y_max]
                        }
                        words_list.append(word_entry)

                        x_coords.extend([x_min, x_max])
                        y_coords.extend([y_min, y_max])

                        self.words_ocr_results.append(word_entry)

                    line_text = " ".join([w["texts"] for w in words_list])

                    line_entry = {
                        "line_index": line_index + 1,
                        "texts": line_text,
                        "coordinates": [
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ]
                    }
                    if line_level:
                        line_entry["words"] = words_list

                    self.lines_ocr_results.append(line_entry)

    def optimize_ocr_results(self, margin):
        def ocr_sort_key(entry):
            x1, y1, x2, y2 = entry["coordinates"]
            return round(y1 / 10), x1

        self.optimized_ocr_results = []
        self.initialize_cells()

        self.assign_words_to_cells(margin=margin)
        self.finalize_cell_texts()

        self.optimized_ocr_results.sort(key=ocr_sort_key)
        self.update_text_classification()
        self.update_htr_texts_coordinates()
        self.visualize_optimized_ocr_results()

    def initialize_cells(self):
        cell_map = {}
        for idx, cell in enumerate(self.layout_analyzer.layout):
            x1, y1, x2, y2 = map(int, [cell.block.x_1, cell.block.y_1, cell.block.x_2, cell.block.y_2])
            area = (x2 - x1) * (y2 - y1)
            cell_map[idx] = {
                "table_id": getattr(cell, "text", -1),
                "coordinates": [x1, y1, x2, y2],
                "area": area,
                "words": []
            }
        self.cell_map = cell_map

    def assign_words_to_cells(self, margin):

        def box_includes(a, b, margin=5):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return (
                    bx1 >= ax1 - margin and bx2 <= ax2 + margin and
                    by1 >= ay1 - margin and by2 <= ay2 + margin
            )

        for word in self.words_ocr_results:
            x1, y1, x2, y2 = word["coordinates"]
            word_text = word.get("texts", "")
            corrected_word, is_word_found = self.correct_word_from_form_content(word_text)
            best_idx, best_area = None, float("inf")

            for idx, cell in self.cell_map.items():
                if box_includes(cell["coordinates"], [x1, y1, x2, y2], margin):
                    if cell["area"] < best_area:
                        best_idx = idx
                        best_area = cell["area"]

            if best_idx is not None:
                self.cell_map[best_idx]["words"].append(corrected_word)
            else:
                ocr_type = "OCR" if is_word_found else "HTR"
                self.optimized_ocr_results.append({
                    "table_id": -1,
                    "texts": corrected_word,
                    "type": ocr_type,
                    "coordinates": [x1, y1, x2, y2]
                })

    def finalize_cell_texts(self):
        for cell in self.cell_map.values():
            if cell["words"]:
                full_text = " ".join(cell["words"])
                fake_entry = {
                    "texts": full_text,
                    "coordinates": cell["coordinates"]
                }

                correction = self.match_against_form_content(fake_entry)
                if correction:
                    corrected_text, text_type, htr_entry = correction

                    if corrected_text:
                        self.optimized_ocr_results.append({
                            "table_id": cell["table_id"],
                            "texts": corrected_text,
                            "type": text_type,
                            "coordinates": cell["coordinates"]
                        })

                    if htr_entry and htr_entry != ":":
                        self.optimized_ocr_results.append({
                            "table_id": cell["table_id"],
                            "texts": htr_entry,
                            "type": "HTR",
                            "coordinates": cell["coordinates"]
                        })
                else:
                    self.optimized_ocr_results.append({
                        "table_id": cell["table_id"],
                        "texts": full_text,
                        "type": "HTR",
                        "coordinates": cell["coordinates"]
                    })

    def correct_word_from_form_content(self, word_text, threshold=0.85):
        for entry in self.form_content:
            for ref_word in entry.get("texts", "").split():
                sim = difflib.SequenceMatcher(None, word_text, ref_word).ratio()
                if sim >= threshold:
                    return ref_word, True
        return word_text, False

    def update_text_classification(self, threshold=0.85):
        """
        Re-evaluates and updates the type (OCR or HTR) of each optimized OCR result,
        based on similarity to known form content entries.
        """
        for i, entry in enumerate(self.optimized_ocr_results):
            if entry["type"] == "HTR":
                for ref in self.form_content:
                    if self.are_ocr_entries_similar(entry, ref, threshold, check_coordinates=False):
                        self.optimized_ocr_results[i]["type"] = "OCR"
                        break

    def update_htr_texts_coordinates(self, threshold=0.85):
        """
            Updates coordinates of HTR entries by finding the most similar OCR word
            based on text content and spatial proximity. Stores original cell coordinates
            in 'cell_coordinates' for reference.
        """
        for i, entry in enumerate(self.optimized_ocr_results):
            if entry["type"] == "HTR":
                for word in self.words_ocr_results:
                    if self.are_ocr_entries_similar(entry, word, threshold,
                                                    check_coordinates=True):
                        self.optimized_ocr_results[i]["htr_coordinates"] = word["coordinates"]
                        break

    def update_htr_texts_coordinates_vx(self, threshold=0.85, margin=20):
        """
            For each HTR entry, the OCR word list is searched for the best match.
            The selection takes into account both text similarity and spatial proximity.
            Even if there are multiple similar OCR words, the best match is chosen.
        """

        def combined_similarity(entry, word):
            text_sim = difflib.SequenceMatcher(None, entry["texts"], word["texts"]).ratio()
            coord_sim = self.are_ocr_entries_similar(entry, word, threshold=threshold, margin=margin,
                                                     check_coordinates=True)
            return text_sim + (0.1 if coord_sim else 0)  # kleiner Bonus bei Koordinaten-Nähe

        for i, entry in enumerate(self.optimized_ocr_results):
            if entry["type"] != "HTR":
                continue

            best_word = None
            best_score = 0.0

            for word in self.words_ocr_results:
                score = combined_similarity(entry, word)
                if score > best_score:
                    best_score = score
                    best_word = word

            if best_word:
                self.optimized_ocr_results[i]["htr_coordinates"] = best_word["coordinates"]

    def match_against_form_content(self, original_val, threshold=0.5, margin=15):
        """
            Tries to find the most similar text from the layout form_content.
            Returns the matched text if similarity is above threshold.
            """
        remaining_text = None
        for val2 in self.form_content:
            if self.are_ocr_entries_similar(original_val, val2, threshold):
                for entry in self.optimized_ocr_results:
                    if self.are_ocr_entries_similar(entry, val2, threshold, margin):
                        return None

                new_val = val2["texts"]
                original_text = original_val.get("texts")

                if len(new_val) < len(original_text):
                    remaining_text = original_text[len(new_val):].strip()
                    if remaining_text in new_val:
                        remaining_text = None

                return new_val, "OCR", remaining_text

        return original_val.get("texts"), "HTR", remaining_text

    def save_ocr_results(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.optimized_ocr_results, f, indent=4, ensure_ascii=False)
        print(f"✅ OCR results saved in {self.image_name}.json")

    def visualize_optimized_ocr_results(self, figsize=(12, 10)):
        """
        Visualizes the optimized OCR results with bounding boxes and labels on the image.
        """
        image = self.image_pp.copy()

        for entry in self.optimized_ocr_results:
            x1, y1, x2, y2 = entry["coordinates"]
            ocr_type = entry.get("type", "")

            color = (0, 255, 0) if ocr_type == "OCR" else (0, 0, 255)  # Green for OCR, Red for HTR
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        # plt.title("Optimized OCR Results")
        plt.savefig(f"../../output/ocr/ocr_{self.image_name}.png", bbox_inches="tight", pad_inches=0)
        plt.show()

    @staticmethod
    def are_ocr_entries_similar(val1, val2, threshold=0.5, margin=30, check_coordinates=True):
        text1 = val1.get("texts", "")
        text2 = val2.get("texts", "")
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

        c1 = val1.get("coordinates", [])
        c2 = val2.get("coordinates", [])

        content_similar = similarity >= threshold
        if check_coordinates:
            coordinates_close = abs(c1[0] - c2[0]) <= margin and abs(c1[1] - c2[1]) <= margin
            return content_similar and coordinates_close

        return content_similar

    def run_htr(self):
        """
            Run TrOCR (Transformer-based handwriting recognition) on recognized layout regions.
        """
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # Initialisiere TrOCR
        processor = TrOCRProcessor.from_pretrained(self.trocr_model_path)
        model = VisionEncoderDecoderModel.from_pretrained(self.trocr_model_path)

        self.extract_htr_results(processor, model)

        self.save_htr_results()


    def extract_htr_results(self, processor, model):
        """
        Displays the recognized layout blocks mad the results of TrOCR recognition block by block  with bounding boxes.
        :param processor:
        :param model:
        :return:
        """
        self.htr_results = []
        self.image_trocr = self.layout_analyzer.image_pp.copy()
        h, w = self.image_trocr.shape[:2]

        for i,block in enumerate(self.optimized_ocr_results):
            if block["type"] == "HTR":
                coordinates_id = "htr_coordinates" if "htr_coordinates" in block.keys() else "coordinates"
                x_min, y_min, x_max, y_max = block[coordinates_id]
                scale_rate = 0
                x_min = max(0, x_min - scale_rate)
                y_min = max(0, y_min - scale_rate)
                x_max = min(w, x_max + scale_rate)
                y_max = min(h, y_max + scale_rate)

                segment = self.image_trocr[y_min:y_max, x_min:x_max]
                segment_pil = Image.fromarray(segment).convert("RGB")

                # image section improving
                # segment_pil=self.improve_image_section(segment, False)

                pixel_values = processor(images=segment_pil, return_tensors="pt").pixel_values
                generated_ids = model.generate(
                    pixel_values,
                    num_beams=5,
                    early_stopping=True,
                    max_length=128,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2
                )

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                plt.figure(figsize=(4, 2))
                plt.imshow(segment_pil)
                plt.axis("off")
                plt.title(f"recognized text:  {text}", fontsize=10)
                plt.savefig(f"../../output/htr/{i}.png")
                plt.show()

                result = {
                    "texts": block["texts"],
                    "htr_texts": text,
                    "coordinates": block["coordinates"]
                }
                # print("HTR results:", result)
                self.optimized_ocr_results[i]["texts"] = text

    def save_htr_results(self):
        """
        Saves the results of TrOCR recognition block by block in JSON format.
        """
        with open(f"../../output/ocr/ocr_{self.image_name}_htr.json", "w", encoding="utf-8") as f:
            json.dump(self.htr_results, f, indent=4, ensure_ascii=False)

        print(f"✅ TrOCR results saved in ocr_{self.image_name}_htr.json")

    def improve_image_section(self, segment, using_processor=True):
        if using_processor:
            self.image_processor.image = segment
            segment_processed = self.image_processor.run_preprocessing()
            segment_rgb = cv2.cvtColor(segment_processed, cv2.COLOR_GRAY2RGB)
            pil = Image.fromarray(segment_rgb).convert("RGB")
            return pil

        segment_pil = Image.fromarray(segment).convert("RGB")

        gray = np.array(segment_pil.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        pil = Image.fromarray(processed).convert("RGB")
        pil = ImageEnhance.Contrast(pil).enhance(1.8)
        pil = ImageEnhance.Sharpness(pil).enhance(1.3)

        return pil
