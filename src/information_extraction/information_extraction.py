import json
import os
import difflib
from collections import defaultdict

from src.text_recognition.image_text_recognition import ImageTextRecognizer


class InformationExtractor:
    def __init__(self, input_folder = "../../input", input_image="page_01.png"):

        self.meta_keys = ["Name, Vorname",
                         "Träger",
                         "Geburtsdatum",
                         "Anzahl bewilligter FLS bzw. ganz- tags/halbtags im Bewilligungszeitraum",
                         "Bescheid von/bis",
                         "Nachweiszeitraum: (Monat/jahr)",
                         "Gesamt FLS im Nachweiszeitraum",
                         "Arbeitsbereich",
                         "Bedarf Laut ITP:",
                         "Monat / Jahr",
                         "Name des Leistungserbringers",
                         "Anzahl bewilligter FLS",
                         "Leistungsart",
                         "Bedarfszeitraum von/bis",
                         ]
        self.input_folder = input_folder
        self.image_name = os.path.splitext(input_image)[0]
        self.image_type = input_image[-3:]
        self.input_files = os.listdir(self.input_folder)
        self.text_recognizer = None
        self.form_type = input_image[0:len(input_image)-4]
        self.data = None
        self.meta_table_id = None
        self.entries_table_id = None
        self.meta_infos = None
        self.table_entries = None
        self.extracted_data = []


    def run(self, rerun_ocr=False):
        json_path = f"../../output/ocr/ocr_{self.image_name}.json"
        if not os.path.exists(json_path) or rerun_ocr:
            self.text_recognizer = ImageTextRecognizer(image_name=self.image_name,
                                                       trocr_model="trocr_german_handwritten",
                                                       image_type=self.image_type,
                                                       lp_score_thresh=0.5)
            self.text_recognizer.layout_analyze()
            self.text_recognizer.run_ocr()
            self.form_type = self.text_recognizer.form_type
            self.data = self.text_recognizer.optimized_ocr_results
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self.extract_tables_ids()
        self.extract_meta_infos()
        self.extract_table_entries()
        self.save_extracted_data()

    def run_for_all_files(self):
        self.text_recognizer = ImageTextRecognizer(
                                                   trocr_model="trocr_german_handwritten",
                                                   image_type=self.image_type,
                                                   lp_score_thresh=0.5)
        print("Input Files: ", self.input_files)

        for filename in self.input_files:
            if filename.endswith(".png") or filename.endswith(".jpg") and filename.__contains__("A"):
                image_name = os.path.splitext(filename)[0]  # z.B. "page_01"

                print(f"\n📄 processing: {image_name}")

                self.text_recognizer.image_name = image_name
                self.text_recognizer.image_type = filename[-3:]
                self.text_recognizer.layout_analyze()
                self.text_recognizer.run_ocr()
                self.form_type = self.text_recognizer.form_type
                self.data = self.text_recognizer.optimized_ocr_results
                self.extract_meta_infos()
                self.extract_table_entries()
                self.save_extracted_data()

    def extract_meta_infos(self):
        """
        Matches HTR values ​​to the nearest OCR labels (keys) based on Y coordinate similarity and X alignment.
        Results are stored in self.matched_pairs as a dictionary: {OCR_key: HTR_value}
        """
        self.meta_infos = {}

        if self.meta_table_id:
            relevant_data = [entry for entry in self.data if entry.get("table_id") == self.meta_table_id]
        else:
            relevant_data = self.data

        def is_htr_side_of_ocr(ocr, htr, x_tolerance=10):
            o_x1, o_y1, o_x2, o_y2 = ocr["coordinates"]
            h_x1, h_y1, h_x2, h_y2 = htr["coordinates"]

            if [o_x1, o_y1, o_x2, o_y2] == [h_x1, h_y1, h_x2, h_y2]:
                return True


            y_overlap = min(o_y2, h_y2) - max(o_y1, h_y1)
            min_height = min(o_y2 - o_y1, h_y2 - h_y1)

            sufficient_vertical_overlap = y_overlap >= min_height * 0.5
            horizontal_relation = h_x1 >= o_x2 - x_tolerance

            return sufficient_vertical_overlap and horizontal_relation

        for i in range(1, len(relevant_data) - 1):
            pre_val = relevant_data[i - 1]
            current_val = relevant_data[i]
            post_val = relevant_data[i + 1]

            if pre_val["type"] == "OCR" and current_val["type"] == "HTR" and is_htr_side_of_ocr(pre_val,
                                                                                                current_val):
                self.meta_infos[pre_val["texts"]] = current_val["texts"]
            elif current_val["type"] == "OCR" and post_val["type"] == "HTR" and is_htr_side_of_ocr(current_val,
                                                                                                   post_val):
                self.meta_infos[current_val["texts"]] = post_val["texts"]

        self.extracted_data.append(self.meta_infos)

    def extract_tables_ids(self):
        """
        Determines the table_id of the metadata table and entry table based on self.meta_keys.
        The metadata table is identified by typical field names.
        """
        table_meta_counter = {}
        table_ids = set()

        for entry in self.data:
            tid = entry.get("table_id", -1)
            if tid == -1:
                continue

            table_ids.add(tid)
            if entry.get("texts") in self.meta_keys:
                table_meta_counter[tid] = table_meta_counter.get(tid, 0) + 1

        if not table_meta_counter:
            self.meta_table_id = -1
            self.entries_table_id = next(iter(table_ids), 1)
            return False

        self.meta_table_id = max(table_meta_counter, key=table_meta_counter.get)
        remaining_ids = table_ids - {self.meta_table_id}

        self.entries_table_id = next(iter(remaining_ids), self.meta_table_id)  # fallback, falls nur eine Tabelle

        return True

    def extract_table_entries(self):
        """
        Initiates the row-by-row extraction of the table contents.
        The extraction logic is selected based on the detected form type.
        """
        if not self.form_type:
            print("⚠️ No form type recognized. Table cannot be extracted.")
            return
        if self.form_type.startswith("form_1"):
            self.extract_table_entries_form1()
        elif self.form_type.startswith("form_2"):
            self.extract_table_entries_form2()
        elif self.form_type.startswith("form_3"):
            self.extract_table_entries_form_3()
        else:
            print(f"⚠️ No extraction logic implemented for {self.form_type}.")

    def extract_table_entries_form1(self):

        ocr_entries = [
            e for e in self.data
            if e["type"] == "OCR" and e.get("table_id") == self.entries_table_id
        ]
        htr_entries = [
            e for e in self.data
            if e["type"] == "HTR" and e.get("table_id") == self.entries_table_id
        ]

        start_entries = [e for e in ocr_entries if "datum" in e["texts"].lower()]
        end_entries = [e for e in ocr_entries if "abgenommen" in e["texts"].lower()
                       and "bewertet" in e["texts"].lower()
                       and "leistungsberechtigte" in e["texts"].lower()]
        right_entries = [e for e in ocr_entries if "handzeichen" in e["texts"].lower() and "pers" in e["texts"].lower()]

        table_blocks = []

        for i in range(min(len(start_entries), len(end_entries), len(right_entries))):
            start = start_entries[i]
            end = end_entries[i]
            right = right_entries[i]

            x1 = start["coordinates"][0]
            y1 = start["coordinates"][1]
            x2 = right["coordinates"][2]
            y2 = end["coordinates"][3]

            block = {
                "block_id": f"{i + 1}",
                "block_coordinates": (x1, y1, x2, y2),
                "OCR": [],
                "HTR": []
            }

            for fake_entry in ocr_entries + htr_entries:
                ex1, ey1, ex2, ey2 = fake_entry["coordinates"]
                if x1 <= ex1 <= x2 and y1 <= ey1 <= y2:
                    entry = {"texts": fake_entry["texts"], "type": fake_entry["type"],
                             "coordinates": fake_entry["coordinates"]}
                    if fake_entry["type" ] == "OCR":
                        block["OCR"].append(entry)
                    else:
                        block["HTR"].append(entry)
            table_blocks.append(block)


        def find_entry(block_content, label, mode="below"):
            """
            Findet den passendsten HTR-Eintrag zu einem OCR-Label innerhalb eines Blocks.
            Positionelles Matching (below, right, same) + Ranking nach Distanz.
            """

            ocr_content = block_content["OCR"]
            htr_content = block_content["HTR"]
            label_entry = next((e for e in ocr_content if label.lower() in e["texts"].lower() and e["type"] == "OCR"),
                               None)
            if not label_entry:
                return None


            ox1, oy1, ox2, oy2 = label_entry["coordinates"]

            for htr in htr_content:
                hx1, hy1, hx2, hy2 = htr["coordinates"]

                if mode == "below" and ox1 <= hx1 and oy1 < hy1 and ox2 >= hx2 and oy2 < hy2 and abs(oy2 - hy1) < 10:
                    return htr["texts"]

                elif mode == "right" and ox1 < hx1 and oy1 <= hy1 and ox2 < hx2 and oy2 >= hy2 and abs(ox2 - hx1) < 10:
                    return htr["texts"]

                elif mode == "same" and ox1 == hx1 and oy1 == hy1 and ox2 == hx2 and oy2 == hy2:
                    return htr["texts"]

            return None

        self.table_entries = []
        for block in table_blocks:
            entry_dict = {
                "ID": block["block_id"],
                "am (Datum) um (Uhrzeit)": find_entry(block,"Datum", "below"),
                "Dauer in FLS": find_entry(block,"dauer in fls", "below"),
                "Allein": find_entry(block,"allein", "below"),
                "In der Gruppe": find_entry(block,"gruppe", "below"),
                "Die Mitarbeiter haben mir geholfen.": find_entry(block,"mitarbeiter", "right"),
                "Ich war mit der Hilfe zufrieden.": find_entry(block,"hilfe", "right"),
                "Unterschrift LB": find_entry(block,"Unterschrift LB", "below"),
                "Zielbezogene Unterstützungsleistung gem. Teilhabeplan, Maßnahmedarstellung und Ort der Leistungserbringung:": find_entry(block,
                    "unterstützungsleistung", "same"),
                "Der Leistungsberechtigte hat die Leistung abgenommen und bewertet": find_entry(block,"abgenommen", "right"),
                "Wenn nicht, weil:": find_entry(block,"weil", "same"),
                "Handzeichen LE+ Pers.Nr.": find_entry(block,"handzeichen", "below")
            }
            self.table_entries.append(entry_dict)
        self.extracted_data.append(self.table_entries)

    def extract_table_entries_form2(self):
        """
        Extracts structured content from the main table of form_2_1 (table_id = 1).
        Falls für eine erwartete Spalte kein Wert gefunden wird, wird None gespeichert.
        """
        table_cells = [x for x in self.data if x.get("table_id") == self.entries_table_id]

        expected_columns = [
            "Datum", "Uhrzeit von-bis", "abgenommen", "bewertet",
            "Wenn nicht, weil", "Unterschrift LE",
            "Ich bin mit der Leistung zufrieden", "Unterschrift LB"
        ]

        expected_columns_data = {}
        for cell in table_cells:
            if cell["type"] == "OCR":
                text = cell["texts"]
                coordinates = cell["coordinates"]
                for col_name in expected_columns:
                    similarity = difflib.SequenceMatcher(None, text.lower(), col_name.lower()).ratio()
                    if similarity > 0.99:
                        expected_columns_data[col_name] = coordinates
                        break

        column_centers = {
            key: (coords[0] + coords[2]) // 2
            for key, coords in expected_columns_data.items()
        }

        if "Datum" not in expected_columns_data:
            print("❌ Spaltenüberschrift 'Datum' nicht gefunden.")
            return

        header_y = expected_columns_data["Datum"][1]
        htr_data = [x for x in table_cells if x["type"] == "HTR" and x["coordinates"][1] > header_y]

        row_clusters = defaultdict(list)
        y_tolerance = 12
        row_refs = []

        for item in htr_data:
            y1 = item["coordinates"][1]
            matched_y = None
            for known_y in row_refs:
                if abs(known_y - y1) <= y_tolerance:
                    matched_y = known_y
                    break
            if matched_y is not None:
                row_clusters[matched_y].append(item)
            else:
                row_refs.append(y1)
                row_clusters[y1].append(item)

        self.table_entries = []
        row_id = 1

        for y, cells in sorted(row_clusters.items()):
            row_dict = {"ID": row_id}
            row_id += 1

            for col in expected_columns:
                row_dict[col] = None

            for cell in cells:
                x_center = (cell["coordinates"][0] + cell["coordinates"][2]) // 2
                best_match = None
                best_dist = float("inf")
                for col, col_x in column_centers.items():
                    dist = abs(col_x - x_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = col
                if best_match:
                    if row_dict[best_match] is None:
                        row_dict[best_match] = cell["texts"]
                    else:
                        row_dict[best_match] += " " + cell["texts"]

            self.table_entries.append(row_dict)

        self.extracted_data.append(self.table_entries)

    def extract_table_entries_form_3(self):
        expected_columns = [
            "Datum", "FLS", "abgenommen", "bewertet",
            "allein", "Gruppe", "Wenn nicht, weil:", "Handzeichen LE",
            "Der Mitarbeiter hat mir geholfen/ Ich war mit der Hilfe: zufrieden"
        ]

        table_cells = [x for x in self.data if x.get("table_id") == self.entries_table_id]

        expected_columns_data = {}
        for cell in table_cells:
            if cell["type"] == "OCR":
                text = cell["texts"]
                coordinates = cell["coordinates"]
                for col_name in expected_columns:
                    similarity = difflib.SequenceMatcher(None, text.lower(), col_name.lower()).ratio()
                    if similarity > 0.90:
                        expected_columns_data[col_name] = coordinates
                        break

        column_centers = {
            key: (coords[0] + coords[2]) // 2
            for key, coords in expected_columns_data.items()
        }

        if "Datum" not in expected_columns_data and "Datum:" not in expected_columns_data:
            return False

        header_y = expected_columns_data["Datum"][1]

        htr_data = [
            x for x in table_cells if x["type"] == "HTR" and x["coordinates"][1] > header_y
        ]

        row_clusters = defaultdict(list)
        y_tolerance = 10
        row_refs = []

        for item in htr_data:
            y1 = item["coordinates"][1]
            matched_y = None
            for known_y in row_refs:
                if abs(known_y - y1) <= y_tolerance:
                    matched_y = known_y
                    break
            if matched_y is not None:
                row_clusters[matched_y].append(item)
            else:
                row_refs.append(y1)
                row_clusters[y1].append(item)

        self.table_entries = []
        row_id = 1
        for y, cells in sorted(row_clusters.items()):
            row_dict = {"ID": row_id}
            row_id += 1

            for col in expected_columns:
                row_dict[col] = None

            for cell in cells:
                x_center = (cell["coordinates"][0] + cell["coordinates"][2]) // 2
                best_match = None
                best_dist = float("inf")
                for col, col_x in column_centers.items():
                    dist = abs(col_x - x_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = col
                if best_match:
                    if row_dict[best_match] is None:
                        row_dict[best_match] = cell["texts"]
                    else:
                        row_dict[best_match] += " " + cell["texts"]

            self.table_entries.append(row_dict)
        self.extracted_data.append(self.table_entries)
        return True

    def save_extracted_data(self):
        """
        Saves the extracted metadata and table entries as a JSON file in the output folder.
        The file is named after the input image, e.g., ocr_form_1_1A_extracted.json
        """
        output_path = "../../output/ie"
        os.makedirs(output_path, exist_ok=True)

        filename = f"{self.image_name}_extracted.json"
        full_path = os.path.join(output_path, filename)
        output_data = {
                "Formtyp": self.form_type,
                "Metadaten": self.extracted_data[0],
                "Tätigkeitsdaten": self.extracted_data[1]
            }
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Extrahierte Daten gespeichert in: {full_path}")

info_extractor = InformationExtractor(input_image="form_3_1A.png")
info_extractor.run(rerun_ocr=False)