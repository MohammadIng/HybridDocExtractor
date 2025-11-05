import json
from src.text_recognition.image_text_recognition import ImageTextRecognizer


class FormHandler:

    def __init__(self, file_type="_standard"):
        self.file_type=file_type
        self.text_recognizer = ImageTextRecognizer(lp_score_thresh=0.5)

    def update_forms_data(self):
        results = {}
        forms = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2", ]
        for form_name in forms:
            full_form_name = f"form_{form_name}"
            self.text_recognizer.image_name = full_form_name
            self.text_recognizer.layout_analyze()

            json_path = f"../../output/lp/layout_parser_form_{form_name}.json"

            features = self.text_recognizer.layout_analyzer.extract_form_features(json_path)
            self.text_recognizer.run_ocr(margin=5)

            results[f"form_{form_name}"] = {
                "form_features": features,
                "form_type": "form_"+form_name,
                "form_content": self.text_recognizer.optimized_ocr_results

            }

        with open(f"../../input/forms/forms_data{self.file_type}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


handler = FormHandler(file_type="")
handler.update_forms_data()







