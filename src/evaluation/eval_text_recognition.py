import difflib
import os
import re

import jiwer
import json
import editdistance
import numpy as np
from matplotlib import pyplot as plt

from src.text_recognition.image_text_recognition import ImageTextRecognizer
import tkinter as tk


class TREvaluator:
    def __init__(self, forms_gt_file="../../input/forms/forms_data_standard.json",
                 eval_output_dir="../../eval_data/ocr",
                 input_dir="../../eval_data/base_data/tr_high_quality/",
                 re_eval=False):
        self.eval_dir = "../../eval_data/ocr"
        self.input_dir = input_dir
        self.forms_gt_file = forms_gt_file
        self.eval_output_dir = eval_output_dir
        self.ocr_results = None
        self.gt_fields = None
        self.re_eval = re_eval
        os.makedirs(self.eval_output_dir, exist_ok=True)

        # GT-Daten laden
        with open(self.forms_gt_file, "r", encoding="utf-8") as f:
            self.forms_gt = json.load(f)

    @staticmethod
    def ask_user_for_gt(pred_text):
        def on_ok():
            nonlocal gt_text
            gt_text = entry.get().strip()
            root.destroy()

        def on_cancel():
            nonlocal gt_text
            gt_text = None
            root.destroy()

        gt_text = ""
        root = tk.Tk()
        root.title("HTR Ground Truth")

        win_width, win_height = 400, 150

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        x_pos = int((screen_width - win_width) / 2)
        y_pos = int((screen_height - win_height) / 2)

        root.geometry(f"{win_width}x{win_height}+{x_pos}+{y_pos}")
        root.resizable(False, False)  # fixierte Größe

        tk.Label(root, text=f"Bitte Ground Truth für '{pred_text}' eingeben:", font=("Arial", 12)).pack(pady=10)

        entry = tk.Entry(root, width=40, font=("Consolas", 14))
        entry.pack(padx=10, pady=10)
        entry.insert(0, pred_text)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="OK", command=on_ok, width=12).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=12).pack(side="left", padx=5)

        root.mainloop()
        return gt_text

    @staticmethod
    def cer(gt, pred):
        if not gt:
            return None
        return editdistance.eval(gt, pred) / len(gt)

    @staticmethod
    def wer(gt, pred):
        if not gt:
            return None
        return jiwer.wer(gt, pred)

    def get_ocr_gt_text(self, pred_text, similarity_threshold=0.8):
        candidates = [item["texts"] for item in self.gt_fields if "texts" in item]
        matches = difflib.get_close_matches(pred_text, candidates, n=1, cutoff=similarity_threshold)
        return matches[0] if matches else None

    def extract_ground_truth_as_image(self, image_path, form_type):
        """
        OCR-GT aus forms_data_standard.json, HTR-GT bleibt leer.
        Pro Bild ein JSON mit GT, Prediction, CER/WER.
        """
        image_name, ext = os.path.splitext(os.path.basename(image_path))
        image_type = ext.lstrip(".").lower()
        output_file = os.path.join(self.eval_output_dir, f"eval_{image_name}.json")

        if not os.path.exists(output_file) or self.re_eval:
            print(f"➡️  Starte Evaluation: {output_file}")

            recognizer = ImageTextRecognizer(folder_path=self.input_dir,
                                             image_name=image_name,
                                             image_type=image_type,
                                             to_run_ocr=True,
                                             to_run_dla=True,
                                             form_type=form_type
                                             )
            recognizer.layout_analyze()
            recognizer.run_ocr()

            self.gt_fields = self.forms_gt.get(form_type, {}).get("form_content", {})
            self.ocr_results = recognizer.optimized_ocr_results

            results = []

            for item in self.ocr_results:
                pred_text = item["texts"]
                text_type = item["type"]
                if text_type == "OCR":
                    gt_text = self.get_ocr_gt_text(pred_text)
                else:
                    gt_text = self.ask_user_for_gt(pred_text)

                if gt_text == "" or gt_text == " ":
                    results.append({
                        "text_type": text_type,
                        "pred_text": "cb",
                        "gt_text": "cb",
                        "CER": self.cer(gt_text, pred_text),
                        "WER": self.wer(gt_text, pred_text)
                    })
                elif gt_text:
                    results.append({
                        "text_type": text_type,
                        "pred_text": pred_text,
                        "gt_text": gt_text,
                        "CER": self.cer(gt_text, pred_text),
                        "WER": self.wer(gt_text, pred_text)
                    })


            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"image_name": image_name, "results": results}, f, indent=2, ensure_ascii=False)

            print(f"✔ GT for {image_name} saved in: {output_file}")
            return results
        else:
            print(f"⚠️ Eval file already exists, skip: {output_file}")
            return None

    def create_ground_truth(self, valid_exts=(".png", ".jpg", ".jpeg")):
        files = [f for f in os.listdir(self.input_dir)
                 if os.path.splitext(f)[1].lower() in valid_exts]

        if not files:
            print("⚠️ no image files found")
            return

        for file in files:
            image_path = os.path.join(self.input_dir, file)

            match = re.match(r"(form_\d+_\d+)", file.lower())
            if match:
                form_type = match.group(1)
            else:
                print(f"⚠️ no valid form_type found: {file}")
                continue

            self.extract_ground_truth_as_image(image_path, form_type)

    def evaluate(self, form_type="all"):
        """
        Liest alle eval_*.json Dateien und berechnet CER/WER + Genauigkeit (%) getrennt für OCR und HTR.
        """
        if form_type != "all":
            files = [f for f in os.listdir(self.eval_dir) if f.endswith(".json") and f.__contains__(form_type)]
        else:
            files = [f for f in os.listdir(self.eval_dir) if f.endswith(".json")]

        if not files:
            print("⚠️ Keine JSON-Dateien gefunden.")
            return None

        summary = {"OCR": [], "HTR": []}

        for file in files:
            with open(os.path.join(self.eval_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            image_name = data.get("image_name", file)
            results = data.get("results", [])

            ocr_cer, ocr_wer, htr_cer, htr_wer = [], [], [], []

            for item in results:
                cer = item.get("CER")
                wer = item.get("WER")
                if cer is None or wer is None:
                    continue

                if item["text_type"] == "OCR":
                    ocr_cer.append(cer)
                    ocr_wer.append(wer)
                elif item["text_type"] == "HTR":
                    htr_cer.append(cer)
                    htr_wer.append(wer)

            # pro Bild Mittelwerte
            ocr_cer_mean = np.mean(ocr_cer) if ocr_cer else None
            ocr_wer_mean = np.mean(ocr_wer) if ocr_wer else None
            htr_cer_mean = np.mean(htr_cer) if htr_cer else None
            htr_wer_mean = np.mean(htr_wer) if htr_wer else None

            # Genauigkeit in %
            ocr_acc_char = (1 - float(ocr_cer_mean)) * 100 if ocr_cer_mean is not None else None
            ocr_acc_word = (1 - float(ocr_wer_mean)) * 100 if ocr_wer_mean is not None else None
            htr_acc_char = (1 - float(htr_cer_mean)) * 100 if htr_cer_mean is not None else None
            htr_acc_word = (1 - float(htr_wer_mean)) * 100 if htr_wer_mean is not None else None

            summary["OCR"].append({
                "image": image_name,
                "CER": ocr_cer_mean, "WER": ocr_wer_mean,
                "CA": ocr_acc_char, "WA": ocr_acc_word
            })
            summary["HTR"].append({
                "image": image_name,
                "CER": htr_cer_mean, "WER": htr_wer_mean,
                "CA": htr_acc_char, "WA": htr_acc_word
            })

            print(f"📄 {image_name}: "
                  f"OCR → CER={ocr_cer_mean:.3f} ({ocr_acc_char:.1f}%), "
                  f"WER={ocr_wer_mean:.3f} ({ocr_acc_word:.1f}%) | "
                  f"HTR → CER={htr_cer_mean:.3f} ({htr_acc_char:.1f}%), "
                  f"WER={htr_wer_mean:.3f} ({htr_acc_word:.1f}%)")

        # Gesamtauswertung
        overall = {
            "OCR": {
                "CER": np.nanmean([min(e["CER"], 1) for e in summary["OCR"] if e["CER"] is not None]),
                "WER": np.nanmean([min(e["WER"], 1) for e in summary["OCR"] if e["WER"] is not None]),
            },
            "HTR": {
                "CER": np.nanmean([min(e["CER"], 1) for e in summary["HTR"] if e["CER"] is not None]),
                "WER": np.nanmean([min(e["WER"], 1) for e in summary["HTR"] if e["WER"] is not None]),
            }
        }
        overall["OCR"]["CA"] = (1 - overall["OCR"]["CER"]) * 100 if overall["OCR"]["CER"] is not None else None
        overall["OCR"]["WA"] = (1 - overall["OCR"]["WER"]) * 100 if overall["OCR"]["WER"] is not None else None
        overall["HTR"]["CA"] = (1 - overall["HTR"]["CER"]) * 100 if overall["HTR"]["CER"] is not None else None
        overall["HTR"]["WA"] = (1 - overall["HTR"]["WER"]) * 100 if overall["HTR"]["WER"] is not None else None

        print(f"=== 📊 Gesamtauswertung für {form_type}===")
        print(f"OCR → CER={overall['OCR']['CER']:.3f} ({overall['OCR']['CA']:.1f}%), "
              f"WER={overall['OCR']['WER']:.3f} ({overall['OCR']['WA']:.1f}%)")
        print(f"HTR → CER={overall['HTR']['CER']:.3f} ({overall['HTR']['CA']:.1f}%), "
              f"WER={overall['HTR']['WER']:.3f} ({overall['HTR']['WA']:.1f}%)")

        return {"per_image": summary, "overall": overall}

    def evaluate_form_type_1_1(self):
        self.evaluate(form_type="form_1_1")

    def evaluate_form_type_1_2(self):
        self.evaluate(form_type="form_1_2")

    def evaluate_form_type_2_1(self):
        self.evaluate(form_type="form_2_1")

    def evaluate_form_type_2_2(self):
        self.evaluate(form_type="form_2_2")

    def evaluate_form_type_3_1(self):
        self.evaluate(form_type="form_3_1")

    def evaluate_form_type_3_2(self):
        self.evaluate(form_type="form_3_2")

    def evaluate_all_types(self):
        self.evaluate(form_type="all")

    def evaluate_and_plot(self):
        """
        Performs the evaluation for all shape types and plots the results for OCR and HTR.
        """
        form_types = ["form_1_1", "form_1_2", "form_2_1", "form_2_2", "form_3_1", "form_3_2", "all"]
        results = {}

        for ft in form_types:
            eval_result = self.evaluate(form_type=ft)
            if eval_result:
                results[ft] = eval_result["overall"]

        if not results:
            print("⚠️ Keine Ergebnisse vorhanden – bitte zuerst create_ground_truth() ausführen.")
            return

        x = list(results.keys())

        # OCR-Werte extrahieren
        ocr_cer = [results[ft]["OCR"]["CER"] for ft in x]
        ocr_wer = [results[ft]["OCR"]["WER"] for ft in x]
        ocr_ca = [results[ft]["OCR"]["CA"] for ft in x]
        ocr_wa = [results[ft]["OCR"]["WA"] for ft in x]

        # HTR-Werte extrahieren
        htr_cer = [results[ft]["HTR"]["CER"] for ft in x]
        htr_wer = [results[ft]["HTR"]["WER"] for ft in x]
        htr_ca = [results[ft]["HTR"]["CA"] for ft in x]
        htr_wa = [results[ft]["HTR"]["WA"] for ft in x]

        # ---------- Plot 1: CER & WER ----------
        plt.figure(figsize=(12, 6))
        plt.plot(x, ocr_cer, marker="o", label="OCR CER")
        plt.plot(x, ocr_wer, marker="s", label="OCR WER")
        plt.plot(x, htr_cer, marker="^", label="HTR CER")
        plt.plot(x, htr_wer, marker="D", label="HTR WER")

        # Werte neben die Punkte schreiben
        for xi, yi in zip(x, ocr_cer):
            plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8)
        for xi, yi in zip(x, ocr_wer):
            plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8)
        for xi, yi in zip(x, htr_cer):
            plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8)
        for xi, yi in zip(x, htr_wer):
            plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8)

        plt.title("CER & WER für OCR und HTR")
        plt.xlabel("Formtypen")
        plt.ylabel("Fehlerrate (0–1)")
        plt.ylim(-0.1, 1.10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../../output/eval/tr_OCR_HTR_cer_wer.png")
        plt.show()

        # ---------- Plot 2: CA & WA ----------
        plt.figure(figsize=(12, 6))
        plt.plot(x, ocr_ca, marker="o", label="OCR CA (%)")
        plt.plot(x, ocr_wa, marker="s", label="OCR WA (%)")
        plt.plot(x, htr_ca, marker="^", label="HTR CA (%)")
        plt.plot(x, htr_wa, marker="D", label="HTR WA (%)")

        # Werte neben die Punkte schreiben
        for xi, yi in zip(x, ocr_ca):
            plt.text(xi, yi + 2, f"{yi:.1f}%", ha="center", fontsize=8)
        for xi, yi in zip(x, ocr_wa):
            plt.text(xi, yi + 2, f"{yi:.1f}%", ha="center", fontsize=8)
        for xi, yi in zip(x, htr_ca):
            plt.text(xi, yi + 2, f"{yi:.1f}%", ha="center", fontsize=8)
        for xi, yi in zip(x, htr_wa):
            plt.text(xi, yi + 2, f"{yi:.1f}%", ha="center", fontsize=8)

        plt.title("CA & WA für OCR und HTR")
        plt.xlabel("Formtypen")
        plt.ylabel("Genauigkeit (%)")
        plt.ylim(-5, 110)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../../output/eval/tr_OCR_HTR_ca_wa.png")
        plt.show()

evaluator = TREvaluator(re_eval=False)
evaluator.create_ground_truth()
# evaluator.evaluate_all_types()
# evaluator.evaluate_and_plot()

