import json
import os
import re

import numpy as np
from matplotlib import pyplot as plt

from src.layout_analysis.image_layout_analysis import ImageLayoutAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class DLAEvaluator:

    def __init__(self,input_dir="../../eval_data/base_data/lp/", rerun_dla=False ):
        self.input_dir = input_dir
        self.rerun_dla = rerun_dla

    def rename_files_by_form_type(self,
                                  valid_exts=(".png", ".jpg", ".jpeg"),
                                  overwrite=False,
                                  dry_run=False):
        files = [f for f in os.listdir(self.input_dir)
                 if os.path.isfile(os.path.join(self.input_dir, f))
                 and os.path.splitext(f)[1].lower() in valid_exts]

        renames = []

        for i, file_name in enumerate(files):
            src_path = os.path.join(self.input_dir, file_name)
            base, ext = os.path.splitext(file_name)
            image_name = base
            image_type = ext.lstrip(".").lower()

            analyzer = ImageLayoutAnalyzer(image_name=image_name,
                                           image_type=image_type,
                                           to_show_results=False,
                                           write_label_text=False,
                                           run_dla=self.rerun_dla,
                                           folder_path=self.input_dir)
            analyzer.analyze()

            form_type = (analyzer.str_form_data or {}).get("form_type", "unknown")
            new_base = f"{form_type}({str(i)})"
            dst_path = os.path.join(self.input_dir, f"{new_base}{ext}")

            if os.path.exists(dst_path) and not overwrite:
                i = 1
                while True:
                    candidate = os.path.join(self.input_dir, f"{new_base}_{i}{ext}")
                    if not os.path.exists(candidate):
                        dst_path = candidate
                        break
                    i += 1

            renames.append((src_path, dst_path))

            if dry_run:
                print(f"↪️  DRY-RUN: {os.path.basename(src_path)}  →  {os.path.basename(dst_path)}")
            else:
                if overwrite and os.path.exists(dst_path):
                    os.remove(dst_path)
                os.rename(src_path, dst_path)
                print(f"✅ renamed: {os.path.basename(src_path)}  →  {os.path.basename(dst_path)}")

        return renames

    def run_evaluation(self,
                       valid_exts=(".png", ".jpg", ".jpeg")):

        results = None
        output_file = "../../output/eval/eval_dla_results.json"

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)

        def harmonic_weighted(p_form, p_tab, p_cell):
            w_f, w_t, w_c = 0.5, 0.3, 0.2
            val = (w_f / p_form if p_form > 0 else float("inf")) \
                  + (w_t / p_tab if p_tab > 0 else float("inf")) \
                  + (w_c / p_cell if p_cell > 0 else float("inf"))
            return (w_f + w_t + w_c) / val if val > 0 else 0.0

        if not results:
            results = {}
            files = [f for f in os.listdir(self.input_dir)
                     if os.path.isfile(os.path.join(self.input_dir, f))
                     and os.path.splitext(f)[1].lower() in valid_exts]

            if not files:
                print("ℹ️ no files found")
                return

            grund_truth = {
                "form_1_1": {"num_tables": 2, "num_cells_total": 75},
                "form_1_2": {"num_tables": 2, "num_cells_total": 85},
                "form_2_1": {"num_tables": 2, "num_cells_total": 79},
                "form_2_2": {"num_tables": 1, "num_cells_total": 94},
                "form_3_1": {"num_tables": 2, "num_cells_total": 87},
                "form_3_2": {"num_tables": 1, "num_cells_total": 129}
            }

            y_true, y_pred = [], []
            form_metrics = {
                form: {"y_true": [], "y_pred": [], "count": 0, "tables": 0, "cells": 0}
                for form in grund_truth.keys()
            }

            for file_name in files:
                image_name, ext = os.path.splitext(file_name)
                image_type = ext.lstrip(".").lower()

                analyzer = ImageLayoutAnalyzer(
                    image_name=image_name,
                    image_type=image_type,
                    to_show_results=self.rerun_dla,
                    write_label_text=self.rerun_dla,
                    folder_path=self.input_dir,
                    run_dla=self.rerun_dla,
                    preprocessing_image=self.rerun_dla,
                )
                analyzer.analyze()

                real_form_type = "unknown"
                match = re.search(r"[_\-]?(\w+)", image_name.lower())
                if match:
                    real_form_type = match.group(1)

                extracted_form_type = analyzer.str_form_data.get("form_type", "unknown")

                y_true.append(real_form_type)
                y_pred.append(extracted_form_type)

                if real_form_type in form_metrics:
                    form_metrics[real_form_type]["y_true"].append(real_form_type)
                    form_metrics[real_form_type]["y_pred"].append(extracted_form_type)
                    form_metrics[real_form_type]["count"] += 1

                    form_features = analyzer.extract_form_features(analyzer.json_output_path)
                    form_metrics[real_form_type]["tables"] += form_features.get("num_tables", 0)
                    form_metrics[real_form_type]["cells"] += form_features.get("num_cells_total", 0)

            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            print(f"All Types: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


            for form_type, data in form_metrics.items():
                if not data["y_true"]:
                    continue
                else:
                    print("Evaluation in Progress")

                y_true_bin = [1 if y == form_type else 0 for y in y_true]
                y_pred_bin = [1 if y == form_type else 0 for y in y_pred]

                cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                p_cls = precision_score(data["y_true"], data["y_pred"],
                                        pos_label=form_type, average="binary", zero_division=0)
                r_cls = recall_score(data["y_true"], data["y_pred"],
                                     pos_label=form_type, average="binary", zero_division=0)
                f_cls = f1_score(data["y_true"], data["y_pred"],
                                 pos_label=form_type, average="binary", zero_division=0)

                avg_tables = data["tables"] / data["count"] if data["count"] else 0
                avg_cells = data["cells"] / data["count"] if data["count"] else 0

                gt_tables = grund_truth[form_type]["num_tables"]
                gt_cells = grund_truth[form_type]["num_cells_total"]

                # Tables
                tp_tables = min(avg_tables, gt_tables)
                fp_tables = max(avg_tables - gt_tables, 0)
                fn_tables = max(gt_tables - avg_tables, 0)
                p_tables = tp_tables / (tp_tables + fp_tables) if (tp_tables + fp_tables) > 0 else 0
                r_tables = tp_tables / (tp_tables + fn_tables) if (tp_tables + fn_tables) > 0 else 0
                f_tables = 2 * p_tables * r_tables / (p_tables + r_tables) if (p_tables + r_tables) > 0 else 0

                # Cells
                tp_cells = min(avg_cells, gt_cells)
                fp_cells = max(avg_cells - gt_cells, 0)
                fn_cells = max(gt_cells - avg_cells, 0)
                p_cells = tp_cells / (tp_cells + fp_cells) if (tp_cells + fp_cells) > 0 else 0
                r_cells = tp_cells / (tp_cells + fn_cells) if (tp_cells + fn_cells) > 0 else 0
                f_cells = 2 * p_cells * r_cells / (p_cells + r_cells) if (p_cells + r_cells) > 0 else 0

                # All
                p_total = harmonic_weighted(p_cls, p_tables, p_cells)
                r_total = harmonic_weighted(r_cls, r_tables, r_cells)
                f_total = 2 * p_total * r_total / (p_total + r_total) if (p_total + r_total) > 0 else 0

                results[form_type] = {
                    # form classification
                    "TP_form": int(tp), "FP_form": int(fp), "FN_form": int(fn), "TN_form": int(tn),
                    "P_form": float(p_cls), "R_form": float(r_cls), "F1_form": float(f_cls),

                    # tables
                    "TP_tables": int(tp_tables), "FP_tables": int(fp_tables), "FN_tables": int(fn_tables),
                    "P_tables": float(p_tables), "R_tables": float(r_tables), "F1_tables": float(f_tables),

                    # cells
                    "TP_cells": int(tp_cells), "FP_cells": int(fp_cells), "FN_cells": int(fn_cells),
                    "P_cells": float(p_cells), "R_cells": float(r_cells), "F1_cells": float(f_cells),

                    # total
                    "P_total": float(p_total), "R_total": float(r_total), "F1_total": float(f_total),
                }

                print(f"\n=== {form_type} ===")
                print(f"Formularklassifikation   → TP={tp}, FP={fp}, FN={fn}, TN={tn} ")
                print(f"                         → P={p_cls:.3f}, R={r_cls:.3f}, F1={f_cls:.3f}")
                print(f"Tabellen: TP={tp_tables}, FP={fp_tables}, FN={fn_tables} → "
                      f"P={p_tables:.3f}, R={r_tables:.3f}, F1={f_tables:.3f}")
                print(f"Zellen:   TP={tp_cells}, FP={fp_cells}, FN={fn_cells} → "
                      f"P={p_cells:.3f}, R={r_cells:.3f}, F1={f_cells:.3f}")
                print(f"Gesamt:   P={p_total:.3f}, R={r_total:.3f}, F1={f_total:.3f}")

            print(results)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\n✅ results of eval_dla saves  in: {output_file}")

        # --- plots ---
        self.plot_evaluation_results(results)

    def plot_evaluation_results(self, results):
        """
           Display results from evaluation as line/scatter plots. - 3 plots:
           Precision, Recall, F1 - X-axis: form types + 'All' - Y-axis: values (0–100%)
        """

        form_types = list(results.keys())
        x = np.arange(len(form_types) + 1)

        # Precision
        P_form = [results[ft]["P_form"] for ft in form_types]
        P_tables = [results[ft]["P_tables"] for ft in form_types]
        P_cells = [results[ft]["P_cells"] for ft in form_types]
        P_total = [results[ft]["P_total"] for ft in form_types]

        # Recall
        R_form = [results[ft]["R_form"] for ft in form_types]
        R_tables = [results[ft]["R_tables"] for ft in form_types]
        R_cells = [results[ft]["R_cells"] for ft in form_types]
        R_total = [results[ft]["R_total"] for ft in form_types]

        # F1
        F_form = [results[ft]["F1_form"] for ft in form_types]
        F_tables = [results[ft]["F1_tables"] for ft in form_types]
        F_cells = [results[ft]["F1_cells"] for ft in form_types]
        F_total = [results[ft]["F1_total"] for ft in form_types]

        form_types_all = form_types + ["All"]

        P_form.append(np.mean(P_form))
        P_tables.append(np.mean(P_tables))
        P_cells.append(np.mean(P_cells))
        P_total.append(np.mean(P_total))

        R_form.append(np.mean(R_form))
        R_tables.append(np.mean(R_tables))
        R_cells.append(np.mean(R_cells))
        R_total.append(np.mean(R_total))

        F_form.append(np.mean(F_form))
        F_tables.append(np.mean(F_tables))
        F_cells.append(np.mean(F_cells))
        F_total.append(np.mean(F_total))

        # ---------- Precision ----------
        self.plot_metric(
            x,
            form_types_all,
            {
                "FormType": P_form,
                "Tabellen": P_tables,
                "Zellen": P_cells,
                "Gesamt": P_total,
            },
            title="Precision pro Formtyp",
            y_label="Precision (%)"
        )

        # ---------- Recall ----------
        self.plot_metric(
            x,
            form_types_all,
            {
                "FormType": R_form,
                "Tabellen": R_tables,
                "Zellen": R_cells,
                "Gesamt": R_total,
            },
            title="Recall pro Formtyp",
            y_label="Recall (%)"
        )

        # ---------- F1 ----------
        self.plot_metric(
            x,
            form_types_all,
            {
                "FormType": F_form,
                "Tabellen": F_tables,
                "Zellen": F_cells,
                "Gesamt": F_total,
            },
            title="F1-Score pro Formtyp",
            y_label="F1-Score (%)"
        )
        # network diagram
        # ---------- Precision ----------
        self.plot_metric_network_diagram(
            # x,
            form_types_all,
            {
                "FormType": P_form,
                "Tabellen": P_tables,
                "Zellen": P_cells,
                "Gesamt": P_total,
            },
            title="Precision pro Formtyp",
            # y_label="Precision (%)"
        )

        # ---------- Recall ----------
        self.plot_metric_network_diagram(
            # x,
            form_types_all,
            {
                "FormType": R_form,
                "Tabellen": R_tables,
                "Zellen": R_cells,
                "Gesamt": R_total,
            },
            title="Recall pro Formtyp",
            # y_label="Recall (%)"
        )

        # ---------- F1 ----------
        self.plot_metric_network_diagram(
            # x,
            form_types_all,
            {
                "FormType": F_form,
                "Tabellen": F_tables,
                "Zellen": F_cells,
                "Gesamt": F_total,
            },
            title="F1-Score pro Formtyp",
            # y_label="F1-Score (%)"
        )


    @staticmethod
    def plot_metric(x, form_types, values_dict,
                    title="Metrik pro Formtyp", y_label="Wert (%)", y_margin=5):
        """
        Plot line/scatter metrics per form type.
        Y-Axis extends slightly above 100%, but tick labels are clipped to max 100%.
        """

        plt.figure(figsize=(12, 6))

        markers = ["o", "s", "^", "D", "x", "*"]
        line_styles = ["-", "-", "-", "--", ":", "-."]

        all_values = []
        for i, (label, values) in enumerate(values_dict.items()):
            percent_values = [v * 100 for v in values]
            all_values.extend(percent_values)

            plt.plot(x, percent_values,
                     marker=markers[i % len(markers)],
                     linestyle=line_styles[i % len(line_styles)],
                     label=label)

        max_val = max(all_values) if all_values else 100
        upper_limit = max(100, max_val + y_margin)  # z. B. 105 oder 110

        plt.xticks(x, form_types, rotation=45)
        plt.ylim(0, upper_limit)

        yticks = plt.gca().get_yticks()
        yticks = [yt for yt in yticks if yt <= 100]
        plt.gca().set_yticks(yticks)

        plt.title(title)
        plt.xlabel("Formtypen")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(f"../../output/eval/lp_{title.replace(' ', '_')}.png")
        plt.show()

    @staticmethod
    def plot_metric_network_diagram(form_types, values_dict,
                    title="Metrik pro Formtyp"):
        """
        Plot Radar/Netzdiagramm für die Metriken pro Formtyp.
        """

        # Anzahl der Achsen = Anzahl Formtypen
        labels = form_types
        num_vars = len(labels)

        # Winkel für jede Achse berechnen
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")

        for label, values in values_dict.items():
            percent_values = [v * 100 for v in values]
            percent_values += percent_values[:1]

            ax.plot(angles, percent_values, label=label, linewidth=2)
            # ax.fill(angles, percent_values, alpha=0.25)

        # Achsenbeschriftungen
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        # Y-Achse (0–100%)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])

        plt.title(title, size=15, y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()

        plt.savefig(f"../../output/eval/lp_{title.replace(' ', '_')}_nw.png")
        plt.show()


evaluator = DLAEvaluator(rerun_dla=False)
# evaluator.rename_files_by_form_type()
evaluator.run_evaluation()
