import json
import os

import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from layoutparser import Rectangle, TextBlock
from matplotlib import patches
from src.preprocessing.iamge_preprocessing import ImagePreprocessor
from src.layout_analysis.tabular_line_enhancer import TabularLineEnhancer


class ImageLayoutAnalyzer:
    def __init__(self,  model_type="fast",
                        score_thresh=0.3,
                        image_name="page_00",
                        image_type="png",
                        to_show_results=False,
                        write_label_text=True,
                        preprocessing_image=False,
                        run_dla = False,
                        folder_path="../../input/",
                        form_type=None,
                        ):

        self.preprocessing_image = preprocessing_image
        self.run_dla = run_dla
        self.init_table_coordinates = None
        self.image_name = image_name
        self.image_type = image_type
        self.image_path = f"../../input/{self.image_name}.{self.image_type}"
        self.preprocessed_image_path = f"../../input/preprocessed/{self.image_name}.{self.image_type}"
        self.json_output_path = f"../../output/lp/layout_parser_{self.image_name}.json"
        self.image_output_path_std = f"../../output/lp/layout_parser_std_{self.image_name}.{self.image_type}"
        self.image_output_path = f"../../output/lp/layout_parser_{self.image_name}.{self.image_type}"
        self.folder_path = folder_path
        self.image_pp = None
        self.image_trocr = None
        self.image_preprocessor = None
        self.tabular_line_enhancer = None
        self.label_map = {0: "Text", 1: "Title", 2: "List", 3: "Figure", 4: "Table", 5: "Misc"}
        self.model_type = model_type
        self.score_thresh = score_thresh
        self.to_show_results = to_show_results
        self.write_label_text = write_label_text
        self.str_form_data = None
        self.layout = None
        self.form_features = None
        self.form_type = form_type


        if model_type == "mask":
            config_path = "../../models/layoutparser/mask_rcnn_X_101_32x8d_FPN_3x/config.yaml"
            model_path = "../../models/layoutparser/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth"
        elif model_type == "fast":
            config_path = "../../models/layoutparser/faster_rcnn_R_50_FPN_3x/config.yml"
            model_path = "../../models/layoutparser/faster_rcnn_R_50_FPN_3x/model_final.pth"
        else:
            raise ValueError("Ungültiger 'model_type'. Erlaubt: 'mask' oder 'fast'.")

        self.model = lp.Detectron2LayoutModel(
            config_path=config_path,
            model_path=model_path,
            label_map=self.label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        )


    def image_preprocessing(self):
        if not os.path.exists(self.preprocessed_image_path) or self.preprocessing_image:
            self.image_preprocessor = ImagePreprocessor(folder_path= self.folder_path, image_name=self.image_name, image_type=self.image_type, image_show=True)
            self.image_preprocessor.run_preprocessing()
            print(f"image was preprocessed: {self.image_name}")



    def analyze(self):
        """
            Performs layout analysis and returns the layout structure.
        """
        if not os.path.exists(self.json_output_path) or self.run_dla:
            self.image_preprocessing()
            self.image_pp = cv2.imread(self.preprocessed_image_path)
            self.tabular_line_enhancer = TabularLineEnhancer(overlay_color=[255, 0, 0])
            self.tabular_line_enhancer.enhance_lines(self.image_pp)
            self.init_table_coordinates = self.tabular_line_enhancer.detect_tables_coordinates()
            self.layout = self.model.detect(self.image_pp)
            self.visualize_layout_standard()
            self.optimize_layout_results()
            self.visualize_layout()
            self.save_results()
            self.assign_form_template()
        else:
            self.image_pp = cv2.imread(self.preprocessed_image_path)
            self.load_and_visualize_layout_from_json()
            try:
                with open(self.json_output_path, "r", encoding="utf-8") as f:
                    layout_data = json.load(f)
                self.init_table_coordinates = [
                    block for block in layout_data if block["type"] == "Table"
                ]
            except Exception as e:
                print(f"⚠️ Could not load init table coordinates from JSON: {e}")
                self.init_table_coordinates = []
            self.optimize_layout_results()
            self.assign_form_template()



    def optimize_layout_results(self, iou_threshold=0.0):
        """
            Optimizes the table frames of DLA recognition by comparing them with
            the table frames obtained by CV (without cells).
        """
        optimized_layout = []
        recognized_tables = []
        table_idx = 0
        for layout_element in self.layout:
            if layout_element.type.lower() == "table":
                table_idx += 1
                block_box = list(map(int, layout_element.coordinates))
                is_table_replaced = False
                for enhancer_table in self.init_table_coordinates:
                    enhancer_box = enhancer_table.get("coordinates")
                    if not enhancer_box:
                        continue

                    if self.is_table_to_replace(iou_threshold, block_box, enhancer_box):
                        is_table_replaced = True
                        x1 = enhancer_box[0]
                        y1 = enhancer_box[1]
                        x2 = enhancer_box[2]
                        y2 = enhancer_box[3]

                        new_block = TextBlock(
                            Rectangle(x_1=x1, y_1=y1, x_2=x2, y_2=y2),
                            type="Table",
                            text=table_idx
                        )
                        existing_coords = [b.coordinates for b in recognized_tables]
                        if new_block.coordinates not in existing_coords:
                            recognized_tables.append(new_block)
                            optimized_layout.append(new_block)
                            t_x1, t_y1, t_x2, t_y2 = enhancer_box
                            for cell in enhancer_table.get("cells", []):
                                c_x1, c_y1, c_x2, c_y2 = cell
                                if t_x1 <= c_x1 and t_y1 <= c_y1 and t_x2 >= c_x2 and t_y2 >= c_y2:
                                    cell_block  = TextBlock(
                                        Rectangle(x_1=c_x1, y_1=c_y1, x_2=c_x2, y_2=c_y2),
                                        type="Cell",
                                        text=table_idx
                                    )
                                    optimized_layout.append(cell_block)
                            break
                if not is_table_replaced:
                    x_1, y_1, x_2, y_2 = layout_element.coordinates
                    text_block = TextBlock(
                        Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2),
                        type="Table",
                        text=table_idx
                    )
                    optimized_layout.append(text_block)
                    optimized_layout.append(layout_element)
            elif layout_element.type.lower() == "figure":
                continue
            else:
                x_1, y_1, x_2, y_2 = layout_element.coordinates
                text_block = TextBlock(
                    Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2),
                    type="Text",
                    text=-1
                )
                optimized_layout.append(text_block)

        self.layout = optimized_layout


    def assign_form_template(self):
        """
        Compares the current form with saved types based on the characteristics
        and assigns the most likely form type.
        """
        current = self.extract_form_features(self.json_output_path)
        current_vector = np.array([
            current.get("num_tables", 0),
            current.get("avg_cells_per_table", 0),
            current.get("num_text_blocks", 0),
            current.get("num_title_blocks", 0),
            current.get("avg_columns_est", 0),
            current.get("avg_rows_est", 0),
            current.get("avg_cell_width", 0),
            current.get("avg_cell_height", 0),
            current.get("x_range_cells", 0),
            current.get("y_range_cells", 0)
        ])

        with open("../../input/forms/forms_data_standard.json", "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        if not self.form_type:
            min_score = float("inf")
            for form_id, form_data in reference_data.items():
                ref = form_data.get("form_features", {})
                ref_vector = np.array([
                    ref.get("num_tables", 0),
                    ref.get("avg_cells_per_table", 0),
                    ref.get("num_text_blocks", 0),
                    ref.get("num_title_blocks", 0),
                    ref.get("avg_columns_est", 0),
                    ref.get("avg_rows_est", 0),
                    ref.get("avg_cell_width", 0),
                    ref.get("avg_cell_height", 0),
                    ref.get("x_range_cells", 0),
                    ref.get("y_range_cells", 0)
                ])
                score = np.linalg.norm(current_vector - ref_vector)
                if score < min_score:
                    min_score = score
                    self.str_form_data = form_data
            self.form_type = self.str_form_data.get('form_type')
            print(f"✅ Assignment (weighted comparison): {self.str_form_data.get('form_type')} (distance: {min_score:.2f})")

        else:
            self.str_form_data = reference_data.get(self.form_type)

    def visualize_layout_standard(self):
        """
            Visualizes the recognized layout boxes in the default style.
        """
        viz = lp.draw_box(self.image_pp, self.layout, box_width=2)
        plt.figure(figsize=(12, 10))
        plt.imshow(viz)
        plt.axis("off")
        plt.title(f"LayoutParser ({self.model_type}) results")
        plt.savefig(self.image_output_path)
        plt.show()

    def visualize_layout(self):
        """
            Visualizes layout boxes with color-coded type differentiation.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(self.image_pp)
        colors = {
            "Text": "red", "Title": "green", "List": "orange",
            "Figure": "darkblue", "Table": "blue", "Misc": "gray", "Cell": "cyan"
        }
        for i, block in enumerate(self.layout):
            x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
            linewidth = 1
            if not block.type == "Cell" and self.write_label_text:
                ax.text(x1 - 50, y1 - 5, block.type, color=colors.get(block.type, "black"),
                    fontsize=10, weight="bold")
                linewidth = 2
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=linewidth,
                                     edgecolor=colors.get(block.type, "black"),
                                     facecolor='none')
            ax.add_patch(rect)

        ax.axis("off")
        if self.write_label_text:
                plt.title(f"LayoutParser ({self.model_type}) – farblich hervorgehoben")
        plt.savefig(self.image_output_path)
        if self.to_show_results:
            plt.show()
        else:
            plt.close()

    def save_results(self):
        """
            save the coordinates and types of all layout elements in JSON format.
        """
        layout_data = []
        tables = []

        for i, block in enumerate(self.layout):
            if block.type == "Table":
                x1, y1, x2, y2 = map(int, block.coordinates)
                tables.append({
                    "block_id": len(tables) + 1,
                    "type": "Table",
                    "coordinates": [x1, y1, x2, y2],
                    "cells": []
                })

        for block in self.layout:
            if block.type == "Cell":
                x1, y1, x2, y2 = map(int, block.coordinates)
                for table in tables:
                    t_x1, t_y1, t_x2, t_y2 = table["coordinates"]
                    if t_x1 <= x1 and t_y1 <= y1 and t_x2 >= x2 and t_y2 >= y2:
                        table["cells"].append([x1, y1, x2, y2])
                        break

        for block in self.layout:
            if block.type not in ["Table", "Cell"]:
                x1, y1, x2, y2 = map(int, block.coordinates)
                layout_data.append({
                    "block_id": len(layout_data) + 1,
                    "type": block.type,
                    "coordinates": [x1, y1, x2, y2]
                })

        layout_data.extend(tables)

        with open(self.json_output_path, "w", encoding="utf-8") as f:
            json.dump(layout_data, f, indent=2)

        print(f"✅ Results saved in JSON format as: {self.json_output_path}")

    def load_and_visualize_layout_from_json(self):
        """
            Loads a saved layout result from JSON and displays it visually.
            Supports tables with nested cells.
        """
        with open(self.json_output_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
        reconstructed_layout = []
        for block in layout_data:
            x1, y1, x2, y2 = block["coordinates"]
            rect = Rectangle(x_1=x1, y_1=y1, x_2=x2, y_2=y2)
            layout_block = TextBlock(rect, type=block["type"])
            reconstructed_layout.append(layout_block)

            if block["type"] == "Table" and "cells" in block:
                for cell_coords in block["cells"]:
                    c_x1, c_y1, c_x2, c_y2 = cell_coords
                    cell_rect = Rectangle(x_1=c_x1, y_1=c_y1, x_2=c_x2, y_2=c_y2)
                    cell_block = TextBlock(cell_rect, type="Cell")
                    reconstructed_layout.append(cell_block)

        self.layout = reconstructed_layout
        self.visualize_layout()

    @staticmethod
    def is_table_to_replace(iou_threshold, boxA, boxB):
        """
            Checks whether two rectangles overlap and whether the IoU is above the threshold.
            :param iou_threshold: Minimum IoU
            :param boxA: [x1, y1, x2, y2]
            :param boxB: [x1, y1, x2, y2]
            :return: True if IoU > threshold and overlap is present
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = boxAArea + boxBArea - interArea
        iou = interArea / float(union) if union > 0 else 0

        overlaps = not (
                boxA[2] < boxB[0] or
                boxA[0] > boxB[2] or
                boxA[3] < boxB[1] or
                boxA[1] > boxB[3]
        )

        return iou > iou_threshold and overlaps

    def extract_form_features(self, file_path):
        """
        Extracts extended numeric and semantic features from LayoutParser blocks
        for subsequent classification of form templates.
        Returns a dictionary of numerical features and layout statistics.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        num_tables = 0
        num_text_blocks = 0
        num_title_blocks = 0
        cell_counts = []
        estimated_columns = []
        estimated_rows = []
        avg_cell_widths = []
        avg_cell_heights = []
        x_spans = []
        y_spans = []
        type_distribution = {}

        for block in blocks:
            block_type = block["type"]
            type_distribution[block_type] = type_distribution.get(block_type, 0) + 1

            if block_type == "Table":
                num_tables += 1
                cells = block.get("cells", [])
                cell_counts.append(len(cells))

                widths = [cell[2] - cell[0] for cell in cells]
                heights = [cell[3] - cell[1] for cell in cells]

                avg_cell_widths.extend(widths)
                avg_cell_heights.extend(heights)

                x_coords = [x for cell in cells for x in [cell[0], cell[2]]]
                y_coords = [y for cell in cells for y in [cell[1], cell[3]]]

                x_span = max(x_coords) - min(x_coords) if x_coords else 0
                y_span = max(y_coords) - min(y_coords) if y_coords else 0

                x_spans.append(x_span)
                y_spans.append(y_span)

                x_range_pairs = [tuple(cell[0:3:2]) for cell in cells]
                y_range_pairs = [tuple(cell[1:4:2]) for cell in cells]

                unique_x = sorted(set(x for x1, x2 in x_range_pairs for x in [x1, x2]))
                unique_y = sorted(set(y for y1, y2 in y_range_pairs for y in [y1, y2]))

                est_cols = max(1, len(unique_x) // 2)
                est_rows = max(1, len(unique_y) // 2)

                estimated_columns.append(est_cols)
                estimated_rows.append(est_rows)

            elif block_type == "Text":
                num_text_blocks += 1
            elif block_type == "Title":
                num_title_blocks += 1

        max_cells = max(cell_counts) if cell_counts else 0
        avg_cells = np.mean(cell_counts) if cell_counts else 0
        num_large_tables = sum(1 for count in cell_counts if count > 50)
        num_small_tables = sum(1 for count in cell_counts if count <= 10)

        form_features = {
            "num_tables": num_tables,
            "num_cells_total": int(np.sum(cell_counts)),
            "avg_cells_per_table": round(avg_cells, 2),
            "num_text_blocks": num_text_blocks,
            "num_title_blocks": num_title_blocks,
            "num_large_tables": num_large_tables,
            "num_small_tables": num_small_tables,
            "max_cells_table": max_cells,
            "avg_columns_est": int(np.mean(estimated_columns)) if estimated_columns else 0,
            "avg_rows_est": int(np.mean(estimated_rows)) if estimated_rows else 0,
            "avg_cell_width": round(np.mean(avg_cell_widths), 1) if avg_cell_widths else 0,
            "avg_cell_height": round(np.mean(avg_cell_heights), 1) if avg_cell_heights else 0,
            "x_range_cells": int(np.mean(x_spans)) if x_spans else 0,
            "y_range_cells": int(np.mean(y_spans)) if y_spans else 0,
            "type_distribution": type_distribution
        }

        form_features["form_score"] = self.compute_form_score(form_features)
        return form_features

    @staticmethod
    def compute_form_score(features):
        """
        Computes a heuristic score for form complexity.
        """
        T = features.get("num_tables", 0)
        C_avg = features.get("avg_cells_per_table", 0)
        B = features.get("num_text_blocks", 0)
        H = features.get("num_title_blocks", 0)
        S = features.get("num_small_tables", 0)
        R = features.get("avg_rows_est", 0)
        K = features.get("avg_columns_est", 0)

        small_table_ratio = S / T if T > 0 else 0
        text_per_table = B / T if T > 0 else 0

        score = (
            0.25 * T +
            0.15 * C_avg +
            0.15 * H +
            0.10 * B +
            0.10 * R +
            0.10 * K -
            0.20 * small_table_ratio -
            0.05 * text_per_table
        )

        return round(score, 2)


# analyzer = ImageLayoutAnalyzer(score_thresh=0.3,
#                                image_name="form_1_2(11)",
#                                folder_path="../../eval_data/base_data/",
#                                preprocessing_image=True,
#                                run_dla=True,
#                                to_show_results=True)
# analyzer.analyze()

