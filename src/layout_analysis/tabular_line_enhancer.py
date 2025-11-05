import cv2
import numpy as np
import json

class TabularLineEnhancer:
    def __init__(self, overlay_color=(255, 0, 0), alpha=0.6):
        self.overlay_color = np.array(overlay_color)
        self.alpha = alpha
        self.blended = None
        self.image = None
        self.all_table_data = []

    @staticmethod
    def get_binary(gray):
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, -2)
        return binary


    def enhance_lines(self, image):
        """
            Enhances tabular lines in the given image.
            :param image: BGR or grayscale image (numpy.ndarray)
            :return: Image with enhanced lines (BGR)
        """
        if len(image.shape) == 2:
            gray = image.copy()
            self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            self.image = image.copy()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        binary = self.get_binary(gray)

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 50, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 10))

        horizontal = cv2.dilate(cv2.erode(binary.copy(), hor_kernel), hor_kernel)
        vertical = cv2.dilate(cv2.erode(binary.copy(), ver_kernel), ver_kernel)
        mask = cv2.add(horizontal, vertical)

        mask_norm = (mask > 0).astype(np.uint8)[:, :, None]
        blended = np.where(
            mask_norm,
            (self.alpha * self.overlay_color + (1 - self.alpha) * self.image).astype(np.uint8),
            self.image
        )
        self.blended = blended
        return self.blended

    def detect_tables_coordinates(self, min_table_lines=5):
        """
            Detects table ranges based on line structures using the preloaded image.
            The final bounding box of each table is computed from the cells.
            :param min_table_lines: Minimum number of detected lines to validate a table
        """
        def bounding_box_from_cells(cells):
            if not cells:
                return None
            x1 = min(box[0] for box in cells)
            y1 = min(box[1] for box in cells)
            x2 = max(box[2] for box in cells)
            y2 = max(box[3] for box in cells)
            return [x1, y1, x2, y2]

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = self.get_binary(gray)

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 80, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 40))

        horizontal = cv2.dilate(cv2.erode(binary.copy(), hor_kernel), hor_kernel)
        vertical = cv2.dilate(cv2.erode(binary.copy(), ver_kernel), ver_kernel)
        table_like = cv2.add(horizontal, vertical)

        merged = cv2.dilate(cv2.bitwise_or(cv2.bitwise_not(gray), table_like),
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.all_table_data.clear()
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            table_coords = (x, y, x + w, y + h)
            table_region = gray[y:y + h, x:x + w]
            edges = cv2.Canny(table_region, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=5)

            if lines is not None and len(lines) >= min_table_lines:
                cells = self.extract_table_cells(table_coords)
                corrected_coords = bounding_box_from_cells(cells)
                if corrected_coords:
                    self.all_table_data.append({
                        "table_id": idx + 1,
                        "coordinates": corrected_coords,
                        "cells": cells
                    })

        return self.all_table_data


    def extract_table_cells(self, table_coords):
        """
            Extracts the inner cells of a table.
            :param table_coords: Tuple with (x1, y1, x2, y2)
            :return: List of cell coordinates
        """
        x1, y1, x2, y2 = map(int, table_coords)
        cropped = self.image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        binary = self.get_binary(gray)

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 80, 1))
        try:
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 40))
        except cv2.error:
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 10))

        horizontal = cv2.dilate(cv2.erode(binary.copy(), hor_kernel), hor_kernel)
        vertical = cv2.dilate(cv2.erode(binary.copy(), ver_kernel), ver_kernel)

        mask = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cell_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 15:
                cell_boxes.append([int(x1 + x), int(y1 + y), int(x1 + x + w), int(y1 + y + h)])

        return cell_boxes


    def save_table_structure(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.all_table_data, f, indent=2)
        print(f"✅ Structure saved under: {output_path}")
