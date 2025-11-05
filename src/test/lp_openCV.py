import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tools import *

def image_black_white(image_path):
    """
    Wandelt ein Bild in Schwarz-Weiß um und gibt das binäre Bild zurück.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

def detect_layout_elements(image_path):
    """
    Erkennt alle rechteckigen Layout-Elemente: Tabellen, Felder, Kästchen, etc.
    """
    binary = image_black_white(image_path)
    binary_inv = cv2.bitwise_not(binary)  # für Linien besser geeignet

    # Kombination aus horizontalen und vertikalen Linien
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_v)

    table_like = cv2.add(horizontal, vertical)

    # Kombiniere Linienmaske mit dem Original für allgemeine Block-Erkennung
    combined = cv2.bitwise_or(binary_inv, table_like)

    # Optional: leichte Glättung (Konturen zusammenführen)
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.dilate(combined, kernel_merge, iterations=1)

    # Konturen finden
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ursprungsbild zum Anzeigen laden
    original = cv2.imread(image_path)
    result = original.copy()

    all_table_data = []
    # Rechtecke für alle Layout-Elemente zeichnen
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 150:
            table_coords = (x, y, x + w, y + h)
            table_region = original[y:y + h, x:x + w]
            edges = cv2.Canny(table_region, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=5)

            if lines is not None and len(lines) >= 20:
                result, cell_boxes = extract_table_cells(result, table_coords)

                all_table_data.append({
                    "table_id": idx + 1,
                    "coordinates": [int(x), int(y), int(x + w), int(y + h)],
                    "cells": cell_boxes
                })

    # save results
    save_table_structure(all_table_data)

    # display tables
    show_table_structure(result)


def show_table_structure(result, output_path="../../output/test/layout_opencv_output.png"):
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(output_path)
    plt.title("OpenCV Layoutanalyse: alle visuell erkennbaren Elemente")
    plt.show()


def save_table_structure(table_blocks, output_path="../../output/test/layout_opencv_output.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(table_blocks, f, indent=2)
    print(f"✅ Struktur gespeichert unter: {output_path}")


def extract_table_cells(image, table_coords):
    x1, y1, x2, y2 = map(int, table_coords)
    cropped = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    horizontal = binary.copy()
    hor_kernel_len = horizontal.shape[1] // 50
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_kernel_len, 1))
    horizontal = cv2.erode(horizontal, hor_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, hor_kernel, iterations=1)

    vertical = binary.copy()
    ver_kernel_len = vertical.shape[0] // 10
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_kernel_len))
    vertical = cv2.erode(vertical, ver_kernel, iterations=1)
    vertical = cv2.dilate(vertical, ver_kernel, iterations=1)

    mask = cv2.add(horizontal, vertical)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cell_boxes = []
    overlaid = cropped.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(overlaid, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cell_boxes.append([int(x1 + x), int(y1 + y), int(x1 + x + w), int(y1 + y + h)])

    image[y1:y2, x1:x2] = overlaid
    return image, cell_boxes

detect_layout_elements(get_processed_image_path())
