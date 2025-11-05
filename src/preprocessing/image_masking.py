import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from matplotlib import pyplot as plt, patches

from src.test.tools import *


def mask_image(image_path, output_path="../../input/masked/masked_image.png", shrink=4):
    #  Modell laden
    model = ocr_predictor(pretrained=True)
    #  PNG oder PDF laden
    doc = DocumentFile.from_images(image_path)
    # Vorhersage starten (OCR + Layoutanalyse)
    result = model(doc)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                x1 = int(x_min * w) + 2
                y1 = int(y_min * h) + shrink
                x2 = int(x_max * w) - shrink
                y2 = int(y_max * h) - shrink
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)
    # Bild speichern
    # Visualisierung mit matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Maskiertes Bild (DocTR OCR)")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    print(f"✅ Maskiertes Bild gespeichert unter: {output_path}")

mask_image(get_processed_image_path())