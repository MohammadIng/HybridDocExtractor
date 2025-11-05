import cv2
import pytesseract
from matplotlib import pyplot as plt
from src.layout_analysis.image_layout_analysis import  ImageLayoutAnalyzer
import json

def extract_text_from_blocks(image, layout, output_path="../../output/test/pytesseract_output.json"):
    """
    Extrahiert Text aus allen Layout-Blöcken (Text, Titel, Tabelle, Zellen) und speichert das Ergebnis als JSON.
    :param image: BGR- oder RGB-Bild
    :param layout: Liste von TextBlock-Elementen
    :param output_path: Pfad zur Ausgabe-JSON
    """
    results = []
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        roi = image_gray[y1:y2, x1:x2]

        try:
            text = pytesseract.image_to_string(roi, lang="deu")  # ggf. "eng" oder beide
            print(text)
        except Exception as e:
            text = f"[OCR-Fehler: {str(e)}]"
            print(text)
        results.append({
            "block_id": i + 1,
            "type": block.type,
            "coordinates": [x1, y1, x2, y2],
            "text": text.strip()
        })
    print(f"✅ OCR-Ergebnisse gespeichert unter: {output_path}")


def extract_and_display_text_from_blocks(image, layout, output_path="../../output/test/pytesseract_output"):
    """
    Extrahiert Text aus Layout-Blöcken, zeigt ihn im Bild an und speichert das Ergebnis als JSON.
    :param image: BGR- oder RGB-Bild
    :param layout: Liste von TextBlock-Elementen
    :param output_path: Pfad zur Ausgabe-JSON
    """
    results = []
    image_disp = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        roi = image_gray[y1:y2, x1:x2]

        try:
            text = pytesseract.image_to_string(roi, lang="deu")
        except Exception as e:
            text = f"[OCR-Fehler: {str(e)}]"

        results.append({
            "block_id": i + 1,
            "type": block.type,
            "coordinates": [x1, y1, x2, y2],
            "text": text.strip()
        })

        label = text.strip().split("\n")[0][:30] + "..." if len(text.strip()) > 30 else text.strip()
        if label:
            # cv2.rectangle(image_disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image_disp, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Anzeige
    plt.figure(figsize=(14, 12))
    plt.imshow(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Erkannte Layout-Elemente mit OCR-Texten")
    plt.savefig(output_path+".png")
    plt.show()

    with open(output_path+".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ OCR-Ergebnisse + Visualisierung gespeichert unter: {output_path}.json")


    print(f"✅ OCR-Ergebnisse gespeichert unter: {output_path}")


analyzer = ImageLayoutAnalyzer(image_name="page_04a",
                               image_preprocessing=False,
                               to_save_results=False)
analyzer.analyze()
extract_and_display_text_from_blocks(analyzer.image_rgb, analyzer.layout)