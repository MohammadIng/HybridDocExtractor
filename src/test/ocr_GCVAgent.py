import layoutparser as lp
import matplotlib.pyplot as plt
import cv2
from tools import get_image_path

# GCVAgent initialisieren
ocr_agent = lp.GCVAgent.with_credential(
    "../../keys/ocr-455618-a90decb41960.json",
    languages=['deu']
)


def run_ocr(image_path = "../../input/page_00.png"):
    # Bildpfade
    out_path = "../../output/test/gcv_google_output.png"

    # Bild laden
    image = cv2.imread(image_path)
    plt.imshow(image)

    # OCR ausführen
    res = ocr_agent.detect(image, return_response=True)

    # Text auf Wortebene extrahieren
    layout = ocr_agent.gather_full_text_annotation(
        res, agg_level=lp.GCVFeatureType.WORD)

    # Optional: Einzelne Wörter + Positionen anzeigen
    print("\n📄 Erkannte Texte:")
    for i, block in enumerate(layout):
        print(f"- {block.text}")
        # print(f"{i+1:03d}: '{block.text}' at {block.block}")

    draw_Layouts(image, layout, out_path)

    save_results(layout)

def draw_Layouts(image, layout, out_path):
    # Visualisierung mit Text und Boxen
    viz = lp.draw_text(image, layout, font_size=12,
                       with_box_on_text=True, text_box_width=1)

    # Bild anzeigen
    plt.figure(figsize=(12, 10))
    plt.imshow(viz)
    plt.axis("off")
    plt.title("GCVAgent")
    plt.savefig(out_path)
    plt.show()


def save_results(layout, output_path="../../output/test/gcv_google_output.txt"):
    lines = []
    for i, block in enumerate(layout):
        # Koordinaten (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, block.coordinates)
        result = f"{i+1:03d}: {block.text} [x1={x1}, y1={y1}, x2={x2}, y2={y2}]"
        lines.append(result)
        print(result)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Ergebnisse gespeichert unter: {output_path}")

run_ocr(get_image_path())