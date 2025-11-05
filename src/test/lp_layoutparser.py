import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from tools import get_image_path


def run_layoutparser(
    image_path,
    model_type="mask",       # "mask" (PubLayNet) oder "fast" (TableBank)
    score_thresh=0.0,
):
    """
    Führt Layoutanalyse mit lokalem Detectron2-Modell durch.
    Modellwahl über 'model_type': "mask" oder "fast".
    Optional: OCR mit Tesseract.
    """
    label_map = {0: "Text", 1: "Title", 2: "List", 3: "Figure", 4: "Table", 5: "Misc"}

    # Modellpfade + LabelMaps je nach Typ
    if model_type == "mask":
        config_path = "../../models/layoutparser/mask_rcnn_X_101_32x8d_FPN_3x/config.yaml"
        model_path = "../../models/layoutparser/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth"


    elif model_type == "fast":
        config_path = "../../models/layoutparser/faster_rcnn_R_50_FPN_3x/config.yml"
        model_path = "../../models/layoutparser/faster_rcnn_R_50_FPN_3x/model_final.pth"

    else:
        raise ValueError("Ungültiger 'model_type'. Erlaubt: 'mask' oder 'fast'.")

    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Modell laden
    model = lp.Detectron2LayoutModel(
        config_path=config_path,
        model_path=model_path,
        label_map=label_map,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
    )

    # Layoutanalyse
    layout = model.detect(image_rgb)

    # visualisierung
    visualize(image_rgb, layout, model_type)
    visualize_custom_color(image_rgb, layout, model_type)

    # speichern Ergebnisse
    save_results(model_type,layout)

    return layout


def visualize(image_rgb,layout, model_type):

    viz = lp.draw_box(image_rgb, layout, box_width=2)
    plt.figure(figsize=(12, 10))
    plt.imshow(viz)
    plt.axis("off")
    plt.title(f"LayoutParser ({model_type}) Ergebnis")
    plt.show()



def visualize_custom_color(image_rgb, layout, model_type):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image_rgb)
    colors = {"Text": "red", "Title": "green", "List": "orange", "Figure": "darkblue", "Table": "blue"}
    # colors = [
    #     'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
    #     'orange', 'purple', 'lime', 'pink',
    #     'teal', 'brown', 'gold', 'navy', 'olive',
    #     'coral', 'indigo', 'turquoise', 'maroon', 'darkgreen',
    #     'darkblue', 'darkred', 'salmon', 'skyblue', 'khaki'
    # ]

    for i, block in enumerate(layout):
        x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor=colors[block.type],
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1-50, y1 - 5, block.type, color=colors[block.type], fontsize=10, weight="bold")

    ax.axis("off")
    plt.title(f"LayoutParser ({model_type}) – mit benutzerdefinierter Farbe")
    plt.savefig("../../output/test/layout_parser_output.png")
    plt.show()

def save_results(model_type,layout):
    output_lines = ["📄 Layout-Erkannte Blöcke:\n"]
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        out = f"📦 Block {i + 1}: {block.type} [x1={x1}, y1={y1}, x2={x2}, y2={y2}]"
        print(out)
        output_lines.append(out)
    out_path = "../../output/test/layout_parser_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n✅ Ergebnisse für Model {model_type} gespeichert unter: {out_path}")

# Beispiel-Aufruf für originales Bild:
# run_layoutparser(image_path="../../input/T1.png",
#                  model_type="fast",
#                  score_thresh=0.5)

# Beispiel-Aufruf für vorverarbeitetes Bild:
run_layoutparser(image_path=get_image_path(),
                 model_type="fast",
                 score_thresh=0.5)
