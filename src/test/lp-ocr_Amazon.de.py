import boto3
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tools import get_image_path

def run_ocr(image_path = "../../input/page_00.png"):
    # Textract-Client
    client = boto3.client('textract', region_name='eu-central-1')

    # Bildpfad
    image = Image.open(image_path)
    img_width, img_height = image.size
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Bild als Bytes laden
    with open(image_path, "rb") as file:
        img_bytes = file.read()

    # Textract aufrufen
    response = client.analyze_document(
        Document={'Bytes': img_bytes},
        FeatureTypes=["TABLES", "FORMS"]
    )

    texts = []

    # Textblöcke extrahieren und visualisieren
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            text = block['Text']
            box = block['Geometry']['BoundingBox']


            # Koordinaten berechnen
            x = int(box['Left'] * img_width)
            y = int(box['Top'] * img_height)
            w = int(box['Width'] * img_width)
            h = int(box['Height'] * img_height)
            result = f"{text}:    ({str(x)}, {str(y)})"

            texts.append(result)

            print(result)


            # Rechteck zeichnen
            cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(img_cv2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    visualize_results(img_cv2)

    save_results(texts)

def visualize_results(img):
    # Bild anzeigen
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Amazon Textract")
    plt.savefig("../../output/test/textract_output.png")
    plt.show()


def save_results(texts):
    with open("../../output/test/textract_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"✅ Textract-Ergebnisse gespeichert")

run_ocr(get_image_path())