import json
import cv2
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from tools import get_processed_image_path


def run_ocr(image_path):
    """
    Run OCR using the DocTR predictor and process the result.
    """
    # Load OCR model
    model = ocr_predictor(pretrained=True)

    # Load image or PDF
    doc = DocumentFile.from_images(image_path)

    # Run OCR and layout analysis
    result = model(doc)

    # Visualize and export
    visualize_results(image_path, result)
    save_results_as_json(result)


def visualize_results(image_path, result):
    """
    Display OCR results with bounding boxes and recognized text using OpenCV and Matplotlib.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                x1 = int(x_min * w)
                y1 = int(y_min * h)
                x2 = int(x_max * w)
                y2 = int(y_max * h)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put recognized text
                cv2.putText(image, str(word.value), (x1, y1),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # Show and save visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("DocTR OCR Output")
    plt.axis("off")
    plt.savefig("../../output/test/doctr_output.png", bbox_inches="tight", pad_inches=0)
    plt.show()


def save_results_as_json(result, output_path="../../output/test/doctr_output.json"):
    """
    Save OCR results in structured JSON format.
    """
    output_data = []

    for page_num, page in enumerate(result.pages):
        page_entry = {
            "page": page_num + 1,
            "lines": []
        }

        for block in page.blocks:
            for line in block.lines:
                text_line = " ".join(str(word.value) for word in line.words)
                (x_min, y_min), (x_max, y_max) = line.geometry

                line_entry = {
                    "text": text_line,
                    "box": {
                        "x_min": round(x_min, 4),
                        "y_min": round(y_min, 4),
                        "x_max": round(x_max, 4),
                        "y_max": round(y_max, 4)
                    }
                }

                page_entry["lines"].append(line_entry)

        output_data.append(page_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ OCR results saved to JSON: {output_path}")


def mask_image(image_path, output_path="../../input/masked/masked_image.png", shrink=4):
    """
    Mask the OCR-detected text regions with white boxes for anonymization or preprocessing.
    """
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(image_path)
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

    # Save and show masked image
    plt.figure(figsize=(12, 10))
    plt.imshow(image_rgb)
    plt.title("Masked Image (DocTR OCR)")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"✅ Masked image saved to: {output_path}")


# Entry point
if __name__ == "__main__":
    run_ocr(get_processed_image_path())
