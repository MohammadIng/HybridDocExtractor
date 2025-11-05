import cv2
import json
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from src.test.tools import get_image_path


def run_ocr(img_path="../../input/page_00.png"):
    """
    Run OCR using PaddleOCR with German language support.
    """
    # Load model
    ocr = PaddleOCR(use_angle_cls=True, lang='de')  # 'de' for German

    # Run OCR
    results = ocr.ocr(img_path, cls=True)

    # Visualize and save results
    visualize_results(img_path, results)
    save_results_as_json(results)


def visualize_results(img_path, results):
    """
    Display OCR results by drawing bounding boxes and recognized text on the image.
    """
    image = cv2.imread(img_path)

    for box, (text, score) in results[0]:
        # Compute bounding box coordinates
        x_min = int(min(pt[0] for pt in box))
        y_min = int(min(pt[1] for pt in box))
        x_max = int(max(pt[0] for pt in box))
        y_max = int(max(pt[1] for pt in box))

        # Draw bounding box and label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (x_min, y_min - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 255), thickness=1)

    # Show image using matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("PaddleOCR Output")
    plt.axis("off")
    plt.savefig("../../output/test/paddleocr_output.png", bbox_inches="tight", pad_inches=0)
    plt.show()


def save_results_as_json(results, output_path="../../output/test/paddleocr_output.json"):
    """
    Save OCR results in structured JSON format.
    """
    json_data = []

    for i, (box, (text, score)) in enumerate(results[0]):
        coordinates = [{"x": int(x), "y": int(y)} for x, y in box]
        json_data.append({
            "id": i + 1,
            "text": text,
            "score": round(score, 4),
            "box": coordinates
        })

        print(f"{i + 1:03d}: {text} [Score: {score:.2f}]")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"✅ OCR results saved to JSON: {output_path}")


# Run the OCR process
if __name__ == "__main__":
    run_ocr(get_image_path())
