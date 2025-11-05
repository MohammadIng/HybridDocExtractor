import layoutparser as lp
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizerFast, ViTImageProcessor
import matplotlib.pyplot as plt

from src.layout_analysis.image_layout_analysis import ImageLayoutAnalyzer



# 🔹 TrOCR für OCR auf erkannte Layout-Blöcke anwenden
def run_tr_ocr(
    model_path="../../models/trocr-large-handwritten",
    image_name="page_04A",
):
    # TrOCR vorbereiten
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # Layout erkennen + Bild laden
    analyzer = ImageLayoutAnalyzer(image_name=image_name, to_show_results=True)
    analyzer.analyze()

    # save results
    save_results(analyzer.layout, analyzer.image_pp, processor, model)

    # visualize results
    visualize_results(analyzer.image_pp, analyzer.layout)


def visualize_results(image_data, layout):
    # Visualisierung
    viz = lp.draw_box(image_data, layout, box_width=2)
    plt.figure(figsize=(12, 16))
    plt.imshow(viz)
    plt.axis("off")
    plt.title("LayoutParser + TrOCR-Erkennung")
    plt.savefig("../../output/test/trocr_microsoft_output.png")
    plt.show()


def save_results(layout, image_data, processor, model):

    results = []

    for i, block in enumerate(layout):
        if block.type in ["Text", "Title", "Cell"]:
            segment = block.crop_image(image_data)
            segment_pil = Image.fromarray(segment).convert("RGB")

            pixel_values = processor(images=segment_pil, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"Block {i + 1} ({block.type}):\n{text}\n{'-' * 60}")

            results.append(f"({block.type}):{text}")

    with open("../../output/test/microsoft_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"✅ TrOCR-Ergebnisse gespeichert")

run_tr_ocr()
