from pdf2image import convert_from_path
import os


def pdf_to_images(pdf_path, output_folder="../../output/", poppler_path="../../poppler/Library/bin"):
    """
    Wandelt ein PDF in einzelne Seitenbilder um.

    :param pdf_path: Pfad zur PDF-Datei
    :param output_folder: Zielordner für die Bilder
    :param dpi: Auflösung in DPI (empfohlen: 300)
    :return: Liste der erzeugten Bildpfade
    """
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    image_paths = []

    for i, page in enumerate(images):
        filename = os.path.join(output_folder, f"page_{i + 1:02d}.png")
        page.save(filename, "PNG")
        image_paths.append(filename)

    print(f"✅ {len(image_paths)} Seiten extrahiert → gespeichert in: {output_folder}")
    return image_paths


def get_image_path():
    return get_processed_image_path()

def get_image_name():
    return "page_04A.png"

def get_input_image_path():
    return "../../input/page_04A.png"

def get_processed_image_path():
    return "../../input/preprocessed/page_04A.png"


def get_openCV_image_path():
    return "../../output/layout_opencv_output.png"

def get_masked_image_path():
    return "../../input/masked/masked_image.png"