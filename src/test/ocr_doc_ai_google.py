from google.cloud import documentai
from google.api_core.client_options import ClientOptions
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tools import *
import os


# log in data
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../keys/ocr-455618-a90decb41960.json"


def process_document_init(type="ocr", file_path="../../input/T1.png"):
    processor_id = None
    if type == "ocr":
        processor_id = "cb25a78fc4d590b9"
    elif type == "form_parser":
        processor_id = "577c715c7d41875c"

    if not processor_id is None:
        doc = process_document(
                    project_id  = "474916283419",
                    location = "eu" ,
                    processor_id= processor_id,
                    file_path = file_path,
                    mime_type = "image/png")


        if type == "form_parser":
            extract_key_value_pairs(doc)
            extract_tables(file_path, doc)
        else:
            save_visualize_results(file_path, doc, output_path=f"../../output/test/documentai_output.txt")


    else:
        print("No processor id found")

def process_document(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str = "application/pdf"  # oder "image/png", "image/jpeg"
):
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )

    processor_name = client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as f:
        document_content = f.read()

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=document_content,
            mime_type=mime_type
        )
    )

    result = client.process_document(request=request)
    doc = result.document

    return doc

def save_visualize_results(image_path, ocr_doc, output_path="../../output/test/documentai_output.txt"):


    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    lines = []


    # Über alle Seiten und Tokens iterieren
    for page_idx, page in enumerate(ocr_doc.pages):
        for token in page.tokens:
            word = ocr_doc.text[
                   token.layout.text_anchor.text_segments[0].start_index: token.layout.text_anchor.text_segments[
                       0].end_index]

            # Bounding Box
            bounding_poly = token.layout.bounding_poly

            # Punkte der Box
            points = [(vertex.x, vertex.y) for vertex in bounding_poly.vertices]

            draw.line(points + [points[0]], width=2, fill="red")  # Box schließen
            # draw.text(points[0], word, fill="blue")

            line_text = f"- {word}"
            lines.append(line_text)

            print(f"{word} : {points}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Document AI OCR Ergebnisse")
    plt.savefig(output_path.replace(".txt", ".png"))
    plt.show()


def extract_key_value_pairs(document):
    print("\n Erkannte Key-Value Paare:")
    keys_values = []

    for page in document.pages:
        for field in page.form_fields:
            key_text = "_"
            value_text = "_"

            if field.field_name and field.field_name.text_anchor.text_segments:
                start_idx = field.field_name.text_anchor.text_segments[0].start_index
                end_idx = field.field_name.text_anchor.text_segments[0].end_index
                key_text = document.text[start_idx:end_idx]

            if field.field_value and field.field_value.text_anchor.text_segments:
                start_idx = field.field_value.text_anchor.text_segments[0].start_index
                end_idx = field.field_value.text_anchor.text_segments[0].end_index
                value_text = document.text[start_idx:end_idx]


            entry = [key_text.strip(), value_text.strip()]

            print(key_text.strip() + ": " + value_text.strip())

            keys_values.append(entry)


def extract_tables(image_path, doc):

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)


    for page_idx, page in enumerate(doc.pages):
        for table_idx, table in enumerate(page.tables):
            bounding_poly = table.layout.bounding_poly
            if bounding_poly.vertices:
                points = [(vertex.x, vertex.y) for vertex in bounding_poly.vertices]

                if len(points) == 4:
                    draw.line(points + [points[0]], width=3, fill="red")
                    draw.text(points[0], f"Tabelle {table_idx + 1}", fill="black")

            for row in table.body_rows:
                for cell in row.cells:
                    cell_poly = cell.layout.bounding_poly
                    if cell_poly.vertices:
                        cell_points = [(int(vertex.x), int(vertex.y)) for vertex in cell_poly.vertices]
                        if len(cell_points) == 4:
                            draw.line(cell_points + [cell_points[0]], width=2, fill="green")

    # Bild anzeigen
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Tabellen Bounding Boxes")
    plt.savefig("../../output/test/documentai_tables_output.png")
    plt.show()


# process_document_init()
process_document_init(type="ocr", file_path=get_image_path())

process_document_init(type="form_parser", file_path=get_processed_image_path())

# `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
# https://eu-documentai.googleapis.com/v1/projects/474916283419/locations/eu/processors/43c2251eb38bbf59/processorVersions/pretrained-layout-parser-v1.0-2024-06-03:process