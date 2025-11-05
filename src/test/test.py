from pdf2image import convert_from_path

pages = convert_from_path("../../input/pdf/form.pdf", dpi=300)

for i, page in enumerate(pages):
    page.save(f"../../output/test/page_{i+1}.png", "PNG")
