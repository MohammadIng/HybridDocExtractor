from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from PIL import Image

def train ():
    model_path = "../../models/trocr-large-handwritten"

    # If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # load image from the IAM dataset
    img_path = "../../input/page_00.png"
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    text = "ich bin mit der Leistung zufrieden"

    # training
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    labels = processor.tokenizer(text, return_tensors="pt").input_ids
    outputs = model(pixel_values, labels=labels)
    loss = outputs.loss
    round(loss.item(), 2)

    # inference
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)



def evaluate ():
    # Beispielbild laden
    img_path = "../../input/page_00.png"

    image = Image.open(img_path).convert("RGB")

    #
    model_path = "../../models/trocr-large-handwritten"
    # Modell & Processor laden
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # Bild vorbereiten & Text generieren
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Erkannter Text:", text)

train()

# evaluate()


