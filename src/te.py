import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# load BLIP model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_caption(image_path):
    # load image
    raw_image = Image.open(image_path).convert("RGB")

    # preprocess
    inputs = processor(raw_image, return_tensors="pt").to(device)

    # generate caption
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    img_path = "example.webp"   # 👉 replace with your uploaded image path
    caption = generate_caption(img_path)
    print("Generated Caption:", caption)
