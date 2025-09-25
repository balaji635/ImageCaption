import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model & processor once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Streamlit UI
st.title(" Image Caption Generator")
st.write("Upload an image and get an AI-generated caption using BLIP.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption button
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=20)
            caption = processor.decode(out[0], skip_special_tokens=True)
        st.success("✅ Caption Generated!")
        st.subheader(caption)
