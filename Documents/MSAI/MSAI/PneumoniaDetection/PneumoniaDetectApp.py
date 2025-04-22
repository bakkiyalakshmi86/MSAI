import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import PneumoniaCNN
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm
from torchvision.transforms import ToPILImage
from camutils import generate_gradcam 


# Title
st.title("Pneumonia Detection with Explainable AI")

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG or JPG):", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load model
    model = PneumoniaCNN()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    # Display prediction
    diagnosis = "Pneumonia Detected" if prob > 0.5 else "Normal"
    st.markdown(f"### Prediction Score: {prob:.4f}")
    st.markdown(f"### Diagnosis: **{diagnosis}**")

    # Grad-CAM heatmap
    st.markdown("### Grad-CAM Heatmap")
    heatmap = generate_gradcam(model, input_tensor, original_image=image)
    st.image(heatmap, caption="Explainability via Grad-CAM", use_column_width=True)
