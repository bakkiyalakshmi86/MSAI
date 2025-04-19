import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import PneumoniaCNN
from datapreprocessing import preprocess_image
from cam_utils import generate_gradcam  # Assume you have a Grad-CAM util script

# Title
st.title("Pneumonia Detection with Explainable AI")

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG or JPG):", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
    input_tensor = preprocess_image(uploaded_file).unsqueeze(0)  # Add batch dimension

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
    heatmap = generate_gradcam(model, input_tensor)
    st.image(heatmap, caption="Explainability via Grad-CAM", use_column_width=True)