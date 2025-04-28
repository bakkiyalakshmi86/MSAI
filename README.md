**Project Overview**
This project builds an AI model to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN).
In addition, it integrates explainable AI (XAI) techniques like Grad-CAM, SHAP, and LIME to visualize model decision-making, making the predictions more transparent for healthcare professionals.

**Features**
Pneumonia classification (Pneumonia vs. Normal) from X-rays.

Visualization of important regions using:

Grad-CAM (Gradient-weighted Class Activation Mapping)

SHAP (SHapley Additive exPlanations)

LIME (Local Interpretable Model-agnostic Explanations)

Streamlit web app for easy interaction (optional).

High-quality model evaluation (Accuracy, Precision, Recall, F1-score).

**Clone the repository:**

git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

**Install dependencies:**
pip install -r requirements.txt

**Launch the Streamlit app:**
streamlit run app/app.py

