import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import PneumoniaCNN
from datapreprocessing import prepare_dataloaders
from camutils import generate_gradcam, explain_with_lime, explain_with_shap
import pandas as pd
import os
from torchvision.transforms import ToPILImage
from sklearn.metrics import precision_score, recall_score, f1_score 


# Paths
DATA_DIR = "/Users/bakkiya/Documents/MSAI/MSAI/PneumoniaDetection/data/rsna-pneumonia-detection-challenge"
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
CSV_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
OUTPUT_DIR = "gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_loader, val_loader = prepare_dataloaders(
    csv_path=CSV_PATH,
    image_dir=IMG_DIR
)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))

to_pil = ToPILImage()

if __name__ == '__main__':
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Evaluate and generate Grad-CAM
    for idx, (img, label, original_img) in enumerate(val_loader):
        img, label = img.to(device), label.to(device)  # Move images and labels to the correct device
        output = model(img)
        probs = torch.sigmoid(output).squeeze()  # Get probabilities
        probs_list = probs.tolist()
        pred_labels = (probs >= 0.5).int()
        true_labels = label.int()

        correct += (pred_labels == label).sum().item()
        total += label.size(0)

        all_preds.extend(pred_labels.cpu().numpy())    # Collect predictions
        all_labels.extend(true_labels.cpu().numpy())   # Collect ground truths
        

        for i in range(img.size(0)):  # Loop through each image in the batch
            image_tensor = img[i].unsqueeze(0) 
            original_tensor = original_img[i] # Select the i-th image and add batch dimension
            original_image = to_pil(original_tensor) # Get corresponding original image
            # Generate Grad-CAM heatmap for the i-th image
        heatmap = generate_gradcam(model, image_tensor, original_image)
        
        # Generate LIME explanation
        lime_img = explain_with_lime(model, original_image)
        
        # Generate SHAP explanation
        shap_img = explain_with_shap(model, image_tensor, original_image)
        
        # Save the original image
        original_image.save(os.path.join(
            OUTPUT_DIR,
            f"original_{idx}_{i}_pred{pred_labels[i].item()}_prob{probs[i]:.4f}_true{true_labels[i].item()}.png"
        ))

        # Save the generated heatmap with prediction probability and true label
        heatmap.save(os.path.join(
            OUTPUT_DIR,
            f"cam_{idx}_{i}_pred{pred_labels[i].item()}_prob{probs[i]:.4f}_true{true_labels[i].item()}.png"
        ))

        # Optionally save the LIME, SHAP, and Score-CAM images
        lime_img.save(os.path.join(
            OUTPUT_DIR,
            f"lime_{idx}_{i}_pred{pred_labels[i].item()}_prob{probs[i]:.4f}_true{true_labels[i].item()}.png"
        ))
        shap_img.save(os.path.join(
            OUTPUT_DIR,
            f"shap_{idx}_{i}_pred{pred_labels[i].item()}_prob{probs[i]:.4f}_true{true_labels[i].item()}.png"
        ))
        
            
            
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"✅ Grad-CAM visualizations saved to gradcam_outputs/")
    print(f"📊 Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"📈 Precision: {precision:.4f}")
    print(f"📉 Recall:    {recall:.4f}")
    print(f"🎯 F1 Score:  {f1:.4f}")