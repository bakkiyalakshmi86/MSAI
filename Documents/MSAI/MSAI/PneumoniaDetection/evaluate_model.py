import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import PneumoniaCNN
from datapreprocessing import prepare_dataloaders
from camutils import generate_gradcam  # Your Grad-CAM function
import pandas as pd
import os
from torchvision.transforms import ToPILImage

# Paths
DATA_DIR = "/content/MSAI/Documents/MSAI/MSAI/PneumoniaDetection/data/rsna"
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

if __name__ == '__main__':
    model.eval()
    correct = 0
    total = 0

    # Evaluate and generate Grad-CAM
    for idx, (img, label, original_img) in enumerate(val_loader):
        img, label = img.to(device), label.to(device)  # Move images and labels to the correct device
        output = model(img)
        probs = torch.sigmoid(output).squeeze()  # Get probabilities
        probs_list = probs.tolist()
        pred_labels = (probs >= 0.5).int()
        true_labels = label.int()

        correct += (pred_labels == true_labels).sum().item()
        total += label.size(0)
        

        for i in range(img.size(0)):  # Loop through each image in the batch
            imag = img[i].unsqueeze(0)  # Select the i-th image and add batch dimension
            original_image = original_img[i]  # Get corresponding original image
            to_pil = ToPILImage()
            original_image = to_pil(original_image)
            # Generate Grad-CAM heatmap for the i-th image
            heatmap = generate_gradcam(model, imag, original_image)
            
            # Save the generated heatmap with prediction probability and true label
            heatmap.save(os.path.join(
                OUTPUT_DIR,
                f"cam_{idx}_{i}_pred{pred_labels[i].item()}_prob{probs[i]:.4f}_true{true_labels[i].item()}.png"
            ))
    accuracy = correct / total
    print(f"✅ Grad-CAM visualizations saved to gradcam_outputs/")
    print(f"📊 Accuracy: {accuracy:.4f} ({correct}/{total})")