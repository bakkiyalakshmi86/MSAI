import os
import torch
from model import PneumoniaCNN
from datapreprocessing import prepare_dataloaders
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define paths
DATA_DIR = "data/rsna-pneumonia-detection-challenge/"
IMG_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
CSV_PATH = os.path.join(DATA_DIR, "stage_2_train_labels.csv")

# Prepare dataloaders
train_loader, val_loader = prepare_dataloaders(
    csv_path=CSV_PATH,
    image_dir=IMG_DIR
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, optimizer
model = PneumoniaCNN().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
# Training loop
    for epoch in range(10):  # You can increase/decrease this
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")