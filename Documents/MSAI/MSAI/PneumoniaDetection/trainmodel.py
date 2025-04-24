import os
import torch
from model import PneumoniaCNN
from datapreprocessing import prepare_dataloaders
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define paths
DATA_DIR = "/content/MSAI/Documents/MSAI/MSAI/PneumoniaDetection/data/rsna"
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
num_epoch=50
train_losses = []
val_losses = []
if __name__ == '__main__':
# Training loop
    for epoch in range(num_epoch):  # You can increase/decrease this
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
        train_losses.append(avg_loss)
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels, _ in val_loader: 
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device).unsqueeze(1)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epoch}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

    # Plot loss curves
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss over Epochs")
plt.savefig("loss_plot.png")
plt.show()