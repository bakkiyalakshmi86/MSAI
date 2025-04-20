import os
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
#import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd

class RSNADataset(Dataset):
    def __init__(self, df, transform=None, return_original=False):
        self.return_original = return_original
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dicom_path = row['path']
        dicom = pydicom.dcmread(dicom_path)
        img = Image.fromarray(dicom.pixel_array).convert("L")
        original_img = img.convert("RGB")
        transformed_img = self.transform(img)
        original_img_tensor = self.original_transform(original_img)
        label = torch.tensor(row['Target'], dtype=torch.float32)
        if self.return_original:
            return transformed_img, label, original_img_tensor
        else:
            return transformed_img, label
        

    """
    def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_source).convert("RGB")
    return transform(img) 
"""

def prepare_dataloaders(csv_path, image_dir, batch_size=32):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="patientId")  # one label per image
    df["path"] = df["patientId"].apply(lambda x: os.path.join(image_dir, f"{x}.dcm"))

    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['Target'], random_state=42)

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    train_dataset = RSNADataset(df_train, transform=transform, return_original=False)
    val_dataset = RSNADataset(df_val, transform=transform, return_original=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader