import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: grayscale image (1 channel)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample by 2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
        )

        self.conv_block5 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
        )

        with torch.no_grad():
          dummy_input = torch.zeros(1, 1, 224, 224)
          x = self.conv_block1(dummy_input)
          x = self.conv_block2(x)
          x = self.conv_block3(x)
          x = self.conv_block4(x)
          x = self.conv_block5(x)
          flatten_dim = x.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # Binary classification
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        #return torch.sigmoid(x)  # Output between 0 and 1
        return x