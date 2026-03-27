import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    A simple baseline Custom CNN model for Food Classification.
    Consists of 3 Convolutional Blocks followed by Fully Connected layers.
    Defaults to 20 classes representing our top-20 foods MVP scope.
    """
    def __init__(self, num_classes=20):
        super(BaselineCNN, self).__init__()
        
        # Block 1: Input Shape (3, 224, 224) -> Output (32, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Input Shape (32, 112, 112) -> Output (64, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Block 3: Input Shape (64, 56, 56) -> Output (128, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Fully Connected Classifier
        # Input features: 128 channels * 28 width * 28 height
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for Dense layers
        x = torch.flatten(x, 1) # Flattens all dimensions starting from dimension 1
        
        # Apply FC -> relu -> dropout -> FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
