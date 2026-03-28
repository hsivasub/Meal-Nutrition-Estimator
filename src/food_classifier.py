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
        return x

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FoodEfficientNet(nn.Module):
    """
    Transfer Learning Model utilizing EfficientNet-B0 backbone.
    Allows dynamic freezing/unfreezing of the network for fine-tuning.
    """
    def __init__(self, num_classes=20, freeze_backbone=True):
        super(FoodEfficientNet, self).__init__()
        
        # Load pretrained model
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        
        # Freeze the backbone weights if required
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Replace the final classification head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def unfreeze_top_layers(self, blocks_to_unfreeze=3):
        """
        Unfreezes the top `blocks_to_unfreeze` layers in the features backbone
        for fine-tuning the pretrained model closer to the top layers.
        """
        for param in self.model.features[-blocks_to_unfreeze:].parameters():
            param.requires_grad = True
