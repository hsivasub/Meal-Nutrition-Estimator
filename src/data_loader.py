import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from src.preprocessing import get_train_transforms, get_val_transforms

class FoodImageDataset(Dataset):
    """
    Custom PyTorch Dataset for Food images.
    Can be used if we need custom logic beyond what ImageFolder provides,
    such as parsing a CSV file with paths and labels.
    """
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (pandas.DataFrame): DataFrame with 'filename' and 'label' columns.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # Create a mapping from label names to indices if needed
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[row['label']]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(train_dir, val_dir, batch_size=32, image_size=224, num_workers=4):
    """
    Creates training and validation dataloaders using torchvision's ImageFolder.
    Assumes standard directory structure: dir/class_name/img.jpg
    """
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Use datasets.ImageFolder if data is neatly organized in folders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes
