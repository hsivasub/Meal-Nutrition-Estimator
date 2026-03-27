import torchvision.transforms as transforms

def get_train_transforms(image_size=224):
    """
    Returns the composition of transforms for training data.
    Includes resizing, data augmentation, and normalization.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    """
    Returns the composition of transforms for validation/testing data.
    Includes resizing, center cropping, and normalization (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_inverse_transforms():
    """
    Returns the composition of inverse transforms to convert normalized 
    tensors back to displayable images. Useful for visualization.
    """
    # Inverse of Standard ImageNet normalization:
    # x_new = (x_old * std) + mean
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
