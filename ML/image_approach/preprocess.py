import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def preprocess():
    """Preprocess the data create custom dataset and return test, train and validation split

    Returns
    -------
    DataLoader
        three dataloaders for train, test and validations
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load custom dataset
    custom_dataset = ImageFolder(root='../data/images/', transform=transform)

    # Split the dataset
    train_size = int(0.8 * len(custom_dataset))
    val_size = (len(custom_dataset) - train_size)//2
    test_size = (len(custom_dataset) - train_size)//2
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Return the dataloaders
    return train_loader, val_loader, test_loader
