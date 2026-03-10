import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
}


def get_train_val_loaders():
    train_tf = data_transforms['train']
    val_tf = data_transforms['val']

    train_dataset = datasets.ImageFolder(root="data/processed/train", transform=train_tf)
    val_dataset = datasets.ImageFolder(root="data/processed/val", transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Training set sample: {len(train_dataset)}")
    print(f"Validation set sample: {len(val_dataset)}")
    return train_loader, val_loader


def get_test_loader():
    test_tf = data_transforms['val']
    test_dataset = datasets.ImageFolder(root="data/processed/test", transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test set sample: {len(test_dataset)}")
    return test_loader
