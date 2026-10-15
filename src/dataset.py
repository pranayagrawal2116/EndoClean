import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2

# Define target classes for this study
TARGET_CLASSES = ['Normal', 'Angiectasia', 'Ulcer', 'Blood', 'Erosion']

def get_transforms(is_train: bool):
    """
    Returns albumentations augmentation pipeline.
    Identical across all 3 data conditions to ensure fair comparison.
    """
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def create_patient_split(metadata_path: str, val_size=0.15, test_size=0.15, random_state=42):
    """
    Creates a PATIENT-WISE stratified train/val/test split to prevent data leakage.
    Assumes metadata CSV has at least: 'filename', 'video_id', 'class'.
    """
    df = pd.read_csv(metadata_path)
    
    # Filter to only the classes we care about
    df = df[df['class'].isin(TARGET_CLASSES)].copy()
    
    # GroupShuffleSplit ensures all images from a video_id stay in the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=random_state)
    train_idx, temp_idx = next(gss.split(df, groups=df['video_id']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Split the temp_df into val and test
    val_ratio = val_size / (val_size + test_size)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=(1 - val_ratio), random_state=random_state)
    val_idx, test_idx = next(gss_test.split(temp_df, groups=temp_df['video_id']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    return train_df, val_df, test_df

class EndoCleanDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        """
        Args:
            df: DataFrame containing 'filename' and 'class'.
            data_dir: Root directory for the images (e.g. /data/raw or /data/pde_denoised)
            transform: Albumentations compose object.
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        class_name = row['class']
        
        # Expected path: data_dir / class_name / img_name
        img_path = self.data_dir / class_name / img_name
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = self.class_to_idx[class_name]
        
        return image, label

def get_dataloaders(data_dir: str, metadata_path: str, batch_size=32, num_workers=4, seed=42):
    """
    Returns train, val, and test dataloaders for a specific data condition.
    """
    train_df, val_df, test_df = create_patient_split(metadata_path, random_state=seed)
    
    train_dataset = EndoCleanDataset(train_df, data_dir, transform=get_transforms(is_train=True))
    val_dataset = EndoCleanDataset(val_df, data_dir, transform=get_transforms(is_train=False))
    test_dataset = EndoCleanDataset(test_df, data_dir, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Calculate class weights for imbalance handling during training
    class_counts = train_df['class'].value_counts()
    weights = [1.0 / class_counts.get(cls, 1) for cls in TARGET_CLASSES]
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() # Normalize
    
    return train_loader, val_loader, test_loader, weights
