import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatelliteDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, image_size=512):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_size = image_size
        
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Get corresponding depth map name
        number = image_name.replace('rgb_', '').replace('.png', '')
        depth_name = f'dsm_{number}.png'
        depth_path = os.path.join(self.depth_dir, depth_name)
        
        # Read image (3 channels)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read depth map (4 channels RGBA)
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Extract depth from alpha channel (4th channel)
        if len(depth_rgba.shape) == 3 and depth_rgba.shape[2] == 4:
            depth = depth_rgba[:, :, 3]  # Alpha channel
        else:
            depth = depth_rgba  # Already single channel
        
        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
            depth = cv2.resize(depth, (self.image_size, self.image_size))
        
        # Normalize depth to [0, 1]
        depth = depth.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=depth)
            image = transformed['image']
            depth = transformed['mask']
        else:
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        return image, depth
    
def get_transforms(image_size=384):
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ToTensorV2(),  # Converts to [0, 1] range
    ], additional_targets={'mask': 'image'})
    
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        ToTensorV2(),  # Converts to [0, 1] range
    ], additional_targets={'mask': 'image'})
    
    return train_transform, val_transform





