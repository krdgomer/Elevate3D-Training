# dataset.py

import torch
from torch.utils.data import Dataset
from transformers import DPTImageProcessor
from PIL import Image
import numpy as np
import os
from config import cfg
from torchvision import transforms

processor = DPTImageProcessor.from_pretrained(cfg.PRETRAINED_MODEL_NAME)

class SatelliteDepthDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, transform=None, normalize_dsm=True):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.transform = transform
        
        self.filenames = [f for f in os.listdir(rgb_dir) if f.endswith(('.tif', '.png', '.jpg'))]
        
        self.dsm_mean, self.dsm_std = self._calculate_dsm_stats() if normalize_dsm else (0, 1)
        
    def _calculate_dsm_stats(self):
        all_dsm_values = []
        for fname in self.filenames:
            dsm_path = os.path.join(self.dsm_dir, fname)
            dsm_image = Image.open(dsm_path)
            dsm_data = np.array(dsm_image, dtype=np.float32)
            valid_data = dsm_data[dsm_data > 0]  # Filter invalid values (e.g., 0 or negative)
            all_dsm_values.extend(valid_data.flatten())
        return np.mean(all_dsm_values), np.std(all_dsm_values)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        rgb_path = os.path.join(self.rgb_dir, fname)
        dsm_path = os.path.join(self.dsm_dir, fname)
        
        # Load and transform RGB
        rgb_image = Image.open(rgb_path).convert('RGB')
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            # Default transform using HF processor
            rgb_image = processor(rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Load and normalize DSM
        dsm_image = Image.open(dsm_path)
        dsm_data = np.array(dsm_image, dtype=np.float32)
        dsm_data[dsm_data < 0] = 0  # Replace invalid values with 0
        dsm_data = (dsm_data - self.dsm_mean) / self.dsm_std
        
        dsm_tensor = torch.FloatTensor(dsm_data).unsqueeze(0)  # Add channel dim
        
        return rgb_image, dsm_tensor

# Example of a custom transform if needed
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])