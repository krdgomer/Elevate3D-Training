# dataset.py

import torch
from torch.utils.data import Dataset
from transformers import DPTImageProcessor
from PIL import Image
import numpy as np
import os
from src.depth_estimation.config import cfg
from torchvision import transforms

processor = DPTImageProcessor.from_pretrained(cfg.PRETRAINED_MODEL_NAME)

class SatelliteDepthDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, transform=None, normalize_dsm=True):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.transform = transform
        
        # Get matching filenames based on numbers
        rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")]
        dsm_files = [f for f in os.listdir(dsm_dir) if f.startswith("dsm_") and f.endswith(".png")]
        
        rgb_numbers = set(f.split("_")[1].split(".")[0] for f in rgb_files)
        dsm_numbers = set(f.split("_")[1].split(".")[0] for f in dsm_files)
        
        # Find matching numbers
        self.matching_numbers = sorted(rgb_numbers.intersection(dsm_numbers))
        
        self.dsm_mean, self.dsm_std = self._calculate_dsm_stats() if normalize_dsm else (0, 1)
        
    def _calculate_dsm_stats(self):
        from tqdm import tqdm  # Import inside the function if needed
        all_dsm_values = []
        
        # Use tqdm to show progress and prevent timeout
        print("Calculating DSM statistics (this prevents Colab timeout)...")
        for num in tqdm(self.matching_numbers, desc="Processing DSM files"):
            try:
                dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
                dsm_image = Image.open(dsm_path)
                dsm_data = np.array(dsm_image, dtype=np.float32)
                valid_data = dsm_data[dsm_data > 0]  # Filter invalid values
                all_dsm_values.extend(valid_data.flatten())
            except Exception as e:
                print(f"Error processing file {dsm_path}: {e}")
                continue
        
        if not all_dsm_values:
            raise ValueError("No valid DSM data found to calculate statistics.")
            
        return np.mean(all_dsm_values), np.std(all_dsm_values)
    
    def __len__(self):
        return len(self.matching_numbers)
    
    def __getitem__(self, idx):
        num = self.matching_numbers[idx]
        rgb_path = os.path.join(self.rgb_dir, f"rgb_{num}.png")
        dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
        
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