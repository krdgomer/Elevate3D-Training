# dataset.py

import torch
from torch.utils.data import Dataset
from transformers import DPTImageProcessor
from PIL import Image
import numpy as np
import os
from src.depth_estimation.config import cfg
from torchvision import transforms
import gc

processor = DPTImageProcessor.from_pretrained(cfg.PRETRAINED_MODEL_NAME)

class SatelliteDepthDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, transform=None, normalize_dsm=True, 
                 batch_size_for_stats=100, resize_to_model_input=True):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.transform = transform
        self.resize_to_model_input = resize_to_model_input
        
        # Get matching filenames
        rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")]
        dsm_files = [f for f in os.listdir(dsm_dir) if f.startswith("dsm_") and f.endswith(".png")]
        
        rgb_numbers = set(f.split("_")[1].split(".")[0] for f in rgb_files)
        dsm_numbers = set(f.split("_")[1].split(".")[0] for f in dsm_files)
        
        self.matching_numbers = sorted(rgb_numbers.intersection(dsm_numbers))
        print(f"Found {len(self.matching_numbers)} matching RGB-DSM pairs")
        
        if normalize_dsm:
            self.dsm_mean, self.dsm_std = self._calculate_dsm_stats_efficient(batch_size_for_stats)
        else:
            self.dsm_mean, self.dsm_std = 0, 1
        
    def _calculate_dsm_stats_efficient(self, batch_size=100):
        """Calculate DSM statistics with proper 8-bit PNG handling"""
        from tqdm import tqdm
        
        print("Calculating DSM statistics for 8-bit PNG files...")
        
        total_sum = 0.0
        total_sum_squared = 0.0
        total_count = 0
        
        # Process files in batches
        for i in tqdm(range(0, len(self.matching_numbers), batch_size), desc="Processing DSM batches"):
            batch_numbers = self.matching_numbers[i:i + batch_size]
            batch_values = []
            
            for num in batch_numbers:
                try:
                    dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
                    dsm_image = Image.open(dsm_path)
                    dsm_data = np.array(dsm_image, dtype=np.float32)
                    
                    # IMPORTANT: 8-bit PNG values are 0-255, scale to meaningful range
                    # Assuming 0-255 represents 0-50 meters (adjust based on your data)
                    MAX_DEPTH = 50.0  # Maximum depth in meters
                    dsm_data = (dsm_data / 255.0) * MAX_DEPTH
                    
                    # Only consider non-zero values
                    valid_mask = dsm_data > 0.1  # Small threshold to avoid noise
                    if np.any(valid_mask):
                        valid_data = dsm_data[valid_mask]
                        batch_values.extend(valid_data.flatten())
                    
                    dsm_image.close()
                    del dsm_data, valid_mask
                    
                except Exception as e:
                    print(f"Error processing {dsm_path}: {e}")
                    continue
            
            if batch_values:
                batch_array = np.array(batch_values, dtype=np.float64)
                total_sum += np.sum(batch_array)
                total_sum_squared += np.sum(batch_array ** 2)
                total_count += len(batch_array)
                
                del batch_values, batch_array
                gc.collect()
        
        if total_count == 0:
            raise ValueError("No valid DSM data found!")
        
        mean = total_sum / total_count
        variance = (total_sum_squared / total_count) - (mean ** 2)
        std = np.sqrt(max(variance, 0))
        
        print(f"📊 DSM Statistics - Mean: {mean:.2f}m, Std: {std:.2f}m, Range: ~{mean-2*std:.1f} to {mean+2*std:.1f}m")
        return float(mean), float(std)
    
    def __getitem__(self, idx):
        num = self.matching_numbers[idx]
        rgb_path = os.path.join(self.rgb_dir, f"rgb_{num}.png")
        dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
        
        try:
            # Load RGB
            rgb_image = Image.open(rgb_path).convert('RGB')
            
            # Load DSM and convert from 8-bit to depth values
            dsm_image = Image.open(dsm_path)
            if dsm_image.mode != 'L':
                dsm_image = dsm_image.convert('L')
            
            dsm_data = np.array(dsm_image, dtype=np.float32)
            
            # Convert 8-bit to depth values (same scaling as in stats calculation)
            MAX_DEPTH = 50.0
            dsm_data = (dsm_data / 255.0) * MAX_DEPTH
            
            # Process RGB with HF processor
            if self.transform:
                if self.resize_to_model_input:
                    rgb_image = rgb_image.resize((384, 384), Image.LANCZOS)
                rgb_tensor = self.transform(rgb_image)
            else:
                processed = processor(rgb_image, return_tensors="pt")
                rgb_tensor = processed["pixel_values"].squeeze(0)
            
            # Process DSM
            if self.resize_to_model_input:
                dsm_pil = Image.fromarray(dsm_data.astype(np.float32))
                dsm_pil = dsm_pil.resize((384, 384), Image.LANCZOS)
                dsm_data = np.array(dsm_pil, dtype=np.float32)
                dsm_pil.close()
            
            # Normalize DSM
            dsm_data = (dsm_data - self.dsm_mean) / self.dsm_std
            
            # Convert to tensor
            dsm_tensor = torch.FloatTensor(dsm_data).unsqueeze(0)
            
            rgb_image.close()
            dsm_image.close()
            
            return rgb_tensor, dsm_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (num={num}): {e}")
            # Return proper shapes
            dummy_rgb = torch.randn(3, 384, 384) * 0.1  # Small random values instead of zeros
            dummy_dsm = torch.randn(1, 384, 384) * 0.1
            return dummy_rgb, dummy_dsm