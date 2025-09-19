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
    def __init__(self, rgb_dir, dsm_dir, transform=None, normalize_dsm=True, batch_size_for_stats=100):
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
        
        if normalize_dsm:
            self.dsm_mean, self.dsm_std = self._calculate_dsm_stats_efficient(batch_size_for_stats)
        else:
            self.dsm_mean, self.dsm_std = 0, 1
        
    def _calculate_dsm_stats_efficient(self, batch_size=100):
        """
        Calculate DSM statistics using incremental computation to avoid memory issues
        """
        from tqdm import tqdm
        
        print("Calculating DSM statistics efficiently (memory-safe)...")
        
        # Initialize variables for incremental mean/std calculation
        total_sum = 0.0
        total_sum_squared = 0.0
        total_count = 0
        
        # Process files in batches
        for i in tqdm(range(0, len(self.matching_numbers), batch_size), desc="Processing DSM batches"):
            batch_numbers = self.matching_numbers[i:i + batch_size]
            batch_values = []
            
            # Load a batch of files
            for num in batch_numbers:
                try:
                    dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
                    dsm_image = Image.open(dsm_path)
                    dsm_data = np.array(dsm_image, dtype=np.float32)
                    valid_data = dsm_data[dsm_data > 0]  # Filter invalid values
                    
                    if len(valid_data) > 0:
                        batch_values.extend(valid_data.flatten())
                    
                    # Close image to free memory
                    dsm_image.close()
                    del dsm_data, valid_data
                    
                except Exception as e:
                    print(f"Error processing file {dsm_path}: {e}")
                    continue
            
            # Update running statistics with this batch
            if batch_values:
                batch_array = np.array(batch_values, dtype=np.float64)
                batch_sum = np.sum(batch_array)
                batch_sum_squared = np.sum(batch_array ** 2)
                batch_count = len(batch_array)
                
                total_sum += batch_sum
                total_sum_squared += batch_sum_squared
                total_count += batch_count
                
                # Clear batch data from memory
                del batch_values, batch_array
                gc.collect()  # Force garbage collection
        
        if total_count == 0:
            raise ValueError("No valid DSM data found to calculate statistics.")
        
        # Calculate final mean and std
        mean = total_sum / total_count
        variance = (total_sum_squared / total_count) - (mean ** 2)
        std = np.sqrt(max(variance, 0))  # Ensure non-negative variance
        
        print(f"DSM Statistics - Mean: {mean:.4f}, Std: {std:.4f}, Total pixels: {total_count}")
        return float(mean), float(std)
    
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
        
        # Clean up
        dsm_image.close()
        
        return rgb_image, dsm_tensor