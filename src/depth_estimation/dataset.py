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
        self.resize_to_model_input = resize_to_model_input  # Resize to match model input (384x384)
        
        # Get matching filenames based on numbers
        rgb_files = [f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(".png")]
        dsm_files = [f for f in os.listdir(dsm_dir) if f.startswith("dsm_") and f.endswith(".png")]
        
        rgb_numbers = set(f.split("_")[1].split(".")[0] for f in rgb_files)
        dsm_numbers = set(f.split("_")[1].split(".")[0] for f in dsm_files)
        
        # Find matching numbers
        self.matching_numbers = sorted(rgb_numbers.intersection(dsm_numbers))
        print(f"Found {len(self.matching_numbers)} matching RGB-DSM pairs")
        
        if normalize_dsm:
            self.dsm_mean, self.dsm_std = self._calculate_dsm_stats_efficient(batch_size_for_stats)
        else:
            self.dsm_mean, self.dsm_std = 0, 1
        
    def _calculate_dsm_stats_efficient(self, batch_size=100):
        """Calculate DSM statistics using incremental computation to avoid memory issues"""
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
                    
                    # Convert 8-bit PNG to float values (0-255 -> proper depth values)
                    dsm_data = np.array(dsm_image, dtype=np.float32)
                    
                    # For 8-bit depth maps, typically we want to scale them
                    # Assuming your 8-bit values represent depth in some meaningful range
                    # You may need to adjust this scaling based on your data
                    # For now, using 0-255 as the range
                    valid_mask = dsm_data > 0  # Assume 0 is invalid
                    if np.any(valid_mask):
                        valid_data = dsm_data[valid_mask]
                        batch_values.extend(valid_data.flatten())
                    
                    # Close image to free memory
                    dsm_image.close()
                    del dsm_data, valid_mask
                    
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
                gc.collect()
        
        if total_count == 0:
            raise ValueError("No valid DSM data found to calculate statistics.")
        
        # Calculate final mean and std
        mean = total_sum / total_count
        variance = (total_sum_squared / total_count) - (mean ** 2)
        std = np.sqrt(max(variance, 0))
        
        print(f"DSM Statistics - Mean: {mean:.4f}, Std: {std:.4f}, Total pixels: {total_count}")
        return float(mean), float(std)
    
    def __len__(self):
        return len(self.matching_numbers)
    
    def __getitem__(self, idx):
        num = self.matching_numbers[idx]
        rgb_path = os.path.join(self.rgb_dir, f"rgb_{num}.png")
        dsm_path = os.path.join(self.dsm_dir, f"dsm_{num}.png")
        
        try:
            # Load RGB image (8-bit, 3-channel, 512x512)
            rgb_image = Image.open(rgb_path).convert('RGB')
            
            # Load DSM image (8-bit, 1-channel, 512x512)
            dsm_image = Image.open(dsm_path)
            if dsm_image.mode != 'L':  # Ensure grayscale
                dsm_image = dsm_image.convert('L')
            
            dsm_data = np.array(dsm_image, dtype=np.float32)
            
            # Handle invalid DSM values (assuming 0 is invalid)
            dsm_data[dsm_data == 0] = 0  # Keep 0 as invalid, or change based on your data
            
            # Process RGB with HuggingFace processor (will resize to 384x384 automatically)
            if self.transform:
                # If using custom transform, handle resizing manually
                if self.resize_to_model_input:
                    rgb_image = rgb_image.resize((384, 384), Image.LANCZOS)
                rgb_tensor = self.transform(rgb_image)
            else:
                # Use HF processor - this automatically resizes to model's expected size (384x384)
                processed = processor(rgb_image, return_tensors="pt")
                rgb_tensor = processed["pixel_values"].squeeze(0)  # Shape: [3, 384, 384]
            
            # Process DSM to match RGB dimensions
            if self.resize_to_model_input:
                # Resize DSM from 512x512 to 384x384 to match model input
                dsm_pil = Image.fromarray(dsm_data.astype(np.uint8))
                dsm_pil = dsm_pil.resize((384, 384), Image.LANCZOS)
                dsm_data = np.array(dsm_pil, dtype=np.float32)
                dsm_pil.close()
            
            # Normalize DSM
            if self.dsm_std != 0:
                dsm_data = (dsm_data - self.dsm_mean) / self.dsm_std
            
            # Convert DSM to tensor with shape [1, 384, 384]
            dsm_tensor = torch.FloatTensor(dsm_data).unsqueeze(0)
            
            # Clean up
            rgb_image.close()
            dsm_image.close()
            
            return rgb_tensor, dsm_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (num={num}): {e}")
            # Return dummy tensors with correct shapes
            dummy_rgb = torch.zeros(3, 384, 384)
            dummy_dsm = torch.zeros(1, 384, 384)
            return dummy_rgb, dummy_dsm