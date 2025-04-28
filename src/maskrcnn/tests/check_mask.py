import os
import numpy as np
import cv2

def check_masks(folder_path):
    """Check all masks in a folder and print names of empty ones."""
    for filename in os.listdir(folder_path):
        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"Could not read {filename}")
            continue

        if not np.any(mask > 0):  # All zeros
            print(f"Empty mask: {filename}")

# Example usage
mask_folder = "data/processed/maskrcnn/organized/test/gti"
check_masks(mask_folder)
