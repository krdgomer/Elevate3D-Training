import numpy as np
import cv2
from skimage.measure import label
import os

def convert_to_instance_masks(input_dir, output_dir):
    """Convert single-value building masks to instance masks with unique IDs"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.png'):
            continue
            
        # Read 16-bit PNG
        mask = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        # Verify it's 16-bit
        assert mask.dtype == np.uint16, f"Expected 16-bit image, got {mask.dtype}"
        
        # Create binary mask (255 -> 1, others -> 0)
        binary_mask = (mask == 255).astype(np.uint8)
        
        # Label connected components (8-connectivity)
        labeled_mask, num_buildings = label(binary_mask, connectivity=2, return_num=True)
        
        # Convert to 16-bit and scale IDs (optional)
        instance_mask = labeled_mask.astype(np.uint16)
        
        # Save new mask
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, instance_mask)
        
        print(f"Processed {filename}: {num_buildings} buildings found")

# Usage
convert_to_instance_masks(
    input_dir="data/processed/maskrcnn/inria/gt",
    output_dir="data/processed/maskrcnn/inria/gti"
)