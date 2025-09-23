# inference_demo.py

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from transformers import DPTImageProcessor
import argparse
import os

# Add the project root to Python path
import sys
sys.path.append('/content/drive/MyDrive/ProjeDosyalari/Elevate3D-Training')

from src.depth_estimation.config import cfg
from src.depth_estimation.model import setup_model, load_saved_model

def load_and_preprocess_image(image_path, processor):
    """Load and preprocess a single image for inference."""
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess using the same method as during training
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].squeeze(0), image

def predict_depth(model, processed_image, device):
    """Predict depth map for a single image."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension and move to device
        inputs = processed_image.unsqueeze(0).to(device)
        outputs = model(pixel_values=inputs)
        prediction = outputs.predicted_depth
    
    # Remove batch dimension and convert to numpy
    return prediction.squeeze().cpu().numpy()

def denormalize_depth(normalized_depth, mean, std):
    """Convert normalized depth back to original scale."""
    return (normalized_depth * std) + mean

def create_side_by_side_plot(original_image, depth_map, output_path=None):
    """Create a side-by-side comparison of original image and depth map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original image
    ax1.imshow(original_image)
    ax1.set_title('Original RGB Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot depth map
    im = ax2.imshow(depth_map, cmap='viridis')  # 'viridis', 'plasma', 'magma' are good colormaps
    ax2.set_title('Predicted Depth Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add colorbar for depth values
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Comparison plot saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run depth estimation on a single image')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to the input RGB image')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                       help='Directory to save output results')
    parser.add_argument('--dsm_mean', type=float, default=0.0,
                       help='Mean used for DSM normalization during training')
    parser.add_argument('--dsm_std', type=float, default=1.0,
                       help='Standard deviation used for DSM normalization during training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processor
    processor = DPTImageProcessor.from_pretrained(cfg.PRETRAINED_MODEL_NAME)
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    processed_image, original_image = load_and_preprocess_image(args.image_path, processor)
    
    # Load model
    print("Loading trained model...")
    model = setup_model(device)
    
    if args.model_path:
        model, _ = load_saved_model(model, path=args.model_path)
    else:
        # Load default best model
        model, _ = load_saved_model(model)
    
    # Run prediction
    print("Running depth prediction...")
    normalized_depth = predict_depth(model, processed_image, device)
    
    # Denormalize to get actual depth values
    depth_map = denormalize_depth(normalized_depth, args.dsm_mean, args.dsm_std)
    
    # Display results
    print(f"Depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")
    print(f"Average depth: {depth_map.mean():.2f}m")
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_path = os.path.join(args.output_dir, f'{base_name}_depth_comparison.png')
    
    # Create and save side-by-side plot
    create_side_by_side_plot(original_image, depth_map, output_path)
    
    # Save raw depth map as numpy array for later use
    depth_map_path = os.path.join(args.output_dir, f'{base_name}_depth_map.npy')
    np.save(depth_map_path, depth_map)
    print(f"Depth map array saved to: {depth_map_path}")
    
    # Save depth map as image
    depth_image_path = os.path.join(args.output_dir, f'{base_name}_depth.png')
    plt.imsave(depth_image_path, depth_map, cmap='viridis')
    print(f"Depth map image saved to: {depth_image_path}")

if __name__ == "__main__":
    main()