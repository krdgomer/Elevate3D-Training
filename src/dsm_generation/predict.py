import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.dsm_generation.generator import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

def load_generator(model_path, device="cpu"):
    print(f"Loading generator model from {model_path}...")
    
    generator = Generator().to(device)  
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:  
        generator.load_state_dict(checkpoint["state_dict"])  
    else:
        generator.load_state_dict(checkpoint)  

    generator.eval()  
    return generator

def normalize_safe(array):
    """
    Safely normalize array to [0,1] range, handling edge cases
    """
    array_min = np.min(array)
    array_max = np.max(array)
    
    if array_max == array_min:
        return np.zeros_like(array)
    
    return (array - array_min) / (array_max - array_min)

def denoise_dsm(dsm_array, kernel_size=3):
    """Remove salt-and-pepper noise while preserving edges."""
    return cv2.medianBlur(dsm_array, kernel_size)

def smooth_preserve_edges(dsm_array):
    """Smooth homogeneous regions while keeping sharp edges intact."""
    # Convert to float32 for bilateral filter
    dsm_float = dsm_array.astype(np.float32)
    smoothed = cv2.bilateralFilter(dsm_float, d=9, sigmaColor=0.1, sigmaSpace=15)
    return smoothed.astype(np.uint8)

def morphological_refinement(dsm_array):
    """Clean up small artifacts and holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close small holes
    closed = cv2.morphologyEx(dsm_array, cv2.MORPH_CLOSE, kernel)
    # Remove small objects
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def guided_filter_processing(dsm_raw, guide_image=None, radius=15, eps=0.01):
    """
    Use guided filter for edge-preserving smoothing.
    If no guide image is provided, use the DSM itself as guide.
    """
    if guide_image is None:
        guide_image = dsm_raw
    
    # Guided filter requires float32 in [0,1] range
    dsm_float = dsm_raw.astype(np.float32) / 255.0
    guide_float = guide_image.astype(np.float32) / 255.0
    
    # Apply guided filter
    filtered = cv2.ximgproc.guidedFilter(guide_float, dsm_float, radius, eps)
    
    # Convert back to uint8
    return (filtered * 255).astype(np.uint8)

def post_process_dsm(dsm_raw, building_mask=None):
    """
    Simplified but effective post-processing pipeline for DSM refinement.
    """
    # Step 1: Initial denoising
    dsm_clean = denoise_dsm(dsm_raw)
    
    # Step 2: Edge-preserving smoothing using guided filter
    try:
        # Try guided filter (more advanced)
        dsm_smooth = guided_filter_processing(dsm_clean)
    except:
        # Fallback to bilateral filter if guided filter not available
        dsm_smooth = smooth_preserve_edges(dsm_clean)
    
    # Step 3: Final morphological cleanup
    dsm_final = morphological_refinement(dsm_smooth)
    
    return dsm_final

def visualize_dsm_comparison(original_img, dsm_raw, dsm_processed, save_path=None):
    """
    Create a comparison visualization of original image, raw DSM, and processed DSM.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Raw DSM
    im1 = axes[1].imshow(dsm_raw, cmap='terrain')
    axes[1].set_title('Raw DSM Prediction', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Processed DSM
    im2 = axes[2].imshow(dsm_processed, cmap='terrain')
    axes[2].set_title('Processed DSM', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def predict_dsm(input_img, model_path, post_process=True):
    """
    Predict DSM from an image and return it as an OpenCV image (uint8).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_generator(model_path, device=device)
    model.eval()

    # Albumentations transforms
    transform = A.Compose([
        A.Resize(width=512, height=512),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    augmented = transform(image=input_img)
    input_tensor = augmented["image"].to(device)

    with torch.no_grad():
        pred_dsms = model(input_tensor.unsqueeze(0))  # Add batch dimension
        pred_np = pred_dsms.cpu().numpy().squeeze()  # Remove batch dimension

        # Normalize the predicted DSM to [0, 255] for visualization
        pred_norm = (normalize_safe(pred_np) * 255).astype(np.uint8)
        
        # Apply post-processing if requested
        if post_process:
            print("Applying post-processing...")
            pred_norm = post_process_dsm(pred_norm)
        
        return pred_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DSM from satellite image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="models/generator.pth", 
                       help="Path to generator model")
    parser.add_argument("--output_path", type=str, default="dsm_output.png", 
                       help="Path to save output visualization")
    parser.add_argument("--no_post_process", action="store_true", 
                       help="Disable post-processing")
    
    args = parser.parse_args()

    # Load input image
    print(f"Loading image from {args.img_path}...")
    input_img = cv2.imread(args.img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # Predict DSM
    print("Predicting DSM...")
    dsm_raw = predict_dsm(input_img, args.model_path, post_process=False)
    
    # Apply post-processing (unless disabled)
    if not args.no_post_process:
        dsm_processed = post_process_dsm(dsm_raw)
    else:
        dsm_processed = dsm_raw
    
    # Save the processed DSM
    cv2.imwrite("dsm_processed.tif", dsm_processed)
    print("Processed DSM saved as dsm_processed.tif")
    
    # Create and display comparison visualization
    print("Generating visualization...")
    visualize_dsm_comparison(input_img, dsm_raw, dsm_processed, args.output_path)
    
    # Print some statistics
    print("\nDSM Statistics:")
    print(f"Raw DSM range: {dsm_raw.min()} - {dsm_raw.max()}")
    print(f"Processed DSM range: {dsm_processed.min()} - {dsm_processed.max()}")
    print(f"Mean value change: {np.mean(dsm_processed) - np.mean(dsm_raw):.2f}")

    # Optional: Show individual DSMs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(dsm_raw, cmap='Grays_r')
    plt.title('Raw DSM')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(dsm_processed, cmap='Grays_r')
    plt.title('Processed DSM')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()