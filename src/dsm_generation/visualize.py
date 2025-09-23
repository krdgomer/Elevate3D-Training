import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.dsm_generation.predict import normalize_safe

def save_combined_image(input_img, pred_dsm, save_path):
    """
    Save combined image with input image on the left and predicted DSM on the right.
    No axes, titles, or colorbars are included.
    """
    # Convert tensors to numpy arrays
    input_np = input_img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    pred_np = pred_dsm.cpu().numpy().squeeze()  # Remove batch and channel dimensions

    # Normalize the predicted DSM to [0, 255] for visualization
    pred_norm = normalize_safe(pred_np) * 255
    pred_norm = pred_norm.astype(np.uint8)

    # Convert input image to uint8 if it's not already
    if input_np.dtype != np.uint8:
        input_np = (normalize_safe(input_np) * 255).astype(np.uint8)

    # Create a blank canvas to combine the images
    height = max(input_np.shape[0], pred_norm.shape[0])
    width = input_np.shape[1] + pred_norm.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Place the input image on the left
    combined_image[:input_np.shape[0], :input_np.shape[1], :] = input_np

    # Place the predicted DSM on the right (convert grayscale to RGB)
    pred_rgb = np.stack([pred_norm] * 3, axis=-1)  # Convert grayscale to RGB
    combined_image[:pred_rgb.shape[0], input_np.shape[1]:, :] = pred_rgb

    # Save the combined image using PIL
    combined_pil = Image.fromarray(combined_image)
    combined_pil.save(save_path)

def visualize_predictions(input_img, target_dsm, pred_dsm, save_path):
    """
    Visualize input image, target DSM, and predicted DSM side by side
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_img.cpu().numpy().transpose(1, 2, 0))
    plt.title('Input Satellite Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(target_dsm.cpu().numpy().squeeze(), cmap='gray')
    plt.colorbar(label='Elevation')
    plt.title('Target DSM')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_dsm.cpu().numpy().squeeze(), cmap='gray')
    plt.colorbar(label='Elevation')
    plt.title('Predicted DSM')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()