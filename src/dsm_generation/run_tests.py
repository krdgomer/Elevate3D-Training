import torch
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import glob
from src.dsm_generation.predict import predict_dsm
from src.dsm_generation.calculate_errors import ErrorCalculator
from src.dsm_generation.visualize import save_combined_image, visualize_predictions

def load_test_images(test_dir):
    """Load test images from directory and split into RGB and DSM parts"""
    image_paths = glob.glob(os.path.join(test_dir, "*.png")) + glob.glob(os.path.join(test_dir, "*.jpg"))
    image_paths.sort()
    
    input_images = []
    target_images = []
    
    for img_path in image_paths:
        # Load the combined image (1024x512)
        combined_img = cv2.imread(img_path)
        if combined_img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
            
        # Split into left (RGB 512x512) and right (DSM 512x512) parts
        rgb_part = combined_img[:, :512, :]  # Left side - RGB
        dsm_part = combined_img[:, 512:, :]  # Right side - DSM
        
        # Convert RGB to grayscale (as done in predict_dsm)
        rgb_gray = cv2.cvtColor(rgb_part, cv2.COLOR_BGR2GRAY)
        
        # Store as numpy arrays (not tensors) since predict_dsm expects numpy
        input_images.append(rgb_gray)
        
        # For target, keep as numpy but convert to single channel if needed
        if len(dsm_part.shape) == 3:
            dsm_gray = cv2.cvtColor(dsm_part, cv2.COLOR_BGR2GRAY)
            target_images.append(dsm_gray)
        else:
            target_images.append(dsm_part)
    
    return input_images, target_images, image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test configuration")
    parser.add_argument("--version", type=str, required=True, help="Model Version")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--no_post_process", action="store_true", help="Disable post-processing")
    args = parser.parse_args()

    MODEL_VERSION = args.version
    SAVE_PREDICTIONS = args.save_predictions
    POST_PROCESS = not args.no_post_process

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load generator weights
    generator_path = f"src/dsm_generation/models/{MODEL_VERSION}/weights/gen.pth.tar"
    tests_save_dir = f"src/dsm_generation/models/{MODEL_VERSION}/test_results"
    predictions_save_dir = f"src/dsm_generation/models/{MODEL_VERSION}/predictions"
    test_dir = "src/dsm_generation/datasets/test"

    # Load test images as numpy arrays
    print("Loading test images...")
    input_images_np, target_images_np, image_paths = load_test_images(test_dir)
    
    os.makedirs(tests_save_dir, exist_ok=True)
    os.makedirs(predictions_save_dir, exist_ok=True)

    print(f"Found {len(input_images_np)} test images")
    
    # Convert target images to tensors for visualization functions
    target_images_tensor = []
    for target_np in target_images_np:
        target_tensor = torch.from_numpy(target_np.astype(np.float32) / 255.0)
        target_images_tensor.append(target_tensor.unsqueeze(0))  # Add channel dimension
    
    for idx in tqdm(range(len(input_images_np)), desc="Processing test images"):
        input_img_np = input_images_np[idx]
        target_img_np = target_images_np[idx]
        target_img_tensor = target_images_tensor[idx]
        
        # Predict DSM - pass numpy array directly
        pred_dsm = predict_dsm(input_img_np, generator_path, post_process=POST_PROCESS)
        
        # Convert prediction to tensor for visualization functions
        pred_dsm_tensor = torch.from_numpy(pred_dsm.astype(np.float32) / 255.0).unsqueeze(0)
        
        # Convert input to tensor for visualization
        input_img_tensor = torch.from_numpy(input_img_np.astype(np.float32) / 255.0).unsqueeze(0)

        if idx % 50 == 0:   
            visualize_predictions(input_img_tensor, target_img_tensor, pred_dsm_tensor, 
                                 f'{tests_save_dir}/sample_{idx}.png')

        if SAVE_PREDICTIONS:
            # Save both raw and processed predictions
            cv2.imwrite(f'{predictions_save_dir}/prediction_raw_{idx}.png', pred_dsm)
            
            # Also save the target for comparison
            cv2.imwrite(f'{predictions_save_dir}/target_{idx}.png', target_img_np)
            
            # Save combined visualization
            save_combined_image(target_img_tensor, pred_dsm_tensor, 
                               f'{predictions_save_dir}/prediction_combined_{idx}.png') 

    error_calculator = ErrorCalculator(predictions_save_dir, MODEL_VERSION)
    error_calculator.calculate_errors(tests_save_dir)