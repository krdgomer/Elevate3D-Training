import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from src.building_segmentation.predict import predict_mask  
from src.building_segmentation.metrics import compute_ap_per_image, compute_iou    
from src.building_segmentation.model import get_model
from src.building_segmentation.visualize import visualize_prediction
import torch
import argparse
import logging
from datetime import datetime

def run_tests(rgb_dir, gt_dir, model_dir, visualize=False):
    # Ensure model_dir exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(model_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Find model weights in model_dir
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') or f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    weights_path = os.path.join(model_dir, model_files[0])
    if len(model_files) > 1:
        logging.warning(f"Multiple model files found in {model_dir}. Using {model_files[0]}")
    
    logging.info(f"Using model weights: {weights_path}")
    logging.info(f"RGB images directory: {rgb_dir}")
    logging.info(f"Ground truth masks directory: {gt_dir}")
    
    # Set up visualization directory inside model_dir
    output_viz_dir = os.path.join(model_dir, "visualizations")
    if visualize:
        os.makedirs(output_viz_dir, exist_ok=True)
        logging.info(f"Visualizations will be saved to: {output_viz_dir}")

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    model.load_state_dict(torch.load(weights_path, map_location=device))

    image_filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    logging.info(f"Found {len(image_filenames)} test images")

    # Metrics storage
    aps_050 = []
    aps_075 = []
    aps_all = []

    iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    for filename in tqdm(image_filenames):
        image_path = os.path.join(rgb_dir, filename)
        gt_path = os.path.join(gt_dir, filename.replace("_rgb", "_gti"))

        pred_masks, pred_scores, _ = predict_mask(model, image_path, device=device, threshold=0.5)

        gt_mask = np.array(Image.open(gt_path))

        if visualize:
            image = Image.open(image_path).convert("RGB")
            visualize_prediction(image, gt_mask, pred_masks, pred_scores, output_viz_dir, filename)

        ap_dict = compute_ap_per_image(pred_masks, pred_scores, gt_mask, iou_thresholds=iou_thresholds)
        aps_050.append(ap_dict.get(0.5, 0.0))
        aps_075.append(ap_dict.get(0.75, 0.0))
        aps_all.append(np.mean([ap_dict.get(t, 0.0) for t in iou_thresholds]))

    # Final metrics
    mean_ap_050 = np.mean(aps_050)
    mean_ap_075 = np.mean(aps_075)
    mean_ap_all = np.mean(aps_all)
    
    logging.info("=" * 50)
    logging.info("TEST RESULTS SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Mean AP@0.50: {mean_ap_050:.4f}")
    logging.info(f"Mean AP@0.75: {mean_ap_075:.4f}")
    logging.info(f"Mean AP@[.50:.95]: {mean_ap_all:.4f}")
    logging.info(f"Log file saved to: {log_file}")
    if visualize:
        logging.info(f"Visualizations saved to: {output_viz_dir}")
    logging.info("=" * 50)
    
    print("=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Mean AP@0.50: {mean_ap_050:.4f}")
    print(f"Mean AP@0.75: {mean_ap_075:.4f}")
    print(f"Mean AP@[.50:.95]: {mean_ap_all:.4f}")
    print(f"Log file saved to: {log_file}")
    if visualize:
        print(f"Visualizations saved to: {output_viz_dir}")
    print("=" * 50)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test Configuration")
    parser.add_argument(
        "--rgb_dir",
        type=str,
        required=True,
        help="Path to the directory containing RGB test images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Path to the directory containing ground truth masks",
    )
    args = parser.parse_args()

    RGB_IMAGES_DIR = args.rgb_dir
    MASK_IMAGES_DIR = args.mask_dir
    
    # Import config here to avoid circular imports
    from src.building_segmentation.config import MODEL_DIR, VISUALIZE
    
    # Convert relative MODEL_DIR to absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_model_dir = os.path.abspath(os.path.join(current_dir, MODEL_DIR))
    
    run_tests(RGB_IMAGES_DIR, MASK_IMAGES_DIR, absolute_model_dir, VISUALIZE)