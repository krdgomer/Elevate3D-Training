import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from tqdm import tqdm
from PIL import Image
from src.maskrcnn.inference.predict import predict_mask  
from src.maskrcnn.tests.metrics import compute_ap_per_image, compute_iou    
from src.maskrcnn.model import get_model
import torch


rgb_dir = "data/processed/maskrcnn/test-train-val-split/test/rgb"
gt_dir = "data/processed/maskrcnn/test-train-val-split/test/gti"
weights_path = "src/maskrcnn/models/v0.3/maskrcnn_weights.pth"

model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


model.load_state_dict(torch.load(weights_path, map_location=device))


image_filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

aps_050 = []
aps_075 = []
aps_all = []


iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]


for filename in tqdm(image_filenames):
    image_path = os.path.join(rgb_dir, filename)
    gt_path = os.path.join(gt_dir, filename.replace("_rgb", "_gti"))

    pred_masks, pred_scores, _ = predict_mask(model, image_path, device=device, threshold=0.5)

    gt_mask = np.array(Image.open(gt_path))

    
    ap_dict = compute_ap_per_image(pred_masks, pred_scores, gt_mask, iou_thresholds=iou_thresholds)
    aps_050.append(ap_dict.get(0.5, 0.0))
    aps_075.append(ap_dict.get(0.75, 0.0))
    aps_all.append(np.mean([ap_dict.get(t, 0.0) for t in iou_thresholds]))

# Final metrics
print("Mean AP@0.50:", np.mean(aps_050))
print("Mean AP@0.75:", np.mean(aps_075))
print("Mean AP@[.50:.95]:", np.mean(aps_all))
