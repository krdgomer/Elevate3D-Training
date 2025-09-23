import os
import numpy as np
from PIL import Image, ImageDraw


# Helper: overlay masks on image
def overlay_masks(base_image, masks, color=(255, 0, 0), scores=None):
    img = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, mask in enumerate(masks):
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size)
        color_mask = Image.new("RGBA", img.size, color + (100,))
        overlay.paste(color_mask, (0, 0), mask)
        
        if scores is not None:
            score = scores[i]
            draw.text((5, 5 + 15 * i), f"{score:.2f}", fill=(255, 255, 255, 255))

    return Image.alpha_composite(img, overlay).convert("RGB")

def visualize_prediction(image, gt_mask, pred_masks, pred_scores, output_dir, filename):
    # Ground truth instance masks
    gt_instances = [(gt_mask == id).astype(np.uint8) for id in np.unique(gt_mask) if id != 0]

    # Overlays
    img_gt = overlay_masks(image, gt_instances, color=(0, 255, 0))
    img_pred = overlay_masks(image, pred_masks, color=(255, 0, 0), scores=pred_scores)

    # Combine side-by-side
    combined = Image.new("RGB", (image.width * 3, image.height))
    combined.paste(image, (0, 0))
    combined.paste(img_gt, (image.width, 0))
    combined.paste(img_pred, (image.width * 2, 0))

    combined.save(os.path.join(output_dir, filename.replace("_rgb", "_viz")))