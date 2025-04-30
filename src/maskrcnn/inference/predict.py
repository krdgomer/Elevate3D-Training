import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from src.maskrcnn.inference.post_process import post_process


def predict_mask(model,image_path,device="cpu",threshold=0.5):
    """
    Predict masks for a given image using a trained Mask R-CNN model.

    Args:
        model (MaskRCNN): MaskRCNN model to use for prediction
        image_path (str): Path to the image file
        device (str, optional): Device to run the model on. Defaults to "cpu".
        threshold (float, optional): Confidence threshold for filtering masks. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the predicted masks and their corresponding scores.
    
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize and convert image to tensor
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Run model prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Extract masks and scores
    masks = (prediction["masks"].squeeze(1) > 0.5).cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    filtered_indices = np.where(scores >= threshold)[0]
    filtered_masks = masks[filtered_indices]
    filtered_scores = scores[filtered_indices]

    # Post-process EACH mask individually
    processed_masks = np.array([post_process(mask) for mask in filtered_masks])

    return processed_masks, filtered_scores ,image


"""if __name__ == "__main__":

    from src.maskrcnn.model import get_model
    import src.configs.maskrcnn_config as cfg
    import matplotlib.pyplot as plt
    # Load model
    model_path = "src/maskrcnn/models/v0.2/maskrcnn_weights.pth"
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    masks,scores,image=predict_mask(model, "data/processed/maskrcnn/test-train-val-split/test/rgb/image109_rgb.png", device=cfg.DEVICE, threshold=0.5)

    overlay_pred = image.copy()
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = np.random.randint(0, 255, (len(masks), 3))

    for idx, (mask, score) in enumerate(zip(masks, scores)):
        if mask.sum() == 0:
            continue  # Skip empty masks

        color = colors[idx].tolist()
        overlay_pred[mask.astype(bool)] = color  # <- FIXED

        # Find center of mask
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        center_x = int(xs.mean())
        center_y = int(ys.mean())

        # Draw score text
        cv2.putText(
            overlay_pred,
            f"{score:.2f}",
            (center_x, center_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )



    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(overlay_pred)
    plt.title(f"Predicted Masks > {0.5} Score")
    plt.axis('off')


    plt.tight_layout()
    plt.show()"""



