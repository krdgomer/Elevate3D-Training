import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def separate_connected_buildings(mask, min_distance=10):
    """
    Use watershed segmentation to separate connected/merged building instances.
    
    Args:
        mask (np.ndarray): Binary mask of connected buildings
        min_distance (int): Minimum distance between markers for watershed
        
    Returns:
        list: List of separated building masks
    """
    # Compute distance transform
    distance = ndi.distance_transform_edt(mask)
    
    # Find local maxima as markers
    local_maxima = peak_local_max(
        distance, 
        min_distance=min_distance, 
        indices=False, 
        footprint=np.ones((3, 3)),
        labels=mask
    )
    
    # Label markers
    markers = ndi.label(local_maxima)[0]
    
    # Apply watershed
    labels = watershed(-distance, markers, mask=mask)
    
    # Extract individual building masks
    separated_masks = []
    for label_val in np.unique(labels):
        if label_val == 0:  # Skip background
            continue
        building_mask = (labels == label_val).astype(np.uint8)
        separated_masks.append(building_mask)
    
    return separated_masks

def regularize_footprint(contour, epsilon_factor=0.01):
    """
    Regularize building footprint using Douglas-Peucker algorithm and simplify geometry.
    
    Args:
        contour (np.ndarray): Building contour points
        epsilon_factor (float): Approximation factor (0.01-0.05)
        
    Returns:
        np.ndarray: Regularized contour
    """
    # Calculate epsilon based on contour perimeter
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    
    # Douglas-Peucker approximation
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Further simplify if too many points (optional)
    if len(approx) > 20:  # If still too complex
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
    return approx

def merge_close_buildings(masks, distance_threshold=5, area_threshold=50):
    """
    Merge buildings that are too close and likely part of the same structure.
    
    Args:
        masks (list): List of building masks
        distance_threshold (int): Maximum distance for merging
        area_threshold (int): Minimum area to consider for merging
        
    Returns:
        list: List of merged building masks
    """
    if not masks:
        return []
    
    # Convert masks to polygons
    polygons = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            poly = Polygon(contours[0].squeeze())
            if poly.area > area_threshold:  # Only consider sufficiently large buildings
                polygons.append(poly)
    
    if not polygons:
        return []
    
    # Merge polygons that are close to each other
    merged_polygons = []
    current_poly = polygons[0]
    
    for i in range(1, len(polygons)):
        distance = current_poly.distance(polygons[i])
        if distance < distance_threshold:
            # Merge close polygons
            current_poly = unary_union([current_poly, polygons[i]])
        else:
            merged_polygons.append(current_poly)
            current_poly = polygons[i]
    
    merged_polygons.append(current_poly)
    
    # Convert back to masks
    merged_masks = []
    height, width = masks[0].shape
    for poly in merged_polygons:
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                mask = polygon_to_mask(sub_poly, height, width)
                merged_masks.append(mask)
        else:
            mask = polygon_to_mask(poly, height, width)
            merged_masks.append(mask)
    
    return merged_masks

def polygon_to_mask(polygon, height, width):
    """Convert Shapely polygon to binary mask."""
    from shapely.geometry import mapping
    import rasterio.features
    
    geom = mapping(polygon)
    mask = rasterio.features.rasterize(
        [geom],
        out_shape=(height, width),
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    return mask

def post_process(mask, use_watershed=True, regularize=True, min_building_area=50):
    """
    Enhanced post-processing with watershed segmentation and regularization.
    
    Args:
        mask (np.ndarray): Raw predicted mask
        use_watershed (bool): Whether to use watershed segmentation
        regularize (bool): Whether to regularize footprints
        min_building_area (int): Minimum area to consider as a building
        
    Returns:
        np.ndarray: Processed binary mask
    """
    # Initial thresholding
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Find all contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    
    final_mask = np.zeros_like(mask)
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_building_area:
            continue
            
        # Create individual mask for this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        if use_watershed:
            # Try to separate connected buildings
            try:
                separated_masks = separate_connected_buildings(contour_mask)
                
                for building_mask in separated_masks:
                    if np.sum(building_mask) < min_building_area:
                        continue
                        
                    # Find contour of separated building
                    building_contours, _ = cv2.findContours(
                        building_mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if building_contours:
                        building_contour = building_contours[0]
                        
                        if regularize:
                            # Regularize the footprint
                            regularized_contour = regularize_footprint(building_contour)
                            cv2.fillPoly(final_mask, [regularized_contour], 255)
                        else:
                            cv2.fillPoly(final_mask, [building_contour], 255)
                            
            except Exception as e:
                # Fallback: use original contour if watershed fails
                if regularize:
                    regularized_contour = regularize_footprint(contour)
                    cv2.fillPoly(final_mask, [regularized_contour], 255)
                else:
                    cv2.fillPoly(final_mask, [contour], 255)
        else:
            # Without watershed, just regularize the original contour
            if regularize:
                regularized_contour = regularize_footprint(contour)
                cv2.fillPoly(final_mask, [regularized_contour], 255)
            else:
                cv2.fillPoly(final_mask, [contour], 255)
    
    return final_mask // 255

def visualize_processing_steps(original_mask, processed_mask, image=None):
    """Visualize the processing steps for debugging."""
    fig, axes = plt.subplots(1, 3 if image is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title('Original Prediction')
    axes[0].axis('off')
    
    axes[1].imshow(processed_mask, cmap='gray')
    axes[1].set_title('Processed Mask')
    axes[1].axis('off')
    
    if image is not None:
        # Overlay processed mask on original image
        overlay = image.copy()
        overlay[processed_mask > 0] = [255, 0, 0]  # Red overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay on Image')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_mask(model, image_path, device="cpu", threshold=0.5, use_watershed=True, 
                 regularize=True, min_building_area=100, visualize=False):
    """
    Enhanced prediction with advanced post-processing.
    
    Args:
        model: Mask R-CNN model
        image_path (str): Path to input image
        device (str): Device for inference
        threshold (float): Confidence threshold
        use_watershed (bool): Use watershed segmentation
        regularize (bool): Regularize footprints
        min_building_area (int): Minimum building area
        visualize (bool): Show processing steps
        
    Returns:
        tuple: (processed_masks, scores, image)
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Extract masks and scores
    masks = (prediction["masks"].squeeze(1) > 0.5).cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    
    # Filter by confidence
    filtered_indices = np.where(scores >= threshold)[0]
    filtered_masks = masks[filtered_indices]
    filtered_scores = scores[filtered_indices]
    
    # Enhanced post-processing
    processed_masks = []
    for i, mask in enumerate(filtered_masks):
        processed_mask = post_process(
            mask, 
            use_watershed=use_watershed,
            regularize=regularize,
            min_building_area=min_building_area
        )
        processed_masks.append(processed_mask)
        
        if visualize:
            print(f"Building {i+1}, Score: {filtered_scores[i]:.3f}")
            visualize_processing_steps(mask, processed_mask, image)
    
    return np.array(processed_masks), filtered_scores, image