import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_ap_per_image(pred_masks, pred_scores, gt_mask, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Compute AP at multiple IoU thresholds for one image (COCO-style).
    
    Returns:
        dict: {threshold: AP at that threshold}
    """
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]  # Remove background
    gt_instances = [(gt_mask == i) for i in gt_ids]
    num_gt = len(gt_instances)
    
    if len(pred_masks) == 0 or num_gt == 0:
        return {t: 0.0 for t in iou_thresholds}
    
    # Sort predictions by score (highest first)
    pred_masks = np.array(pred_masks)
    pred_scores = np.array(pred_scores)
    sorted_indices = np.argsort(-pred_scores)
    pred_masks = pred_masks[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    ap_results = {}
    
    for thresh in iou_thresholds:
        # Initialize variables for this threshold
        tp = np.zeros(len(pred_masks))
        fp = np.zeros(len(pred_masks))
        matched_gt = set()
        
        # For each prediction, find best matching GT
        for pred_idx, pred in enumerate(pred_masks):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_instances):
                if gt_idx in matched_gt:
                    continue
                
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= thresh:
                matched_gt.add(best_gt_idx)
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Append (0,1) and (1,0) to ensure full curve
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        
        # Smooth precision-recall curve
        for i in range(len(precisions)-1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])
        
        # Compute AP (area under PR curve)
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        ap_results[thresh] = ap

    result = {}
    for t in iou_thresholds:
        result[t] = ap_results.get(t, 0.0)  # Use calculated AP or default
    
    return result
