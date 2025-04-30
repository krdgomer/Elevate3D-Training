from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, FastRCNNPredictor

def get_model():
    # Load ResNet-101 backbone with FPN
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    
    # Initialize Mask R-CNN
    model = MaskRCNN(
        backbone,
        num_classes=2,  # Background + your class
        min_size=800,   # Better for building detection
        max_size=1333
    )
    
    # Modify predictor heads (keep your existing architecture)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    
    return model