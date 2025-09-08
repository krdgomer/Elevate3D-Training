import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Simple Roof Type Predictor
class SimpleRoofPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define class names (should match your training)
        self.class_names = ['complex', 'flat', 'gable', 'hip', 'pyramid']
        
        # Load model
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(235),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained model"""
        # Create model architecture (EfficientNet-B0 like during training)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.class_names))
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def predict(self, image_path):
        """Predict roof type for an image"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probs = probabilities[0].cpu().numpy()
            
            return predicted_class, confidence_score, all_probs, image
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None

# Main function - just input your paths here
if __name__ == "__main__":
    # === JUST CHANGE THESE TWO PATHS ===
    MODEL_PATH = "src/roof_classification/roof_classifier_improved.pth"  # Path to your model weights
    IMAGE_PATH = "src/roof_classification/test2.png"                 # Path to your test image
    # ===================================
    
    # Create predictor and make prediction
    predictor = SimpleRoofPredictor(MODEL_PATH)
    pred_class, confidence, probs, image = predictor.predict(IMAGE_PATH)
    
    if pred_class is not None:
        # Print results
        print("\n" + "="*50)
        print("ROOF TYPE PREDICTION")
        print("="*50)
        print(f"Image: {IMAGE_PATH}")
        print(f"Predicted: {pred_class}")
        print(f"Confidence: {confidence:.3f}")
        print("\nAll probabilities:")
        for i, cls in enumerate(predictor.class_names):
            print(f"  {cls}: {probs[i]:.4f}")
        print("="*50)
        
        # Show visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f'Predicted: {pred_class}\nConfidence: {confidence:.3f}')
        ax1.axis('off')
        
        # Show probabilities
        colors = ['red' if i == predictor.class_names.index(pred_class) else 'blue' 
                 for i in range(len(predictor.class_names))]
        bars = ax2.barh(predictor.class_names, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        
        # Add values to bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Prediction failed. Please check your file paths.")