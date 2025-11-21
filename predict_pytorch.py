"""
Warranty Card Prediction - PyTorch Inference
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (same as training)
# ══════════════════════════════════════════════════════════════════════════════

class WarrantyClassifier(nn.Module):
    def __init__(self):
        super(WarrantyClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class WarrantyClassifierPyTorch:
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = device
        self.class_names = ['authentic', 'fraudulent']
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load model
        print(f"📥 Loading model from {model_path}...")
        self.model = WarrantyClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def predict(self, image_path):
        """Predict if warranty card is authentic or fraudulent"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
        
        # Get results
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        result = {
            'image_path': str(image_path),
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'authentic': probabilities[0].item(),
                'fraudulent': probabilities[1].item()
            },
            'is_authentic': predicted_class == 'authentic',
            'is_fraudulent': predicted_class == 'fraudulent'
        }
        
        return result

# ══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default='models/best_model_pytorch.pth')
    args = parser.parse_args()
    
    classifier = WarrantyClassifierPyTorch(model_path=args.model)
    
    if args.image:
        result = classifier.predict(args.image)
        print(f"\n✅ {result['prediction'].upper()} ({result['confidence']*100:.1f}%)")
    
    elif args.batch:
        folder = Path(args.batch)
        for img_path in folder.glob('*.jpg'):
            result = classifier.predict(img_path)
            icon = "✅" if result['is_authentic'] else "❌"
            print(f"{icon} {img_path.name:30s} → {result['prediction'].upper():12s} ({result['confidence']*100:.1f}%)")
