import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import os

class MedicalImageAI:
    def __init__(self):
       
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.classes = ["Normal", "Pneumonia", "COVID-19"]

    def _image_quality_check(self, img_np):
        
        mean = img_np.mean()
        std = img_np.std()

        if mean < 20 or mean > 235:
            return False
        if std < 10:
            return False
        return True

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        
        if not self._image_quality_check(img_np):
            return {
                "prediction": "Invalid",
                "confidence": 0.0,
                "status": "Poor image quality / blank image"
            }

        
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        confidence = torch.max(probs).item()
        prediction_idx = torch.argmax(probs).item()

        
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

        if confidence < 0.60 or entropy > 1.0:
            return {
                "prediction": "Unknown",
                "confidence": round(confidence, 3),
                "status": "Model uncertain — requires clinical review"
            }

        return {
            "prediction": self.classes[prediction_idx],
            "confidence": round(confidence, 3),
            "status": "DL-CNN Inference"
        }
