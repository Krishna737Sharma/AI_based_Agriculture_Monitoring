import io
import os
import torch
import torchvision
import numpy as np
import cv2
import logging
from typing import Dict, Any, List
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropMonitor:
    _instance = None
    
    def __init__(self, model_path: str = None, class_names: List[str] = None):
        self.model = None
        self.class_names = class_names or []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_val_transform()
        
        if model_path:
            self.load_model(model_path)
            if not self.class_names:
                self._load_class_names(os.path.dirname(model_path))

    @classmethod
    @st.cache_resource
    def get_instance(cls, model_path: str = "models/crop_classifier.pth"):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
            logger.info("Initialized new CropMonitor instance")
        return cls._instance

    def load_model(self, model_path: str) -> None:
        try:
            model = torchvision.models.efficientnet_b4(pretrained=False)
            num_classes = len(self.class_names) if self.class_names else 30
            
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model.to(self.device)
            
            self.model = model
            logger.info(f"Loaded crop classification model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load crop classification model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    def _load_class_names(self, model_dir: str) -> None:
        class_file ="agritech-ai-analyzer/classes/class_names.txt"
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.class_names)} class names from {class_file}")
        else:
            logger.warning(f"Class names file not found at {class_file}")

    def _get_val_transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _segment_green_pixels(self, image: np.ndarray) -> tuple:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return mask, segmented

    def _assess_plant_health(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        avg_color = cv2.mean(image, mask=mask)[:3]
        total = sum(avg_color) + 1e-6
        green_ratio = avg_color[1] / total
        
        if green_ratio > 0.4:
            health = "Healthy"
        elif green_ratio > 0.25:
            health = "Moderate"
        else:
            health = "Unhealthy"
            
        return {
            "health_status": health,
            "color_analysis": {
                "avg_rgb": [round(c, 2) for c in avg_color],
                "green_ratio": round(green_ratio, 4)
            }
        }

    def _create_visualization(self, original_img: np.ndarray, segmented_img: np.ndarray, 
                            crop_class: str, health_status: str) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_img)
        ax1.set_title(f"Original: {crop_class}")
        ax1.axis('off')
        
        ax2.imshow(segmented_img)
        ax2.set_title(f"Segmented - Status: {health_status}")
        ax2.axis('off')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def analyze_crop(self, image_path: str) -> Dict[str, Any]:
        """Analyze crop from image file path"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            # Perform segmentation
            mask, segmented_img = self._segment_green_pixels(img_array)
            
            # Get health assessment
            health_data = self._assess_plant_health(img_array, mask)
            
            # Get crop classification
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, preds = torch.max(outputs, 1)
                crop_class = self.class_names[preds.item()]
            
            # Create visualization
            visualization = self._create_visualization(
                img_array, 
                segmented_img,
                crop_class,
                health_data["health_status"]
            )
            
            return {
                "status": "success",
                "crop_class": crop_class,
                "health_status": health_data["health_status"],
                "color_analysis": health_data["color_analysis"],
                "visualization": f"data:image/png;base64,{visualization}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing crop: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def analyze_crop_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze crop from image bytes"""
        try:
            # Create a temporary file
            temp_path = "temp_crop_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            
            # Analyze using the file-based method
            result = self.analyze_crop(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
def get_crop_monitor():
    return CropMonitor.get_instance()
