import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/pest_detection_model.keras"
DEFAULT_CLASS_NAMES_PATH = "models/pest_class_names.npy"

class PestDetector:
    _instance = None
    
    def __init__(self, 
                model_path: str = DEFAULT_MODEL_PATH,
                class_names_path: str = DEFAULT_CLASS_NAMES_PATH):
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Class names file not found at {class_names_path}")
            
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model from {model_path}")
            
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            if not self.class_names:
                raise ValueError("Class names file is empty")
            logger.info(f"Loaded {len(self.class_names)} pest classes")
            
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warm_up_model(self) -> None:
        try:
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img = image.resize(self.input_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def create_visualization(self, original_img: Image.Image,
                           pest_type: str, confidence: float) -> str:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(original_img)
            ax.set_title(f"Predicted Pest: {pest_type}\nConfidence: {confidence:.1f}%", pad=10)
            ax.axis('off')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return ""

    def analyze_pest_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.model:
            return {"status": "error", "error": "Model not loaded"}
        if not self.class_names:
            return {"status": "error", "error": "Class names not loaded"}
            
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_array = self.preprocess_image(image)
            
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            pest_class = self.class_names[predicted_class]
            
            visualization = self.create_visualization(image, pest_class, confidence)
            recommendations = self.get_pest_recommendations(pest_class)
            top_preds = self.get_top_predictions(predictions[0])
            
            return {
                "status": "success",
                "pest_detected": pest_class.lower() != "no_pest",
                "pest_type": pest_class,
                "confidence": confidence,
                "top_predictions": top_preds,
                "recommendations": recommendations,
                "visualization": f"data:image/png;base64,{visualization}" if visualization else None
            }
            
        except Exception as e:
            logger.error(f"Pest detection failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_top_predictions(self, predictions: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.class_names or len(predictions) != len(self.class_names):
            return []
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [{
            "pest_type": self.class_names[i],
            "confidence": float(predictions[i]) * 100
        } for i in top_indices]

    def get_pest_recommendations(self, pest_type: str) -> List[str]:
        pest_lower = pest_type.lower()
        recommendations = []
        
        if pest_lower == "no_pest":
            recommendations.append("No pests detected. Maintain regular monitoring.")
        else:
            recommendations.append(f"Detected: {pest_type.replace('_', ' ').title()}")
            recommendations.append("Recommended Actions:")
            recommendations.extend([
                "- Consult with agricultural expert for specific treatment",
                "- Consider integrated pest management (IPM)",
                "- Use targeted pesticides if necessary",
                "- Remove affected plant parts",
                "\nGeneral Prevention:",
                "- Maintain crop rotation",
                "- Use pest-resistant varieties",
                "- Monitor fields regularly"
            ])
        
        return recommendations

def get_pest_detector():
    return PestDetector.get_instance()
