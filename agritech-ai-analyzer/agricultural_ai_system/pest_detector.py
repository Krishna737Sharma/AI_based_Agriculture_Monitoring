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

DEFAULT_MODEL_PATH = "models/pest_detection_best_model.keras"
DEFAULT_CLASS_NAMES_PATH = "agritech-ai-analyzer/classes/pest_class_names.npy"

class PestDetector:
    _instance = None
    
    def __init__(self, 
                model_path: str = DEFAULT_MODEL_PATH,
                class_names_path: str = DEFAULT_CLASS_NAMES_PATH):
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        self.confidence_threshold = 0.5
        
        try:
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Class names file not found at {class_names_path}")
            
            # Load model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Model loaded from {model_path}")
            
            # Load class names
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            logger.info(f"Loaded {len(self.class_names)} pest classes")
            
            # Verify model output matches class names
            if len(self.class_names) != self.model.output_shape[-1]:
                logger.warning(f"Class count mismatch: {len(self.class_names)} classes vs model output {self.model.output_shape[-1]}")
            
            # Warm up model
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        """Singleton instance with Streamlit caching"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warm_up_model(self):
        """Run a dummy prediction to initialize model"""
        try:
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Prepare image for model prediction"""
        try:
            # Convert to RGB if needed and resize
            img = image.convert('RGB').resize(self.input_size)
            
            # Convert to array and normalize
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1] range
            return np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Could not preprocess image: {str(e)}")

    def create_visualization(self, original_img: Image.Image,
                           pest_type: str, confidence: float) -> str:
        """Create base64 encoded visualization of results"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(original_img)
            ax.set_title(f"Predicted: {pest_type.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%", pad=10)
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
        """Main analysis method that takes image bytes and returns results"""
        if not self.model:
            return {"status": "error", "error": "Model not loaded"}
        if not self.class_names:
            return {"status": "error", "error": "Class names not loaded"}
            
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_bytes))
            img_array = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            pest_class = self.class_names[predicted_class]
            
            # Generate outputs
            visualization = self.create_visualization(image, pest_class, confidence)
            recommendations = self.get_pest_recommendations(pest_class, confidence)
            top_preds = self.get_top_predictions(predictions[0])
            
            return {
                "status": "success",
                "pest_detected": self.is_pest_detected(pest_class, confidence),
                "pest_type": pest_class,
                "confidence": confidence,
                "top_predictions": top_preds,
                "recommendations": recommendations,
                "visualization": f"data:image/png;base64,{visualization}" if visualization else None
            }
            
        except Exception as e:
            logger.error(f"Pest detection failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def is_pest_detected(self, pest_class: str, confidence: float) -> bool:
        """Determine if we should consider this a pest detection"""
        return (pest_class.lower() != "no_pest" and 
                confidence >= self.confidence_threshold * 100)

    def get_top_predictions(self, predictions: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top predictions with confidence scores"""
        if not self.class_names or len(predictions) != len(self.class_names):
            return []
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [{
            "pest_type": self.class_names[i],
            "confidence": float(predictions[i]) * 100
        } for i in top_indices]

    def get_pest_recommendations(self, pest_type: str, confidence: float) -> List[str]:
        """Generate recommendations based on pest type"""
        recommendations = []
        pest_display = pest_type.replace('_', ' ').title()
        
        if not self.is_pest_detected(pest_type, confidence):
            recommendations.append("✅ No significant pest detected")
            recommendations.append("\n🔍 Continue regular monitoring:")
            recommendations.extend([
                "- Check plants weekly for early signs",
                "- Look for unusual leaf patterns or damage",
                "- Monitor plant health and growth"
            ])
        else:
            recommendations.append(f"🐛 Detected: {pest_display}")
            recommendations.append(f"📊 Confidence: {confidence:.1f}%")
            recommendations.append("\n🚨 Recommended Actions:")
            recommendations.extend([
                "- Isolate affected plants if possible",
                "- Consult with agricultural expert",
                "- Consider integrated pest management (IPM)",
                "- Use targeted, eco-friendly treatments",
                "\n🛡️ Prevention Measures:",
                "- Maintain proper plant spacing",
                "- Remove plant debris regularly",
                "- Use companion planting strategies"
            ])
        
        return recommendations

def get_pest_detector():
    """Factory function to get the detector instance"""
    return PestDetector.get_instance()
