import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import streamlit as st
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated default paths
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
        self.min_input_size = (64, 64)
        
        try:
            # Check if files exist
            if not os.path.exists(model_path):
                alt_model_path = "models/pest_detection_final_model.keras"
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                    logger.info(f"Using alternative model path: {alt_model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or {alt_model_path}")
            
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Class names file not found at {class_names_path}")
            
            # Load with custom objects
            custom_objects = {
                'Adam': tf.keras.optimizers.Adam,
                'RandomFlip': tf.keras.layers.RandomFlip,
                'RandomRotation': tf.keras.layers.RandomRotation,
                'RandomZoom': tf.keras.layers.RandomZoom,
                'RandomContrast': tf.keras.layers.RandomContrast,
                'RandomBrightness': tf.keras.layers.RandomBrightness
            }
            
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects
            )
            
            # Verify model structure
            self._validate_model()
            
            logger.info(f"Successfully loaded model from {model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            
            # Load class names
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            if not self.class_names:
                raise ValueError("Class names file is empty")
            logger.info(f"Loaded {len(self.class_names)} pest classes")
            
            # Verify output matches class names
            if len(self.class_names) != self.model.output_shape[-1]:
                logger.warning(
                    f"Mismatch: {len(self.class_names)} classes but model outputs "
                    f"{self.model.output_shape[-1]}"
                )
            
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _validate_model(self):
        """Validate loaded model structure"""
        if not isinstance(self.model, tf.keras.Model):
            raise ValueError("Loaded object is not a Keras model")
        
        if len(self.model.inputs) != 1:
            raise ValueError("Model should have exactly one input")
            
        if self.model.input_shape[1:3] != self.input_size:
            logger.warning(
                f"Model expects input size {self.model.input_shape[1:3]} "
                f"but detector is configured for {self.input_size}"
            )
        
        if not isinstance(self.model.layers[-1], tf.keras.layers.Dense):
            raise ValueError("Final layer is not Dense - unexpected model architecture")

    @classmethod
    def get_instance(cls, model_path: str = None, class_names_path: str = None):
        """Get singleton instance with optional custom paths"""
        if cls._instance is None or (model_path is not None or class_names_path is not None):
            cls._instance = cls(
                model_path or DEFAULT_MODEL_PATH,
                class_names_path or DEFAULT_CLASS_NAMES_PATH
            )
        return cls._instance

    def warm_up_model(self) -> None:
        """Warm up the model with a dummy prediction"""
        try:
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def validate_image(self, image: Image.Image) -> bool:
        """Validate input image meets requirements"""
        if not image:
            return False
        if min(image.size) < min(self.min_input_size):
            return False
        try:
            image.verify()
            return True
        except Exception:
            return False

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid image - failed validation")
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img = image.resize(self.input_size, Image.Resampling.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Note: No division by 255 here since model has Rescaling layer
            return np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing error: {str(e)}")

    def create_visualization(self, original_img: Image.Image,
                           pest_type: str, confidence: float,
                           top_predictions: List[Dict[str, Any]] = None) -> str:
        """Create visualization with prediction results"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(original_img)
            ax1.set_title(f"Input Image", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            if top_predictions and len(top_predictions) > 1:
                classes = [pred['pest_type'].replace('_', ' ').title() for pred in top_predictions[:5]]
                confidences = [pred['confidence'] for pred in top_predictions[:5]]
                
                colors = ['green' if i == 0 else 'lightblue' for i in range(len(classes))]
                bars = ax2.barh(range(len(classes)), confidences, color=colors)
                
                ax2.set_yticks(range(len(classes)))
                ax2.set_yticklabels(classes)
                ax2.set_xlabel('Confidence (%)')
                ax2.set_title('Top Predictions', fontsize=12, fontweight='bold')
                ax2.set_xlim(0, 100)
                
                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    ax2.text(conf + 1, bar.get_y() + bar.get_height()/2, 
                            f'{conf:.1f}%', va='center', fontsize=10)
            else:
                ax2.text(0.5, 0.5, 
                        f"Predicted: {pest_type.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%",
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax2.axis('off')
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return ""

    def analyze_pest_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze pest from image bytes"""
        if not self.model:
            return {"status": "error", "error": "Model not loaded"}
        if not self.class_names:
            return {"status": "error", "error": "Class names not loaded"}
            
        try:
            image = Image.open(BytesIO(image_bytes))
            original_size = image.size
            img_array = self.preprocess_image(image)
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            pest_class = self.class_names[predicted_class_idx]
            
            top_preds = self.get_top_predictions(predictions[0])
            pest_detected = self.is_pest_detected(pest_class, confidence)
            visualization = self.create_visualization(image, pest_class, confidence, top_preds)
            recommendations = self.get_pest_recommendations(pest_class, confidence)
            reliability = self.calculate_reliability(predictions[0])
            
            return {
                "status": "success",
                "pest_detected": pest_detected,
                "pest_type": pest_class,
                "confidence": confidence,
                "reliability": reliability,
                "top_predictions": top_preds,
                "recommendations": recommendations,
                "image_info": {
                    "original_size": original_size,
                    "processed_size": self.input_size
                },
                "visualization": f"data:image/png;base64,{visualization}" if visualization else None
            }
            
        except Exception as e:
            logger.error(f"Pest detection failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def analyze_pest_from_file(self, image_file) -> Dict[str, Any]:
        """Analyze pest from uploaded file"""
        try:
            return self.analyze_pest_from_bytes(image_file.read())
        except Exception as e:
            logger.error(f"File analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def is_pest_detected(self, pest_class: str, confidence: float) -> bool:
        """Determine if a pest is actually detected"""
        pest_lower = pest_class.lower()
        no_pest_indicators = ['no_pest', 'healthy', 'normal', 'clean']
        
        if any(indicator in pest_lower for indicator in no_pest_indicators):
            return False
        if confidence < self.confidence_threshold * 100:
            return False
        return True

    def calculate_reliability(self, predictions: np.ndarray) -> str:
        """Calculate prediction reliability"""
        max_conf = np.max(predictions)
        second_max_conf = np.partition(predictions, -2)[-2]
        confidence_gap = max_conf - second_max_conf
        
        if max_conf > 0.8 and confidence_gap > 0.3:
            return "High"
        elif max_conf > 0.6 and confidence_gap > 0.2:
            return "Medium"
        return "Low"

    def get_top_predictions(self, predictions: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top k predictions with confidence scores"""
        if not self.class_names or len(predictions) != len(self.class_names):
            return []
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [{
            "pest_type": self.class_names[i],
            "confidence": float(predictions[i]) * 100,
            "class_index": int(i)
        } for i in top_indices if predictions[i] > 0.01]

    def get_pest_recommendations(self, pest_type: str, confidence: float) -> List[str]:
        """Get detailed recommendations"""
        pest_lower = pest_type.lower()
        recommendations = []
        
        if confidence < 70:
            recommendations.extend([
                "⚠️ Low confidence detection - consider getting a second opinion",
                ""
            ])
        
        if not self.is_pest_detected(pest_type, confidence):
            recommendations.extend([
                "✅ No significant pest detected",
                "",
                "🔍 Continue regular monitoring:",
                "• Check plants weekly for early signs",
                "• Look for unusual leaf patterns or damage",
                "• Monitor plant health and growth"
            ])
        else:
            pest_display = pest_type.replace('_', ' ').title()
            recommendations.extend([
                f"🐛 Detected: {pest_display}",
                f"📊 Confidence: {confidence:.1f}%",
                "",
                "🚨 Immediate Actions:",
                "• Isolate affected plants if possible",
                "• Document the extent of infestation",
                "• Take additional photos for expert consultation",
                "",
                "💡 Treatment Options:",
                "• Consult with local agricultural extension office",
                "• Consider integrated pest management (IPM)",
                "• Use targeted, eco-friendly treatments first",
                "• Apply pesticides only if necessary",
                "",
                "🛡️ Prevention Measures:",
                "• Maintain proper plant spacing",
                "• Ensure adequate ventilation",
                "• Remove plant debris regularly",
                "• Use companion planting strategies",
                "• Monitor soil health and nutrition"
            ])
        
        return recommendations

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "model_layers": len(self.model.layers),
            "trainable_params": self.model.count_params()
        }

def get_pest_detector(model_path: str = None, class_names_path: str = None):
    """Factory function to get pest detector instance"""
    return PestDetector.get_instance(model_path, class_names_path)
