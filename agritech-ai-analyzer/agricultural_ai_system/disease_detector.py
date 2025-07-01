import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50V2
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths relative to the project root
DEFAULT_MODEL_PATH = "models/plant_disease_model.keras"
DEFAULT_WEIGHTS_PATH = "models/plant_disease_model.weights.h5"
DEFAULT_CLASS_NAMES_PATH = "models/disease_class_names.txt"

class DiseaseDetector:
    _instance = None
    
    def __init__(self, 
                model_path: str = DEFAULT_MODEL_PATH,
                weights_path: str = DEFAULT_WEIGHTS_PATH,
                class_names_path: str = DEFAULT_CLASS_NAMES_PATH):
        """Initialize the disease detector"""
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)  # Standard size for ResNet-based models
        
        try:
            # Load class names first to get number of classes
            self.load_class_names(class_names_path)
            
            # Verify paths exist
            if not os.path.exists(model_path) and not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model file not found at {model_path} or weights at {weights_path}")
            
            # Load model with multiple fallback strategies
            self.model = self._load_model_with_fallbacks(model_path, weights_path)
            logger.info(f"Successfully loaded model from {weights_path if os.path.exists(weights_path) else model_path}")
            
            # Verify model compatibility
            self.verify_model_compatibility()
            
            # Warm up the model
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        """Singleton instance getter with Streamlit caching"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model_with_fallbacks(self, model_path: str, weights_path: str) -> tf.keras.Model:
        """Try multiple strategies to load the model"""
        # Try loading full model first
        if os.path.exists(model_path):
            try:
                return self._load_full_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load full model: {str(e)}")
        
        # Then try loading from weights with correct architecture
        if os.path.exists(weights_path):
            try:
                # Build model with correct architecture matching the weights
                base_model = ResNet50V2(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(128, activation='relu')(x)
                predictions = Dense(len(self.class_names), activation='softmax')(x)
                
                model = Model(inputs=base_model.input, outputs=predictions)
                model.load_weights(weights_path)
                return model
            except Exception as e:
                logger.warning(f"Failed to load from weights: {str(e)}")
    
        raise ValueError("All loading strategies failed")

    def _load_from_weights(self, weights_path: str) -> tf.keras.Model:
        """Load model by rebuilding architecture and loading weights"""
        logger.info("Loading model from weights")
        
        # Properly formatted Sequential model construction
        model = Sequential([
            Input(shape=(224, 224, 3)),  # Note the double closing parentheses
            ResNet50V2(weights=None, include_top=False),
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(len(self.class_names), activation='softmax')
        ])
        model.load_weights(weights_path)
        return model
        model.load_weights(weights_path)
        return model

    def _load_full_model(self, model_path: str) -> tf.keras.Model:
        """Load complete model file"""
        logger.info("Loading full model")
        custom_objects = {
            'ResNet50V2': ResNet50V2,
            'Functional': Model
        }
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

    def _rebuild_model(self, file_path: str) -> tf.keras.Model:
        """Rebuild model architecture and load weights"""
        logger.info("Rebuilding model architecture")
        base_model = ResNet50V2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(len(self.class_names), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Try loading weights (works for both .h5 and .keras files)
        model.load_weights(file_path)
        return model

    def load_class_names(self, class_names_path: str) -> None:
        """Load class names from text file"""
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            
            if not self.class_names:
                raise ValueError("Class names file is empty")
            
            logger.info(f"Loaded {len(self.class_names)} disease classes")
            
        except Exception as e:
            logger.error(f"Failed to load class names: {str(e)}")
            self.class_names = []
            raise

    def verify_model_compatibility(self):
        """Verify model output matches class names count"""
        if hasattr(self.model, 'output_shape'):
            expected_classes = self.model.output_shape[-1]
            if expected_classes != len(self.class_names):
                raise ValueError(
                    f"Model expects {expected_classes} outputs "
                    f"but found {len(self.class_names)} classes in names file"
                )

    def warm_up_model(self) -> None:
        """Run a dummy prediction to initialize the model"""
        try:
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        img = image.resize(self.input_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def create_visualization(self, 
                           original_img: Image.Image,
                           disease_type: str,
                           confidence: float) -> str:
        """Generate annotated result visualization"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(original_img)
            ax.set_title(f"Predicted: {disease_type}\nConfidence: {confidence:.1f}%", pad=10)
            ax.axis('off')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return ""

    def analyze_disease_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze disease from image bytes (for Streamlit)"""
        if not self.model:
            return {"status": "error", "error": "Model not loaded"}
        if not self.class_names:
            return {"status": "error", "error": "Class names not loaded"}
            
        try:
            # Convert bytes to image
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_array = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            disease_class = self.class_names[predicted_class]
            
            # Generate output
            visualization = self.create_visualization(image, disease_class, confidence)
            recommendations = self.get_disease_recommendations(disease_class)
            top_preds = self.get_top_predictions(predictions[0])
            
            return {
                "status": "success",
                "disease_detected": disease_class.lower() != "healthy",
                "disease_type": disease_class,
                "confidence": confidence,
                "top_predictions": top_preds,
                "recommendations": recommendations,
                "visualization": f"data:image/png;base64,{visualization}" if visualization else None
            }
            
        except Exception as e:
            logger.error(f"Disease detection failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_top_predictions(self, 
                          predictions: np.ndarray,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top k predictions with confidence scores"""
        if not self.class_names or len(predictions) != len(self.class_names):
            return []
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [{
            "disease_type": self.class_names[i],
            "confidence": float(predictions[i]) * 100
        } for i in top_indices]

    def get_disease_recommendations(self, disease_type: str) -> List[str]:
        """Generate treatment recommendations"""
        disease_lower = disease_type.lower()
        recommendations = []
        
        if disease_lower == "healthy":
            recommendations.append("Plant appears healthy. Maintain current care.")
        else:
            recommendations.append(f"Detected: {disease_type.replace('_', ' ').title()}")
            recommendations.append("Recommended Actions:")
            
            if any(x in disease_lower for x in ["powdery", "mildew"]):
                recommendations.extend([
                    "- Apply sulfur or potassium bicarbonate",
                    "- Improve air circulation",
                    "- Remove infected leaves"
                ])
            elif "blight" in disease_lower:
                recommendations.extend([
                    "- Apply copper-based fungicides",
                    "- Destroy infected plants",
                    "- Avoid overhead watering"
                ])
            elif "rust" in disease_lower:
                recommendations.extend([
                    "- Apply fungicides containing myclobutanil",
                    "- Remove affected leaves",
                    "- Avoid wetting foliage"
                ])
            elif "spot" in disease_lower:
                recommendations.extend([
                    "- Apply copper fungicide",
                    "- Improve air circulation",
                    "- Water at soil level"
                ])
            else:
                recommendations.append("- Consult with agricultural expert for specific treatment")
            
            recommendations.extend([
                "\nGeneral Prevention:",
                "- Rotate crops regularly",
                "- Use disease-resistant varieties",
                "- Sterilize tools between plants",
                "- Remove plant debris at season end"
            ])
        
        return recommendations

def get_disease_detector():
    """Public function to get the disease detector instance"""
    return DiseaseDetector.get_instance()
