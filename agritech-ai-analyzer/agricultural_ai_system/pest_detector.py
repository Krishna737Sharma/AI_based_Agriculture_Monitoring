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

# Updated default paths to match your training script
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
        self.confidence_threshold = 0.5  # Minimum confidence for reliable detection
        
        try:
            # Check if files exist
            if not os.path.exists(model_path):
                # Try alternative model path
                alt_model_path = "models/pest_detection_final_model.keras"
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                    logger.info(f"Using alternative model path: {alt_model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or {alt_model_path}")
            
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Class names file not found at {class_names_path}")
            
            # Load model with proper error handling and custom objects
            try:
                # First try loading with compile=False
                self.model = tf.keras.models.load_model(model_path, compile=False)
                logger.info(f"Successfully loaded model from {model_path}")
            except Exception as load_error:
                logger.warning(f"Failed to load model normally: {load_error}")
                
                # Try loading with custom objects for potential compatibility issues
                try:
                    custom_objects = {
                        'KerasLayer': tf.keras.utils.get_custom_objects().get('KerasLayer', None)
                    }
                    self.model = tf.keras.models.load_model(
                        model_path, 
                        compile=False, 
                        custom_objects=custom_objects
                    )
                    logger.info(f"Successfully loaded model with custom objects from {model_path}")
                except Exception as custom_load_error:
                    logger.error(f"Failed to load model with custom objects: {custom_load_error}")
                    
                    # Try rebuilding the model architecture
                    logger.info("Attempting to rebuild model architecture...")
                    self.model = self._rebuild_model_architecture()
                    if self.model is None:
                        raise Exception("Could not rebuild model architecture")
                    
                    # Load weights
                    try:
                        self.model.load_weights(model_path.replace('.keras', '_weights.h5'))
                        logger.info("Successfully loaded weights into rebuilt model")
                    except:
                        logger.warning("Could not load separate weights file, using current model state")
            
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
            # Load class names
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            if not self.class_names:
                raise ValueError("Class names file is empty")
            logger.info(f"Loaded {len(self.class_names)} pest classes: {self.class_names}")
            
            # Verify model output matches class names
            expected_output_shape = len(self.class_names)
            actual_output_shape = self.model.output_shape[-1]
            if expected_output_shape != actual_output_shape:
                logger.warning(f"Mismatch: {expected_output_shape} classes but model outputs {actual_output_shape}")
            
            # Recompile the model to ensure proper functionality
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _rebuild_model_architecture(self):
        """Rebuild the model architecture based on the training logs"""
        try:
            # Build the same architecture as shown in your training logs
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze base model initially
            
            model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(9, activation='softmax')  # 9 classes as per your training
            ])
            
            logger.info("Successfully rebuilt model architecture")
            return model
            
        except Exception as e:
            logger.error(f"Failed to rebuild model architecture: {e}")
            return None

    @classmethod
    def get_instance(cls, model_path: str = None, class_names_path: str = None):
        """Get singleton instance with optional custom paths"""
        if cls._instance is None or (model_path is not None or class_names_path is not None):
            # Create new instance if paths are provided
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

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            img = image.resize(self.input_size, Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Note: If using Rescaling layer in model, don't normalize here
            # Otherwise, normalize to [0,1] range
            if not self._has_rescaling_layer():
                img_array = img_array / 255.0
            
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def _has_rescaling_layer(self) -> bool:
        """Check if model has rescaling layer"""
        if self.model is None:
            return False
        
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            # Check for Sequential models
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, tf.keras.layers.Rescaling):
                        return True
        return False

    def create_visualization(self, original_img: Image.Image,
                           pest_type: str, confidence: float,
                           top_predictions: List[Dict[str, Any]] = None) -> str:
        """Create visualization with prediction results"""
        try:
            # Create figure with better layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Display original image
            ax1.imshow(original_img)
            ax1.set_title(f"Input Image", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Create prediction results plot
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
                
                # Add confidence values on bars
                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    ax2.text(conf + 1, bar.get_y() + bar.get_height()/2, 
                            f'{conf:.1f}%', va='center', fontsize=10)
            else:
                ax2.text(0.5, 0.5, f"Predicted: {pest_type.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%",
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
            
            plt.tight_layout()
            
            # Convert to base64
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
            # Load and preprocess image
            image = Image.open(BytesIO(image_bytes))
            original_size = image.size
            img_array = self.preprocess_image(image)
            
            # Make prediction with error handling
            try:
                predictions = self.model.predict(img_array, verbose=0)
            except Exception as pred_error:
                logger.error(f"Prediction failed: {pred_error}")
                return {"status": "error", "error": f"Prediction failed: {str(pred_error)}"}
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            pest_class = self.class_names[predicted_class_idx]
            
            # Get top predictions
            top_preds = self.get_top_predictions(predictions[0])
            
            # Determine if pest is detected based on confidence and class
            pest_detected = self.is_pest_detected(pest_class, confidence)
            
            # Create visualization
            visualization = self.create_visualization(image, pest_class, confidence, top_preds)
            
            # Get recommendations
            recommendations = self.get_pest_recommendations(pest_class, confidence)
            
            # Calculate prediction reliability
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
        """Analyze pest from uploaded file (Streamlit UploadedFile)"""
        try:
            return self.analyze_pest_from_bytes(image_file.read())
        except Exception as e:
            logger.error(f"File analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def is_pest_detected(self, pest_class: str, confidence: float) -> bool:
        """Determine if a pest is actually detected"""
        pest_lower = pest_class.lower()
        
        # Check for no-pest classes
        no_pest_indicators = ['no_pest', 'healthy', 'normal', 'clean']
        if any(indicator in pest_lower for indicator in no_pest_indicators):
            return False
        
        # Check confidence threshold
        if confidence < self.confidence_threshold * 100:
            return False
            
        return True

    def calculate_reliability(self, predictions: np.ndarray) -> str:
        """Calculate prediction reliability based on confidence distribution"""
        max_conf = np.max(predictions)
        second_max_conf = np.partition(predictions, -2)[-2]
        
        confidence_gap = max_conf - second_max_conf
        
        if max_conf > 0.8 and confidence_gap > 0.3:
            return "High"
        elif max_conf > 0.6 and confidence_gap > 0.2:
            return "Medium"
        else:
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
        } for i in top_indices if predictions[i] > 0.01]  # Only show predictions > 1%

    def get_pest_recommendations(self, pest_type: str, confidence: float) -> List[str]:
        """Get detailed recommendations based on pest type and confidence"""
        pest_lower = pest_type.lower()
        recommendations = []
        
        # Add confidence warning if low
        if confidence < 70:
            recommendations.append("⚠️ Low confidence detection - consider getting a second opinion")
            recommendations.append("")
        
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
