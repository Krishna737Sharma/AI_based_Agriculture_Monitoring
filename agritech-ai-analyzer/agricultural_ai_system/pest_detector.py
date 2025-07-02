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
        self.confidence_threshold = 0.3  # Minimum confidence for reliable detection
        
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
            
            # Load model with enhanced error handling
            self.model = self._load_model_robust(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Load class names
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            if not self.class_names:
                raise ValueError("Class names file is empty")
            logger.info(f"Loaded {len(self.class_names)} pest classes: {self.class_names}")
            
            # Verify model compatibility
            self._verify_model_compatibility()
            
            # Warm up model
            self.warm_up_model()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _load_model_robust(self, model_path: str):
        """Load model with multiple fallback strategies"""
        loading_strategies = [
            # Strategy 1: Normal loading
            lambda: tf.keras.models.load_model(model_path),
            
            # Strategy 2: Load without compilation
            lambda: tf.keras.models.load_model(model_path, compile=False),
            
            # Strategy 3: Load with custom objects
            lambda: tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={
                    'rescaling': tf.keras.layers.Rescaling,
                    'global_average_pooling2d': tf.keras.layers.GlobalAveragePooling2D,
                    'batch_normalization': tf.keras.layers.BatchNormalization,
                    'dropout': tf.keras.layers.Dropout,
                    'dense': tf.keras.layers.Dense
                }
            ),
            
            # Strategy 4: Load and rebuild
            lambda: self._load_and_rebuild_model(model_path)
        ]
        
        for i, strategy in enumerate(loading_strategies, 1):
            try:
                logger.info(f"Trying loading strategy {i}...")
                model = strategy()
                logger.info(f"Successfully loaded model using strategy {i}")
                return model
            except Exception as e:
                logger.warning(f"Loading strategy {i} failed: {str(e)}")
                if i == len(loading_strategies):
                    raise Exception(f"All loading strategies failed. Last error: {str(e)}")
        
        return None

    def _load_and_rebuild_model(self, model_path: str):
        """Load model and rebuild if necessary"""
        try:
            # Try to load the model
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Test if model works with a dummy input
            dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            
            return model
        except Exception as e:
            logger.warning(f"Direct loading failed: {e}")
            # If direct loading fails, try to rebuild based on your training architecture
            return self._rebuild_transfer_learning_model(len(self.class_names) if self.class_names else 9)

    def _rebuild_transfer_learning_model(self, num_classes: int):
        """Rebuild the transfer learning model based on your training script architecture"""
        logger.info("Rebuilding transfer learning model...")
        
        # Load pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model exactly as in your training script
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(*self.input_size, 3)),
            tf.keras.layers.Rescaling(1./255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model rebuilt successfully")
        return model

    def _verify_model_compatibility(self):
        """Verify model compatibility with expected inputs/outputs"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Check input shape
        expected_input_shape = (None, *self.input_size, 3)
        actual_input_shape = self.model.input_shape
        
        if actual_input_shape[1:] != expected_input_shape[1:]:
            logger.warning(f"Input shape mismatch: expected {expected_input_shape}, got {actual_input_shape}")
            # Update input size based on model
            if len(actual_input_shape) >= 3:
                self.input_size = actual_input_shape[1:3]
                logger.info(f"Updated input size to {self.input_size}")
        
        # Check output shape
        if self.class_names:
            expected_output_shape = len(self.class_names)
            actual_output_shape = self.model.output_shape[-1]
            if expected_output_shape != actual_output_shape:
                logger.warning(f"Output shape mismatch: {expected_output_shape} classes but model outputs {actual_output_shape}")

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
            
            # Convert to array
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Check if model expects normalized input (0-1) or raw input (0-255)
            # Your model has Rescaling layer, so we should pass raw values
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

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
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
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
            
            # Specific recommendations based on pest type
            pest_specific_advice = self._get_pest_specific_advice(pest_type)
            
            recommendations.extend([
                f"🐛 Detected: {pest_display}",
                f"📊 Confidence: {confidence:.1f}%",
                "",
                "🚨 Immediate Actions:",
                "• Isolate affected plants if possible",
                "• Document the extent of infestation",
                "• Take additional photos for expert consultation",
                ""
            ])
            
            # Add pest-specific advice
            if pest_specific_advice:
                recommendations.extend(pest_specific_advice)
                recommendations.append("")
            
            recommendations.extend([
                "💡 General Treatment Options:",
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

    def _get_pest_specific_advice(self, pest_type: str) -> List[str]:
        """Get specific advice for different pest types"""
        pest_advice = {
            'aphids': [
                "🐛 Aphid Management:",
                "• Use insecticidal soap or neem oil",
                "• Introduce beneficial insects (ladybugs, lacewings)",
                "• Spray with water to dislodge aphids",
                "• Remove heavily infested plant parts"
            ],
            'armyworm': [
                "🐛 Armyworm Control:",
                "• Apply biological control agents (Bt)",
                "• Use pheromone traps for monitoring",
                "• Hand-pick larvae in small infestations",
                "• Consider targeted insecticides for severe cases"
            ],
            'beetle': [
                "🐛 Beetle Management:",
                "• Use row covers during peak activity",
                "• Apply beneficial nematodes to soil",
                "• Hand-pick beetles when possible",
                "• Use targeted beetle traps"
            ],
            'bollworm': [
                "🐛 Bollworm Control:",
                "• Monitor for egg masses and larvae",
                "• Use biological control (Bt, natural enemies)",
                "• Apply targeted insecticides when necessary",
                "• Destroy crop residues after harvest"
            ],
            'grasshopper': [
                "🐛 Grasshopper Management:",
                "• Use barrier methods (row covers)",
                "• Apply biological control agents",
                "• Remove weeds and alternate hosts",
                "• Use targeted baits in severe infestations"
            ],
            'mites': [
                "🐛 Mite Control:",
                "• Increase humidity around plants",
                "• Use miticides or insecticidal soap",
                "• Introduce predatory mites",
                "• Ensure proper plant nutrition"
            ],
            'mosquito': [
                "🐛 Mosquito Prevention:",
                "• Remove standing water sources",
                "• Use biological control (Bt israelensis)",
                "• Apply appropriate larvicides",
                "• Improve drainage in growing areas"
            ],
            'sawfly': [
                "🐛 Sawfly Control:",
                "• Hand-pick larvae when visible",
                "• Use insecticidal soap spray",
                "• Apply biological control agents",
                "• Prune and destroy infested branches"
            ],
            'stem_borer': [
                "🐛 Stem Borer Management:",
                "• Cut and destroy infested stems",
                "• Use pheromone traps for monitoring",
                "• Apply systemic insecticides if necessary",
                "• Maintain field sanitation"
            ]
        }
        
        return pest_advice.get(pest_type.lower(), [])

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

    def test_model_functionality(self) -> Dict[str, Any]:
        """Test if model is working correctly"""
        try:
            # Create a test image
            test_image = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
            test_pil_image = Image.fromarray(test_image)
            
            # Test preprocessing
            processed = self.preprocess_image(test_pil_image)
            
            # Test prediction
            predictions = self.model.predict(processed, verbose=0)
            
            # Test post-processing
            top_preds = self.get_top_predictions(predictions[0])
            
            return {
                "status": "success",
                "message": "Model is working correctly",
                "processed_shape": processed.shape,
                "prediction_shape": predictions.shape,
                "top_prediction": top_preds[0] if top_preds else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model test failed: {str(e)}"
            }

def get_pest_detector(model_path: str = None, class_names_path: str = None):
    """Factory function to get pest detector instance"""
    return PestDetector.get_instance(model_path, class_names_path)

# Test function to verify the detector works
def test_pest_detector():
    """Test the pest detector with the trained model"""
    try:
        detector = get_pest_detector()
        test_result = detector.test_model_functionality()
        print(f"Test result: {test_result}")
        
        model_info = detector.get_model_info()
        print(f"Model info: {model_info}")
        
        return detector
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    # Test the detector
    test_pest_detector()
