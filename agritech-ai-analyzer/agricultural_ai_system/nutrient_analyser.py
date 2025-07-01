import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Any, Dict, List
import logging
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
from io import BytesIO
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutrientAnalyzer:
    _instance = None
    
    def __init__(self, model_path: str = "models/nutrient_model.keras", 
                 class_names_path: str = "agritech-ai-analyzer/classes/nutrient_class_names.npy"):
        self.model = None
        self.class_names = []
        self.image_size = (256, 256)
        
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Initialized new NutrientAnalyzer instance")
        return cls._instance

    def load_model(self, model_path: str, class_names_path: str) -> None:
        try:
            custom_objects = {
                'EfficientNetB0': keras.applications.EfficientNetB0,
                'RandomFlip': keras.layers.RandomFlip,
                'RandomRotation': keras.layers.RandomRotation,
                'Functional': keras.models.Model
            }
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            if not self.class_names:
                raise ValueError("Class names file is empty")
            logger.info(f"Loaded nutrient analysis model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load nutrient analysis model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
                
            image = cv2.resize(image, self.image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image = keras.applications.efficientnet.preprocess_input(image)
            return np.expand_dims(image, axis=0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def analyze_leaf_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "error", "error": "Nutrient analysis model not loaded"}
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            temp_path = "temp_leaf_image.jpg"
            image.save(temp_path)
            
            original_img = np.array(image)
            img_array = self.preprocess_image(original_img)
            
            predictions = self.model.predict(img_array)
            scores = tf.nn.softmax(predictions[0]).numpy()
            predicted_idx = np.argmax(scores)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            fig = self._create_visualization(original_img, scores, predicted_class, confidence, img_array)
            recommendations = self._generate_recommendations(predicted_class, scores)
            
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            os.remove(temp_path)
            
            return {
                "status": "success",
                "primary_deficiency": predicted_class,
                "confidence": confidence * 100,
                "all_deficiencies": {name: float(score) for name, score in zip(self.class_names, scores)},
                "recommendations": recommendations,
                "visualization": f"data:image/png;base64,{img_base64}" if img_base64 else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing leaf image: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _create_visualization(self, original_img: np.ndarray, scores: np.ndarray, 
                            predicted_class: str, confidence: float, 
                            img_array: np.ndarray) -> plt.Figure:
        try:
            fig = plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title(f"Input Leaf\nPredicted: {predicted_class}\nConfidence: {confidence*100:.1f}%")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            df = pd.DataFrame({
                'Deficiency': self.class_names,
                'Confidence (%)': (scores * 100).round(1)
            }).sort_values('Confidence (%)', ascending=False)
            
            sns.barplot(data=df, x='Confidence (%)', y='Deficiency', 
            hue='Deficiency', palette='viridis', legend=False)
            plt.title('Nutrient Deficiency Probabilities')
            plt.xlim(0, 100)
            
            plt.subplot(1, 3, 3)
            plt.imshow(original_img)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(original_img)
            plt.title(f"Input Leaf\nPredicted: {predicted_class}\nConfidence: {confidence*100:.1f}%")
            plt.axis('off')
            plt.tight_layout()
            return fig

    def _generate_recommendations(self, predicted_class: str, scores: np.ndarray) -> List[str]:
        recommendations = []
        
        if predicted_class.lower() == "healthy":
            recommendations.append("Leaf appears healthy. No action needed.")
        else:
            recommendations.append(f"Primary Action: Address {predicted_class} deficiency")
            
            secondary = [(n, s) for n, s in zip(self.class_names, scores) 
                        if s > 0.2 and n != predicted_class and n.lower() != "healthy"]
            if secondary:
                recommendations.append("Secondary Potential Issues:")
                for name, score in secondary:
                    recommendations.append(f"- Possible {name} deficiency ({score*100:.1f}% confidence)")
        
            deficiency_lower = predicted_class.lower()
            if "nitrogen" in deficiency_lower:
                recommendations.extend([
                    "- Apply nitrogen-rich fertilizer (e.g., urea)",
                    "- Incorporate organic matter",
                    "- Monitor soil pH"
                ])
            elif "phosphorus" in deficiency_lower:
                recommendations.extend([
                    "- Apply phosphate fertilizer",
                    "- Ensure proper soil drainage",
                    "- Test soil for nutrient levels"
                ])
            else:
                recommendations.append("- Consult with agricultural expert for specific treatment")
            
            recommendations.extend([
                "\nGeneral Prevention:",
                "- Conduct regular soil tests",
                "- Use balanced fertilizers",
                "- Rotate crops to prevent nutrient depletion"
            ])
        
        recommendations.append("Note: For severe cases, consult an agronomist for soil testing")
        return recommendations

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert figure to base64: {str(e)}")
            return ""

def get_nutrient_analyzer():
    return NutrientAnalyzer.get_instance()
