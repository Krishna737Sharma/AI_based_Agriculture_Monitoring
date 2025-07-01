import io
import os
import cv2
import joblib
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoilImageAnalyzer:
    _instance = None
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), 
                 model_dir: str = "models"):
        self.image_size = image_size
        self.model_dir = model_dir
        self.scaler = None
        self.label_encoder = None
        self.classification_model = None
        self.deep_model = None
        
        self.load_models()

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Initialized new SoilImageAnalyzer instance")
        return cls._instance

    def load_models(self) -> None:
        try:
            rf_path = os.path.join(self.model_dir, "soil_classifier_rf.pkl")
            scaler_path = os.path.join(self.model_dir, "soil_scaler.pkl")
            encoder_path ="agritech-ai-analyzer/classes/soil_label_encoder.pkl"
            cnn_path = os.path.join(self.model_dir, "soil_classifier_cnn.h5")
            
            if os.path.exists(rf_path):
                self.classification_model = joblib.load(rf_path)
                logger.info("Loaded Random Forest model")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
                
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Loaded label encoder")
                
            if os.path.exists(cnn_path):
                self.deep_model = load_model(cnn_path)
                logger.info("Loaded CNN model")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3:
            raise ValueError("Image should have 3 channels (RGB)")
            
        img = cv2.resize(image, self.image_size)
        return img.astype(np.float32) / 255.0

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        color_features = []
        for i in range(3):
            color_features.extend([
                np.mean(image[:, :, i]), 
                np.std(image[:, :, i]), 
                np.median(image[:, :, i])
            ])
            color_features.extend([
                np.mean(hsv[:, :, i]), 
                np.std(hsv[:, :, i])
            ])
            color_features.extend([
                np.mean(lab[:, :, i]), 
                np.std(lab[:, :, i])
            ])
        
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        texture_features = [
            np.mean(grad_x), np.std(grad_x),
            np.mean(grad_y), np.std(grad_y),
            np.var(gray), np.mean(gray), np.std(gray)
        ]
        
        return np.array(color_features + texture_features)

    def analyze_soil_image_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            temp_path = "temp_soil_image.jpg"
            image.save(temp_path)
            
            img = cv2.imread(temp_path)
            if img is None:
                raise ValueError(f"Could not read image from path: {temp_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            rf_pred, cnn_pred = self.predict_soil_type(img)
            
            visualization = self._create_soil_visualization(img, rf_pred or "Unknown", cnn_pred or "Unknown")
            os.remove(temp_path)
            
            return {
                "status": "success",
                "classification_prediction": rf_pred or "Unknown",
                "deep_learning_prediction": cnn_pred or "Unknown",
                "image_size": img.shape,
                "visualization": visualization
            }
            
        except Exception as e:
            logger.error(f"Error analyzing soil image: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def predict_soil_type(self, image: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        if (self.classification_model is None or 
            self.deep_model is None or 
            self.scaler is None or 
            self.label_encoder is None):
            logger.error("Required models or preprocessing objects not loaded")
            return None, None

        try:
            processed_img = self.preprocess_image(image)
            
            features = self._extract_features(processed_img)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            clf_pred = self.classification_model.predict(features_scaled)
            clf_label = self.label_encoder.inverse_transform(clf_pred)[0]
            
            dl_pred = self.deep_model.predict(processed_img.reshape(1, *self.image_size, 3))
            dl_label = self.label_encoder.inverse_transform(np.argmax(dl_pred, axis=1))[0]
            
            return clf_label, dl_label
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Failed to predict soil type: {str(e)}")

    def _create_soil_visualization(self, image: np.ndarray, rf_pred: str, cnn_pred: str) -> str:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            ax.set_title(f"RF Prediction: {rf_pred}\nCNN Prediction: {cnn_pred}", pad=10)
            ax.axis('off')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return ""

def get_soil_analyzer():
    return SoilImageAnalyzer.get_instance()
