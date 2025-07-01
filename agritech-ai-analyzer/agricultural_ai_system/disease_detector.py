import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50V2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseDetector:
    _instance = None
    
    def __init__(self):
        try:
            self.model = self._build_model()
            logger.info("Disease detector initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _build_model(self):
        base_model = ResNet50V2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(5, activation='softmax')(x)  # Temporary: replace 5 with your actual class count
        
        return Model(inputs=base_model.input, outputs=predictions)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_disease_detector():
    return DiseaseDetector.get_instance()
