import io
import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from typing import Dict, Any
import torch
from PIL import Image
from io import BytesIO
import base64
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeedDetector:
    _instance = None
    
    def __init__(self, model_path: str = "models/yolov8x-seg.pt"):
        try:
            self.model = YOLO(model_path)
            logger.info(f"Weed detection model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load weed detection model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    @classmethod
    @st.cache_resource
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Initialized new WeedDetector instance")
        return cls._instance

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> str:
        try:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            colored_mask = np.zeros_like(image)
            colored_mask[..., 0] = 255  # Red channel
            output_img = (image * (1 - 0.3) + colored_mask * 0.3).astype(np.uint8)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            logger.warning(f"Mask application failed: {str(e)}")
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    def detect_weed_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            temp_path = "temp_weed_image.jpg"
            image.save(temp_path)
            
            img = cv2.imread(temp_path)
            if img is None:
                return self._error_response("Invalid image file", code="invalid_image")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.model.predict(temp_path, save=False, verbose=False)
            
            weed_detected = False
            confidence = 0.0
            mask = None
            
            if results and results[0].masks is not None and len(results[0].masks) > 0:
                weed_detected = True
                mask = results[0].masks.data[0].cpu().numpy()
                
                if results[0].boxes is not None and len(results[0].boxes.conf) > 0:
                    confidence = float(results[0].boxes.conf[0].cpu().numpy()) * 100
                
                visualization = self._apply_mask(img_rgb, mask)
            else:
                visualization = self._apply_mask(img_rgb, np.zeros_like(img_rgb))

            os.remove(temp_path)
            torch.cuda.empty_cache()
            
            return {
                "status": "success",
                "weed_present": weed_detected,
                "confidence": confidence,
                "visualization": visualization
            }

        except Exception as e:
            logger.error(f"Weed detection error: {str(e)}", exc_info=True)
            return self._error_response(f"Detection failed: {str(e)}", code="detection_error")

    def _error_response(self, message: str, code: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": message,
            "code": code,
            "weed_present": False,
            "confidence": 0.0,
            "visualization": None
        }

def get_weed_detector():
    return WeedDetector.get_instance()