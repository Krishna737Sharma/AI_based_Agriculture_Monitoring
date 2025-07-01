# agricultural_ai_system/__init__.py
from .crop_monitor import get_crop_monitor
from .disease_detector import get_disease_detector
from .nutrient_analyser import get_nutrient_analyzer
from .pest_detector import get_pest_detector
from .soil_monitor import get_soil_analyzer
from .weed_detector import get_weed_detector

__all__ = [
    'get_crop_monitor',
    'get_disease_detector',
    'get_nutrient_analyzer',
    'get_pest_detector',
    'get_soil_analyzer',
    'get_weed_detector'
]
