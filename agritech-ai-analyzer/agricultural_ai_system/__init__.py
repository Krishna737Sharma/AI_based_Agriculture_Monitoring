# agricultural_ai_system/__init__.py
from agritech-ai-analyzer.agricultural_ai_system.crop_monitor import get_crop_monitor
from agritech-ai-analyzer.agricultural_ai_system.disease_detector import get_disease_detector
from agritech-ai-analyzer.agricultural_ai_system.nutrient_analyser import get_nutrient_analyzer
from agritech-ai-analyzer.agricultural_ai_system.pest_detector import get_pest_detector
from agritech-ai-analyzer.agricultural_ai_system.soil_monitor import get_soil_analyzer
from agritech-ai-analyzer.agricultural_ai_system.weed_detector import get_weed_detector

__all__ = [
    'get_crop_monitor',
    'get_disease_detector',
    'get_nutrient_analyzer',
    'get_pest_detector',
    'get_soil_analyzer',
    'get_weed_detector'
]
