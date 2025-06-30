import psutil
import gc
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def memory_report():
    """Generate memory usage report"""
    mem = psutil.virtual_memory()
    report = {
        'used_mb': mem.used / 1024 / 1024,
        'available_mb': mem.available / 1024 / 1024,
        'percent_used': mem.percent
    }
    
    if torch.cuda.is_available():
        report.update({
            'gpu_used_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
        })
    
    logger.info(
        f"Memory: {report['used_mb']:.1f}MB used, "
        f"{report['available_mb']:.1f}MB available"
    )
    return report

def cleanup():
    """Clean up memory resources"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_report()