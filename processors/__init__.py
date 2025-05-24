"""
Dataset processors for the Thai Audio Dataset Collection project.
"""

from typing import Dict, Any
import logging

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.processed_voice_th import ProcessedVoiceTHProcessor
from processors.mozilla_cv import MozillaCommonVoiceProcessor

logger = logging.getLogger(__name__)

def get_processor(processor_class_name: str, config: Dict[str, Any]):
    """
    Factory function to get processor instance by class name.
    
    Args:
        processor_class_name: Name of the processor class
        config: Configuration dictionary
        
    Returns:
        Processor instance
    """
    # Map processor class names to classes
    processor_map = {
        "GigaSpeech2Processor": GigaSpeech2Processor,
        "ProcessedVoiceTHProcessor": ProcessedVoiceTHProcessor,
        "MozillaCommonVoiceProcessor": MozillaCommonVoiceProcessor
    }
    
    if processor_class_name not in processor_map:
        raise ValueError(f"Unknown processor class: {processor_class_name}")
        
    return processor_map[processor_class_name](config)

__all__ = [
    'BaseProcessor',
    'GigaSpeech2Processor',
    'ProcessedVoiceTHProcessor',
    'MozillaCommonVoiceProcessor',
    'get_processor'
]