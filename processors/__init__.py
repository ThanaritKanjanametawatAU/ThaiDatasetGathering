"""
Dataset processors for the Thai Audio Dataset Collection project.
"""

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.processed_voice_th import ProcessedVoiceTHProcessor
from processors.vistec_cv_th import VistecCommonVoiceTHProcessor
from processors.mozilla_cv import MozillaCommonVoiceProcessor

__all__ = [
    'BaseProcessor',
    'GigaSpeech2Processor',
    'ProcessedVoiceTHProcessor',
    'VistecCommonVoiceTHProcessor',
    'MozillaCommonVoiceProcessor'
]