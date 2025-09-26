"""
Zipformer CTC Module
====================

This module contains the CTC-based implementation of the Zipformer
architecture for phonetic transcription.

Main Components:
- model: Core model architecture
- train: Training functionality  
- ctc_decode: CTC decoding utilities
- expnet_ctc: ExpNet CTC implementation
"""

# Import key components for easy access
from .model import *
from .train import get_model
from .ctc_decode import *
from .expnet_ctc import *

__all__ = [
    "get_model",
]
