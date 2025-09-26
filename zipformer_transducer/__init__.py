"""
Zipformer Transducer Module
===========================

This module contains the transducer-based implementation of the Zipformer
architecture for phonetic transcription.

Main Components:
- model: Core model architecture
- train: Training functionality
- decode: Decoding and inference utilities
- beam_search: Beam search implementations
- joiner: Joiner network for transducer
"""

# Import key components for easy access
from .model import *
from .train import get_model
from .beam_search import greedy_search, greedy_search_batch
from .joiner import Joiner
from .decoder import Decoder

__all__ = [
    "get_model",
    "greedy_search",
    "greedy_search_batch",
    "Joiner",
    "Decoder",
]
