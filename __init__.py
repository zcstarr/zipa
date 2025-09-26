"""
ZIPA: Zero-shot IPA (International Phonetic Alphabet) transcription models
===========================================================================

This package provides pre-trained models for automatic phonetic transcription
using Zipformer architectures with both CTC and Transducer variants.

Main Classes:
- ZIPA_Transducer: Transducer-based phonetic transcription model
- ZIPA_CTC: CTC-based phonetic transcription model

Main Functions:
- initialize_transducer_model: Initialize a transducer model with model and BPE paths
- initialize_ctc_model: Initialize a CTC model with model and BPE paths

Subpackages:
- zipformer_transducer: Transducer implementation and training code
- zipformer_crctc: CTC implementation and training code
- scripts: Evaluation and utility scripts
- ipa_simplified: IPA tokenization and vocabulary files
"""

__version__ = "1.0.0"
__author__ = "ZIPA Team"

# Import main classes and functions
from .zipa_transducer_inference import (
    ZIPA_Transducer,
    initialize_model as initialize_transducer_model,
    small_params as transducer_small_params,
    large_params as transducer_large_params,
)

from .zipa_ctc_inference import (
    ZIPA_CTC,
    initialize_model as initialize_ctc_model,
    small_params as ctc_small_params,
    large_params as ctc_large_params,
)

# Expose subpackages
from . import zipformer_transducer
from . import zipformer_crctc
from . import scripts

# Main API convenience functions


def create_transducer_model(model_path, bpe_model_path):
    """
    Create a ZIPA Transducer model.

    Args:
        model_path (str): Path to the model weights file
        bpe_model_path (str): Path to the BPE model file

    Returns:
        ZIPA_Transducer: Initialized model ready for inference
    """
    return initialize_transducer_model(model_path, bpe_model_path)


def create_ctc_model(model_path, bpe_model_path):
    """
    Create a ZIPA CTC model.

    Args:
        model_path (str): Path to the model weights file
        bpe_model_path (str): Path to the BPE model file

    Returns:
        ZIPA_CTC: Initialized model ready for inference
    """
    return initialize_ctc_model(model_path, bpe_model_path)


# Expose all main components
__all__ = [
    # Main classes
    "ZIPA_Transducer",
    "ZIPA_CTC",

    # Initialization functions
    "initialize_transducer_model",
    "initialize_ctc_model",
    "create_transducer_model",
    "create_ctc_model",

    # Parameter configurations
    "transducer_small_params",
    "transducer_large_params",
    "ctc_small_params",
    "ctc_large_params",

    # Subpackages
    "zipformer_transducer",
    "zipformer_crctc",
    "scripts",
]
