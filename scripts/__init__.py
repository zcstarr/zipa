"""
ZIPA Scripts Module
===================

This module contains evaluation and utility scripts for the ZIPA package.

Available Scripts:
- allosaurus_eval: Evaluation against Allosaurus model
- evaluate: General evaluation utilities
- filter_dataset: Dataset filtering tools
- generate_training_data: Training data generation
- whisper_eval: Evaluation against Whisper model
- xlsr_eval: Evaluation against XLSR model
"""

# Scripts are meant to be run as modules, not imported
# But we can expose them for programmatic access if needed

__all__ = [
    "allosaurus_eval",
    "evaluate",
    "filter_dataset",
    "generate_training_data",
    "whisper_eval",
    "xlsr_eval",
]
