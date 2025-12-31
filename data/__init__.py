"""
Data module for the Agentic Text2Cypher pipeline.

Provides data loading utilities for ground truth and results.
"""

from .ground_truth_loader import (
    GroundTruthItem,
    GroundTruthLoader,
    load_ground_truth,
)

__all__ = [
    "GroundTruthItem",
    "GroundTruthLoader",
    "load_ground_truth",
]
