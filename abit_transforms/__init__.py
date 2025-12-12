"""
ABIT Transform Analysis Framework
A modular framework for analyzing time series transforms and their combinations.
"""

__version__ = "0.1.0"
__author__ = "Preethi"

# Make key classes available at package level
from .base import Pattern, Transform, TransformCascade
from .config import TransformConfig, DEFAULT_CONFIG
from .data_generator import generate_cookie_sales, load_or_generate_data

__all__ = [
    'Pattern',
    'Transform', 
    'TransformCascade',
    'TransformConfig',
    'DEFAULT_CONFIG',
    'generate_cookie_sales',
    'load_or_generate_data'
]
