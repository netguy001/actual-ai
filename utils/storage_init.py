"""
AI Engine Package
Core machine learning and model management components
"""

# Import core classes from modules
from .model import AIModel, ModelManager
from .updater import OnlineUpdater

# Package metadata
__version__ = "1.0.0"
__author__ = "AI System Development"

# Export main classes
__all__ = ["AIModel", "ModelManager", "OnlineUpdater"]
