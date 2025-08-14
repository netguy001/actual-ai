"""
Core AI Model functionality for local prediction and learning.
Handles model training, prediction, and incremental learning.
"""

import os
import json
import pickle
import numpy as np
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModel:
    """
    Core AI model class that handles predictions and incremental learning.
    Supports both text and numerical data processing.
    """

    def __init__(self, model_type: str = "text_classifier"):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        self.is_trained = False
        self.version = "1.0.0"
        self.created_at = datetime.now()
        self.last_updated = self.created_at

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on model_type."""
        if self.model_type == "text_classifier":
            self.model = SGDClassifier(loss="log_loss", random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        elif self.model_type == "numerical_classifier":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_data(
        self, X: List, y: Optional[List] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess input data based on model type.

        Args:
            X: Input features
            y: Target labels (optional for prediction)

        Returns:
            Preprocessed features and labels
        """
        if self.model_type == "text_classifier":
            if not self.is_trained and self.vectorizer is not None:
                X_processed = self.vectorizer.fit_transform(X)
            elif self.vectorizer is not None:
                X_processed = self.vectorizer.transform(X)
            else:
                raise ValueError("Vectorizer not initialized")
        else:
            X_processed = np.array(X)

        y_processed = np.array(y) if y is not None else None
        return X_processed, y_processed

    def train(self, X: List, y: List, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model with provided data.

        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction of data to use for validation

        Returns:
            Training results including metrics
        """
        logger.info(f"Starting training with {len(X)} samples")

        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y)

        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, test_size=validation_split, random_state=42
            )
        else:
            X_train, y_train = X_processed, y_processed
            X_val, y_val = None, None

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.last_updated = datetime.now()

        # Calculate metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        results = {
            "train_accuracy": train_accuracy,
            "training_samples": len(X_train),
            "model_version": self.version,
            "timestamp": self.last_updated.isoformat(),
        }

        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results["val_accuracy"] = val_accuracy
            results["validation_samples"] = len(X_val)
            logger.info(
                f"Training completed - Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}"
            )
        else:
            logger.info(f"Training completed - Train Accuracy: {train_accuracy:.3f}")

        return results

    def predict(self, X: List) -> Tuple[List, List]:
        """
        Make predictions on input data.

        Args:
            X: Input features

        Returns:
            Predictions and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_processed, _ = self.preprocess_data(X)
        predictions = self.model.predict(X_processed)

        # Get confidence scores
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_processed)
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = [1.0] * len(
                predictions
            )  # Default confidence for models without probability

        return predictions.tolist(), confidences.tolist()

    def incremental_learning(self, X: List, y: List) -> Dict[str, Any]:
        """
        Perform incremental learning with new data.

        Args:
            X: New training features
            y: New training labels

        Returns:
            Update results
        """
        if not self.is_trained:
            logger.warning("Model not initially trained. Performing full training.")
            return self.train(X, y)

        logger.info(f"Performing incremental learning with {len(X)} new samples")

        X_processed, y_processed = self.preprocess_data(X, y)

        # Perform partial fit for supported models
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_processed, y_processed)
        else:
            # For models that don't support partial_fit, retrain with combined data
            logger.info(
                "Model doesn't support incremental learning. Consider using SGD-based models."
            )
            return {"status": "incremental_learning_not_supported"}

        self.last_updated = datetime.now()

        return {
            "status": "success",
            "new_samples": len(X),
            "timestamp": self.last_updated.isoformat(),
            "model_version": self.version,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "model_type": self.model_type,
            "version": self.version,
            "is_trained": self.is_trained,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "model_class": str(type(self.model).__name__),
        }


class ModelManager:
    """
    Manages model persistence, loading, and versioning.
    """

    def __init__(self, storage_path: str = "storage"):
        self.storage_path = storage_path
        self.models_dir = os.path.join(storage_path, "models")
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure necessary directories exist."""
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model: AIModel, model_name: str = "default") -> str:
        """
        Save model to disk.

        Args:
            model: AIModel instance to save
            model_name: Name for the saved model

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        model_data = {
            "model": model.model,
            "vectorizer": model.vectorizer,
            "model_type": model.model_type,
            "feature_names": model.feature_names,
            "is_trained": model.is_trained,
            "version": model.version,
            "created_at": model.created_at,
            "last_updated": model.last_updated,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "filename": model_filename,
            "path": model_path,
            "model_info": model.get_model_info(),
            "saved_at": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_name: str = "default") -> Optional[AIModel]:
        """
        Load model from disk.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded AIModel instance or None if not found
        """
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")

        if not os.path.exists(metadata_path):
            logger.warning(f"Model metadata not found: {metadata_path}")
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model_path = metadata["path"]
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct AIModel
        model = AIModel(model_data["model_type"])
        model.model = model_data["model"]
        model.vectorizer = model_data["vectorizer"]
        model.feature_names = model_data["feature_names"]
        model.is_trained = model_data["is_trained"]
        model.version = model_data["version"]
        model.created_at = model_data["created_at"]
        model.last_updated = model_data["last_updated"]

        logger.info(f"Model loaded from {model_path}")
        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models with their metadata."""
        models = []

        for filename in os.listdir(self.models_dir):
            if filename.endswith("_metadata.json"):
                metadata_path = os.path.join(self.models_dir, filename)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                models.append(metadata)

        return models

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model and its metadata.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if successful, False otherwise
        """
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")

        if not os.path.exists(metadata_path):
            logger.warning(f"Model not found: {model_name}")
            return False

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Delete model file
        model_path = metadata["path"]
        if os.path.exists(model_path):
            os.remove(model_path)

        # Delete metadata
        os.remove(metadata_path)

        logger.info(f"Model deleted: {model_name}")
        return True
