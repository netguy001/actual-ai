"""
Data Preprocessing Utilities for AI System
Compatible with AIModel's preprocess_data() method expectations
"""

import re
import string
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities compatible with AIModel text_classifier"""

    def __init__(self):
        self.stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "would",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation but keep spaces
        text = text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )

        # Remove extra spaces
        text = " ".join(text.split())

        return text.strip()

    def remove_stop_words(self, text: str) -> str:
        """
        Remove common stop words from text

        Args:
            text: Cleaned text string

        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Preprocessed text string

        Returns:
            List of tokens
        """
        return text.split()

    def preprocess_text_data(
        self, texts: List[str], remove_stopwords: bool = True
    ) -> List[str]:
        """
        Complete text preprocessing pipeline
        Compatible with AIModel's expected input format

        Args:
            texts: List of raw text strings
            remove_stopwords: Whether to remove stop words

        Returns:
            List of preprocessed text strings
        """
        processed_texts = []

        for text in texts:
            # Clean text
            cleaned = self.clean_text(text)

            # Remove stop words if requested
            if remove_stopwords:
                cleaned = self.remove_stop_words(cleaned)

            processed_texts.append(cleaned)

        logger.debug(f"Preprocessed {len(texts)} text samples")
        return processed_texts


class NumericalPreprocessor:
    """Numerical data preprocessing utilities compatible with AIModel numerical_classifier"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []

    def handle_missing_values(
        self, data: np.ndarray, strategy: str = "mean"
    ) -> np.ndarray:
        """
        Handle missing values in numerical data

        Args:
            data: Numpy array with potential missing values
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'zero')

        Returns:
            Data with missing values handled
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Handle different data types
        if data.dtype == "object":
            # For mixed types, convert to float where possible
            data = pd.to_numeric(data, errors="coerce")

        if strategy == "mean":
            fill_value = np.nanmean(data, axis=0)
        elif strategy == "median":
            fill_value = np.nanmedian(data, axis=0)
        elif strategy == "zero":
            fill_value = 0
        else:  # default to mean
            fill_value = np.nanmean(data, axis=0)

        # Replace NaN values
        mask = np.isnan(data)
        data[mask] = fill_value

        return data

    def scale_features(
        self, data: np.ndarray, method: str = "standard", fit: bool = True
    ) -> np.ndarray:
        """
        Scale numerical features

        Args:
            data: Numerical data array
            method: Scaling method ('standard', 'minmax')
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            Scaled data array
        """
        if method not in self.scalers or fit:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            if fit:
                scaled_data = scaler.fit_transform(data)
                self.scalers[method] = scaler
            else:
                scaled_data = scaler.transform(data)
        else:
            scaled_data = self.scalers[method].transform(data)

        return scaled_data

    def encode_categorical(self, data: List[str], fit: bool = True) -> np.ndarray:
        """
        Encode categorical variables to numerical

        Args:
            data: List of categorical values
            fit: Whether to fit the encoder

        Returns:
            Encoded numerical array
        """
        if "categorical" not in self.encoders or fit:
            encoder = LabelEncoder()
            if fit:
                encoded_data = encoder.fit_transform(data)
                self.encoders["categorical"] = encoder
            else:
                encoded_data = encoder.transform(data)
        else:
            encoded_data = self.encoders["categorical"].transform(data)

        return encoded_data

    def preprocess_numerical_data(
        self,
        data: Union[List, np.ndarray, pd.DataFrame],
        fit: bool = True,
        scale_method: str = "standard",
    ) -> np.ndarray:
        """
        Complete numerical preprocessing pipeline
        Compatible with AIModel's expected input format

        Args:
            data: Numerical data in various formats
            fit: Whether to fit preprocessors
            scale_method: Scaling method to use

        Returns:
            Preprocessed numerical array
        """
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        # Handle missing values
        data = self.handle_missing_values(data)

        # Scale features
        data = self.scale_features(data, method=scale_method, fit=fit)

        logger.debug(f"Preprocessed numerical data with shape: {data.shape}")
        return data


class DataValidator:
    """Data validation utilities for input checking"""

    @staticmethod
    def validate_text_input(data: Any) -> bool:
        """
        Validate text input data

        Args:
            data: Input data to validate

        Returns:
            True if valid text data
        """
        if isinstance(data, str):
            return len(data.strip()) > 0
        elif isinstance(data, list):
            return all(isinstance(item, str) and len(item.strip()) > 0 for item in data)
        return False

    @staticmethod
    def validate_numerical_input(data: Any) -> bool:
        """
        Validate numerical input data

        Args:
            data: Input data to validate

        Returns:
            True if valid numerical data
        """
        try:
            if isinstance(data, (int, float)):
                return not np.isnan(data) and not np.isinf(data)
            elif isinstance(data, (list, np.ndarray)):
                arr = np.array(data)
                return arr.dtype in [
                    np.int32,
                    np.int64,
                    np.float32,
                    np.float64,
                ] and not np.any(np.isinf(arr))
            elif isinstance(data, pd.DataFrame):
                return data.select_dtypes(include=[np.number]).shape[1] > 0
        except (ValueError, TypeError):
            return False
        return False

    @staticmethod
    def detect_data_type(data: Any) -> str:
        """
        Detect whether input data is text or numerical

        Args:
            data: Input data

        Returns:
            'text' or 'numerical' or 'mixed' or 'unknown'
        """
        if DataValidator.validate_text_input(data):
            return "text"
        elif DataValidator.validate_numerical_input(data):
            return "numerical"
        else:
            return "unknown"


class DataFormatter:
    """Format conversion utilities for compatibility with AIModel"""

    @staticmethod
    def format_for_aimodel(data: Any, data_type: str) -> Union[List[str], np.ndarray]:
        """
        Format data for AIModel compatibility

        Args:
            data: Raw input data
            data_type: Type of data ('text' or 'numerical')

        Returns:
            Formatted data ready for AIModel
        """
        if data_type == "text":
            if isinstance(data, str):
                return [data]
            elif isinstance(data, list):
                return [str(item) for item in data]
            else:
                return [str(data)]

        elif data_type == "numerical":
            if isinstance(data, (int, float)):
                return np.array([[data]])
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], (list, tuple)):
                    return np.array(data)
                else:
                    return np.array([data])
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    return data.reshape(1, -1)
                return data
            elif isinstance(data, pd.DataFrame):
                return data.values

        raise ValueError(f"Cannot format data type: {data_type}")


# Convenience function for quick preprocessing
def preprocess_input(
    data: Any, force_type: Optional[str] = None
) -> Tuple[Union[List[str], np.ndarray], str]:
    """
    Automatically preprocess input data based on detected type

    Args:
        data: Raw input data
        force_type: Force a specific data type ('text' or 'numerical')

    Returns:
        Tuple of (processed_data, data_type)
    """
    # Detect or use forced data type
    if force_type:
        data_type = force_type
    else:
        data_type = DataValidator.detect_data_type(data)

    if data_type == "text":
        preprocessor = TextPreprocessor()
        if isinstance(data, str):
            processed = preprocessor.preprocess_text_data([data])
        else:
            processed = preprocessor.preprocess_text_data(data)

    elif data_type == "numerical":
        preprocessor = NumericalPreprocessor()
        processed = preprocessor.preprocess_numerical_data(data)

    else:
        raise ValueError(f"Cannot process data type: {data_type}")

    # Format for AIModel
    formatted_data = DataFormatter.format_for_aimodel(processed, data_type)

    return formatted_data, data_type
