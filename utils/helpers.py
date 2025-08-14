"""
Utility Functions and Helpers for AI System
Compatible with existing storage system and database schema
"""

import os
import json
import sqlite3
import logging
import functools
import time
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
import traceback


# Logging Configuration
class LogManager:
    """Centralized logging configuration and management"""
    
    @staticmethod
    def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
        """
        Configure logging for the AI system
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_file: Optional log file path
            
        Returns:
            Configured logger instance
        """
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        return root_logger


# File I/O Utilities
class FileManager:
    """File I/O utilities for the AI system"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't
        
        Args:
            path: Directory path
            
        Returns:
            Path object of created/existing directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def safe_file_write(file_path: Union[str, Path], content: Union[str, bytes], 
                       mode: str = 'w', backup: bool = True) -> bool:
        """
        Safely write content to file with optional backup
        
        Args:
            file_path: Target file path
            content: Content to write
            mode: File write mode
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Create backup if file exists
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f'.bak_{int(time.time())}')
                file_path.rename(backup_path)
                logging.debug(f"Created backup: {backup_path}")
            
            # Ensure parent directory exists
            FileManager.ensure_directory(file_path.parent)
            
            # Write content
            with open(file_path, mode) as f:
                f.write(content)
            
            logging.debug(f"Successfully wrote to: {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to write file {file_path}: {e}")
            return False
    
    @staticmethod
    def safe_file_read(file_path: Union[str, Path], mode: str = 'r', 
                      default: Any = None) -> Union[str, bytes, Any]:
        """
        Safely read content from file
        
        Args:
            file_path: Source file path
            mode: File read mode
            default: Default value if file doesn't exist
            
        Returns:
            File content or default value
        """
        try:
            with open(file_path, mode) as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"File not found: {file_path}")
            return default
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return default


# JSON Utilities
class JSONManager:
    """JSON serialization and knowledge.json manipulation utilities"""
    
    @staticmethod
    def load_json(file_path: Union[str, Path], default: Dict = None) -> Dict:
        """
        Load JSON data from file
        
        Args:
            file_path: JSON file path
            default: Default dict if file doesn't exist
            
        Returns:
            Parsed JSON data or default dict
        """
        if default is None:
            default = {}
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"JSON file not found: {file_path}")
            return default
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {file_path}: {e}")
            return default
        except Exception as e:
            logging.error(f"Failed to load JSON {file_path}: {e}")
            return default
    
    @staticmethod
    def save_json(data: Dict, file_path: Union[str, Path], 
                  indent: int = 2, backup: bool = True) -> bool:
        """
        Save data to JSON file
        
        Args:
            data: Dictionary to save
            file_path: Target JSON file path
            indent: JSON indentation
            backup: Whether to create backup
            
        Returns:
            True if successful
        """
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            return FileManager.safe_file_write(file_path, json_content, backup=backup)
        except Exception as e:
            logging.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def update_knowledge_json(storage_path: str, updates: Dict) -> bool:
        """
        Update knowledge.json with new data
        Compatible with existing storage structure
        
        Args:
            storage_path: Storage directory path
            updates: Dictionary of updates to merge
            
        Returns:
            True if successful
        """
        knowledge_path = Path(storage_path) / 'knowledge.json'
        
        # Load existing knowledge
        knowledge = JSONManager.load_json(knowledge_path, {
            "user_preferences": {},
            "learned_patterns": [],
            "interaction_history": [],
            "custom_models": {},
            "last_updated": None
        })
        
        # Merge updates
        for key, value in updates.items():
            if key in knowledge:
                if isinstance(knowledge[key], dict) and isinstance(value, dict):
                    knowledge[key].update(value)
                elif isinstance(knowledge[key], list) and isinstance(value, list):
                    knowledge[key].extend(value)
                else:
                    knowledge[key] = value
            else:
                knowledge[key] = value
        
        # Update timestamp
        knowledge["last_updated"] = datetime.now().isoformat()
        
        return JSONManager.save_json(knowledge, knowledge_path)


# Database Utilities
class DatabaseManager:
    """Database connection and query utilities compatible with existing schema"""
    
    @staticmethod
    @contextmanager
    def get_connection(db_path: Union[str, Path]):
        """
        Context manager for database connections
        
        Args:
            db_path: Database file path
            
        Yields:
            SQLite connection object
        """
        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def execute_query(db_path: Union[str, Path], query: str, 
                     params: Tuple = (), fetch: bool = False) -> Union[List[Dict], bool]:
        """
        Execute SQL query with error handling
        
        Args:
            db_path: Database file path
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, otherwise success boolean
        """
        try:
            with DatabaseManager.get_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if fetch:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                else:
                    conn.commit()
                    return True
                    
        except Exception as e:
            logging.error(f"Query execution failed: {e}\nQuery: {query}")
            return [] if fetch else False
    
    @staticmethod
    def log_interaction(db_path: Union[str, Path], session_id: str, 
                       user_input: str, ai_response: str, confidence: float,
                       model_used: str, response_time: float) -> bool:
        """
        Log user interaction to database
        Compatible with existing user_interactions table schema
        
        Args:
            db_path: Database file path
            session_id: Unique session identifier
            user_input: User's input text
            ai_response: AI system's response
            confidence: Prediction confidence score
            model_used: Name of model used
            response_time: Response time in seconds
            
        Returns:
            True if successful
        """
        query = """
        INSERT INTO user_interactions 
        (session_id, timestamp, user_input, ai_response, confidence_score, model_used, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            session_id,
            datetime.now().isoformat(),
            user_input,
            ai_response,
            confidence,
            model_used,
            int(response_time * 1000)  # Convert to milliseconds
        )
        
        return DatabaseManager.execute_query(db_path, query, params)
    
    @staticmethod
    def log_feedback(db_path: Union[str, Path], session_id: str, 
                    interaction_id: int, feedback_type: str, 
                    feedback_value: Union[str, int], notes: str = "") -> bool:
        """
        Log user feedback to database
        Compatible with existing user_feedback table schema
        
        Args:
            db_path: Database file path
            session_id: Session identifier
            interaction_id: Related interaction ID
            feedback_type: Type of feedback ('rating', 'correction', etc.)
            feedback_value: Feedback value
            notes: Additional notes
            
        Returns:
            True if successful
        """
        query = """
        INSERT INTO user_feedback 
        (session_id, interaction_id, timestamp, feedback_type, feedback_value, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            session_id,
            interaction_id,
            datetime.now().isoformat(),
            feedback_type,
            str(feedback_value),
            notes
        )
        
        return DatabaseManager.execute_query(db_path, query, params)


# Configuration Management
class ConfigManager:
    """Configuration management utilities"""
    
    DEFAULT_CONFIG = {
        "storage_path": "storage",
        "log_level": "INFO",
        "log_file": "storage/logs/ai_system.log",
        "default_model_type": "text_classifier",
        "auto_update_check": True,
        "update_interval_hours": 24,
        "max_interaction_history": 1000,
        "backup_enabled": True,
        "preprocessing": {
            "remove_stopwords": True,
            "scaling_method": "standard",
            "handle_missing": "mean"
        },
        "model_settings": {
            "confidence_threshold": 0.5,
            "incremental_learning": True,
            "auto_save": True
        }
    }
    
    @staticmethod
    def load_config(config_path: str = "config.json") -> Dict:
        """
        Load configuration from file
        
        Args:
            config_path: Configuration file path
            
        Returns:
            Configuration dictionary
        """
        config = ConfigManager.DEFAULT_CONFIG.copy()
        
        # Try to load from file
        file_config = JSONManager.load_json(config_path, {})
        
        # Deep merge configurations
        ConfigManager._deep_merge(config, file_config)
        
        return config
    
    @staticmethod
    def save_config(config: Dict, config_path: str = "config.json") -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Configuration file path
            
        Returns:
            True if successful
        """
        return JSONManager.save_json(config, config_path)
    
    @staticmethod
    def _deep_merge(target: Dict, source: Dict) -> None:
        """
        Deep merge two dictionaries
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(target[key], value)
            else:
                target[key] = value


# Data Serialization Utilities
class SerializationManager:
    """Data serialization and deserialization utilities"""
    
    @staticmethod
    def serialize_object(obj: Any, file_path: Union[str, Path]) -> bool:
        """
        Serialize object to file using pickle
        
        Args:
            obj: Object to serialize
            file_path: Target file path
            
        Returns:
            True if successful
        """
        try:
            FileManager.ensure_directory(Path(file_path).parent)
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            logging.debug(f"Serialized object to: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Serialization failed for {file_path}: {e}")
            return False
    
    @staticmethod
    def deserialize_object(file_path: Union[str, Path], default: Any = None) -> Any:
        """
        Deserialize object from file using pickle
        
        Args:
            file_path: Source file path
            default: Default value if deserialization fails
            
        Returns:
            Deserialized object or default value
        """
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            logging.debug(f"Deserialized object from: {file_path}")
            return obj
        except FileNotFoundError:
            logging.warning(f"Serialized file not found: {file_path}")
            return default
        except Exception as e:
            logging.error(f"Deserialization failed for {file_path}: {e}")
            return default


# Error Handling Decorators
def handle_errors(default_return=None, log_errors=True):
    """
    Decorator for error handling and logging
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {e}")
                    logging.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def measure_performance(log_result=True):
    """
    Decorator to measure function performance
    
    Args:
        log_result: Whether to log performance metrics
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            if log_result:
                logging.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            # Attach timing info to result if it's a dict
            if isinstance(result, dict):
                result['_execution_time'] = execution_time
            
            return result
        return wrapper
    return decorator


# Utility Functions
class UtilityFunctions:
    """Miscellaneous utility functions"""
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate unique session ID
        
        Returns:
            Unique session identifier
        """
        timestamp = str(int(time.time() * 1000))
        random_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_str}"
    
    @staticmethod
    def calculate_hash(data: Union[str, bytes, Dict]) -> str:
        """
        Calculate MD5 hash of data
        
        Args:
            data: Data to hash
            
        Returns:
            MD5 hash string
        """
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human readable format
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted size string (e.g., "1.2 MB")
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean filename by removing invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        import string
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        cleaned = ''.join(c for c in filename if c in valid_chars)
        return cleaned.strip()
    
    @staticmethod
    def get_system_info() -> Dict:
        """
        Get basic system information
        
        Returns:
            Dictionary with system info
        """
        import platform
        import psutil
        
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
            }


# Performance Monitoring
class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.metrics = {}
        self.start_time = time.time()
    
    def log_metric(self, name: str, value: Union[int, float], timestamp: Optional[float] = None) -> None:
        """
        Log performance metric
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'timestamp': timestamp,
            'value': value
        })
    
    def get_metric_summary(self, name: str) -> Dict:
        """
        Get summary statistics for a metric
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with summary statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1]
        }
    
    def save_metrics(self) -> bool:
        """
        Save metrics to file
        
        Returns:
            True if successful
        """
        metrics_file = Path(self.storage_path) / 'performance_metrics.json'
        
        # Add summary data
        summary_data = {
            'session_start': self.start_time,
            'session_duration': time.time() - self.start_time,
            'metrics': self.metrics,
            'summaries': {name: self.get_metric_summary(name) for name in self.metrics}
        }
        
        return JSONManager.save_json(summary_data, metrics_file)


# Cache Management
class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        if key not in self.cache:
            return default
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() > entry['expires']:
            del self.cache[key]
            return default
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cached value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl,
            'created': time.time()
        }
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if current_time > entry['expires']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)


# Export commonly used functions for easy imports
__all__ = [
    'LogManager',
    'FileManager', 
    'JSONManager',
    'DatabaseManager',
    'ConfigManager',
    'SerializationManager',
    'UtilityFunctions',
    'PerformanceMonitor',
    'CacheManager',
    'handle_errors',
    'measure_performance'
]