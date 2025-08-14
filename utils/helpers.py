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
import re

# Import existing functions
from .storage_init import initialize_ai_storage
from .preprocess import preprocess_input


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
    """File management utilities"""
    
    @staticmethod
    def safe_write_file(filepath: str, content: str, backup: bool = True) -> bool:
        """
        Safely write content to file with optional backup
        
        Args:
            filepath: Path to the file
            content: Content to write
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful
        """
        try:
            import os
            from pathlib import Path
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists and backup is requested
            if backup and os.path.exists(filepath):
                backup_path = f"{filepath}.bak"
                counter = 1
                while os.path.exists(backup_path):
                    backup_path = f"{filepath}.bak_{counter}"
                    counter += 1
                
                try:
                    import shutil
                    shutil.copy2(filepath, backup_path)
                except Exception as e:
                    logging.warning(f"Failed to create backup: {e}")
            
            # Write new content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to write file {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_read_file(filepath: str, default_content: str = "") -> str:
        """
        Safely read content from file
        
        Args:
            filepath: Path to the file
            default_content: Default content if file doesn't exist
            
        Returns:
            File content or default content
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            return default_content
            
        except Exception as e:
            logging.error(f"Failed to read file {filepath}: {e}")
            return default_content
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """Check if file exists"""
        import os
        return os.path.exists(filepath)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Get file size in bytes"""
        try:
            import os
            return os.path.getsize(filepath)
        except Exception:
            return 0
    
    @staticmethod
    def delete_file(filepath: str) -> bool:
        """Delete file safely"""
        try:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to delete file {filepath}: {e}")
            return False


# JSON Utilities
class JSONManager:
    """JSON file management utilities"""
    
    @staticmethod
    def save_json(file_path: str, data: Any, backup: bool = True) -> bool:
        """
        Save data to JSON file
        
        Args:
            file_path: Path to JSON file
            data: Data to save
            backup: Whether to create backup
            
        Returns:
            True if successful
        """
        try:
            json_content = safe_json_dumps(data)
            return FileManager.safe_write_file(file_path, json_content, backup=backup)
        except Exception as e:
            logging.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str, default_data: Any = None) -> Any:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            default_data: Default data if file doesn't exist
            
        Returns:
            Loaded data or default data
        """
        try:
            content = FileManager.safe_read_file(file_path, "")
            if content:
                return safe_json_loads(content) or default_data
            return default_data
        except Exception as e:
            logging.error(f"Failed to load JSON from {file_path}: {e}")
            return default_data
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validate JSON string"""
        try:
            safe_json_loads(json_str)
            return True
        except Exception:
            return False


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
    """Utility functions for the AI system"""
    
    @staticmethod
    def calculate_hash(text: str) -> str:
        """Calculate MD5 hash of text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID"""
        import time
        import random
        timestamp = int(time.time() * 1000)
        random_part = random.randint(1000, 9999)
        return f"session_{timestamp}_{random_part}"
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Don't remove mathematical operators and numbers
        # Only remove special characters that might cause issues, but keep +, -, *, /, =, ., and digits
        text = re.sub(r'[^\w\s\-.,!?;:()+\-*/=0-9]', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords
    
    @staticmethod
    def is_question(text: str) -> bool:
        """Check if text is a question"""
        if not text:
            return False
        
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose']
        text_lower = text.lower().strip()
        
        # Check if starts with question word
        for word in question_words:
            if text_lower.startswith(word):
                return True
        
        # Check if ends with question mark
        if text_lower.endswith('?'):
            return True
        
        return False
    
    @staticmethod
    def classify_query_type(text: str) -> str:
        """Classify the type of query"""
        if not text:
            return "general"
        
        text_lower = text.lower()
        
        # Math queries - check for numbers and operators
        if any(char in text for char in ['+', '-', '*', '/', '=']) and any(char.isdigit() for char in text):
            return "math"
        
        # Time-related queries
        if any(word in text_lower for word in ['time', 'date', 'when']):
            return "time"
        
        # Weather queries
        if any(word in text_lower for word in ['weather', 'temperature', 'forecast']):
            return "weather"
        
        # Definition queries
        if text_lower.startswith(('what is', 'who is', 'define', 'meaning')):
            return "definition"
        
        # How-to queries
        if text_lower.startswith('how'):
            return "howto"
        
        # General questions
        if UtilityFunctions.is_question(text):
            return "question"
        
        return "general"
    
    @staticmethod
    def format_response_time(seconds: float) -> str:
        """Format response time in a human-readable way"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 200) -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        
        # Ensure it's not empty
        if not filename:
            filename = "unnamed"
        
        return filename


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
    'initialize_ai_storage',
    'preprocess_input',
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
    'measure_performance',
    'JSONEncoder',
    'safe_json_dumps',
    'safe_json_loads',
    'calculate_hash',
    'clean_text',
    'extract_keywords',
    'is_question',
    'classify_query_type',
    'generate_session_id',
    'format_response_time',
    'truncate_text',
    'sanitize_filename'
]

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects and other special types"""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string"""
    try:
        return json.dumps(obj, cls=JSONEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        # Fallback: convert to string representation
        return json.dumps(str(obj), ensure_ascii=False)

def safe_json_loads(s: str) -> Any:
    """Safely deserialize JSON string"""
    try:
        return json.loads(s)
    except Exception as e:
        return None

def calculate_hash(text: str) -> str:
    """Calculate MD5 hash of text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Don't remove mathematical operators and numbers
    # Only remove special characters that might cause issues, but keep +, -, *, /, =, ., and digits
    text = re.sub(r'[^\w\s\-.,!?;:()+\-*/=0-9]', '', text)
    
    return text

def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text"""
    if not text:
        return []
    
    # Simple keyword extraction
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords[:10]  # Limit to top 10 keywords

def is_question(text: str) -> bool:
    """Check if text is a question"""
    if not text:
        return False
    
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose']
    text_lower = text.lower().strip()
    
    # Check if starts with question word
    for word in question_words:
        if text_lower.startswith(word):
            return True
    
    # Check if ends with question mark
    if text_lower.endswith('?'):
        return True
    
    return False

def classify_query_type(text: str) -> str:
    """Classify the type of query"""
    if not text:
        return "general"
    
    text_lower = text.lower()
    
    # Math queries - check for numbers and operators
    if any(char in text for char in ['+', '-', '*', '/', '=']) and any(char.isdigit() for char in text):
        return "math"
    
    # Time-related queries
    if any(word in text_lower for word in ['time', 'date', 'when']):
        return "time"
    
    # Weather queries
    if any(word in text_lower for word in ['weather', 'temperature', 'forecast']):
        return "weather"
    
    # Definition queries
    if text_lower.startswith(('what is', 'who is', 'define', 'meaning')):
        return "definition"
    
    # How-to queries
    if text_lower.startswith('how'):
        return "howto"
    
    # General questions
    if is_question(text):
        return "question"
    
    return "general"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import time
    import random
    timestamp = int(time.time() * 1000)
    random_part = random.randint(1000, 9999)
    return f"session_{timestamp}_{random_part}"

def format_response_time(seconds: float) -> str:
    """Format response time in a human-readable way"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed"
    
    return filename