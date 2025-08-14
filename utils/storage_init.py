"""
Storage initialization utilities for AI System
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any


class StorageManager:
    """Manages storage initialization and database operations"""
    
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "database.db"
        self.knowledge_path = self.storage_path / "knowledge.json"
        
    def initialize_storage(self):
        """Initialize storage directories and files"""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize knowledge file
        self._initialize_knowledge_file()
        
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                input_data TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                confidence_score REAL,
                model_used TEXT,
                response_time_ms INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                interaction_id INTEGER,
                timestamp TEXT,
                feedback_type TEXT,
                feedback_value TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                model_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _initialize_knowledge_file(self):
        """Initialize knowledge.json file"""
        if not self.knowledge_path.exists():
            initial_knowledge = {
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "knowledge_base": {},
                "training_data": []
            }
            with open(self.knowledge_path, 'w') as f:
                json.dump(initial_knowledge, f, indent=2)


def initialize_ai_storage(storage_path: str = "storage") -> bool:
    """
    Initialize AI storage system
    
    Args:
        storage_path: Path to storage directory
        
    Returns:
        True if initialization successful
    """
    try:
        storage_manager = StorageManager(storage_path)
        storage_manager.initialize_storage()
        return True
    except Exception as e:
        print(f"Error initializing storage: {e}")
        return False
