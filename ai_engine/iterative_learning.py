"""
Iterative Learning Module for AI System
Handles periodic retraining, fine-tuning, and similarity matching
"""

import json
import time
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import sqlite3
from pathlib import Path
import threading
import schedule
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

@dataclass
class TrainingData:
    """Data structure for training data"""
    questions: List[str]
    answers: List[str]
    labels: List[str]
    scores: List[float]
    timestamps: List[str]

@dataclass
class SimilarityMatch:
    """Data structure for similarity match"""
    question: str
    answer: str
    similarity_score: float
    source_type: str
    timestamp: str

class IterativeLearningModule:
    """
    Iterative Learning Module for continuous model improvement
    """
    
    def __init__(self, 
                 evaluation_db_path: str = "storage/evaluation.db",
                 error_db_path: str = "storage/error_analysis.db",
                 model_path: str = "storage/iterative_model.pkl",
                 embeddings_path: str = "storage/embeddings.pkl"):
        
        self.evaluation_db_path = evaluation_db_path
        self.error_db_path = error_db_path
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
        # Training parameters
        self.min_training_samples = 50
        self.retraining_interval_hours = 24
        self.similarity_threshold = 0.7
        self.max_similarity_results = 5
        
        # Load existing model and embeddings
        self._load_existing_model()
        
        # Start background retraining scheduler
        self._start_retraining_scheduler()
    
    def _load_existing_model(self):
        """Load existing trained model and embeddings"""
        try:
            # Load model
            if Path(self.model_path).exists():
                model_data = joblib.load(self.model_path)
                self.classifier = model_data['classifier']
                self.label_encoder = model_data['label_encoder']
                self.logger.info("Loaded existing iterative learning model")
            
            # Load embeddings
            if Path(self.embeddings_path).exists():
                with open(self.embeddings_path, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    self.vectorizer = embeddings_data['vectorizer']
                    self.question_embeddings = embeddings_data['embeddings']
                    self.question_answers = embeddings_data['answers']
                    self.question_scores = embeddings_data['scores']
                self.logger.info("Loaded existing embeddings")
            else:
                self.question_embeddings = None
                self.question_answers = []
                self.question_scores = []
                
        except Exception as e:
            self.logger.error(f"Error loading existing model: {e}")
            self.question_embeddings = None
            self.question_answers = []
            self.question_scores = []
    
    def _start_retraining_scheduler(self):
        """Start background scheduler for periodic retraining"""
        def retrain_job():
            try:
                self.logger.info("Starting scheduled retraining...")
                self.retrain_model()
            except Exception as e:
                self.logger.error(f"Error in scheduled retraining: {e}")
        
        # Schedule retraining every 24 hours
        schedule.every(self.retraining_interval_hours).hours.do(retrain_job)
        
        # Start scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        self.logger.info("Started iterative learning scheduler")
    
    def collect_training_data(self) -> TrainingData:
        """Collect training data from evaluation and error analysis databases"""
        try:
            training_data = TrainingData([], [], [], [], [])
            
            # Collect successful responses from evaluation database
            conn = sqlite3.connect(self.evaluation_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question, answer, overall_score, model_type, timestamp
                FROM response_evaluations
                WHERE overall_score >= 0.7  # Only high-quality responses
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')
            
            for row in cursor.fetchall():
                question, answer, score, model_type, timestamp = row
                training_data.questions.append(question)
                training_data.answers.append(answer)
                training_data.labels.append(model_type)
                training_data.scores.append(score)
                training_data.timestamps.append(timestamp)
            
            conn.close()
            
            # Collect corrected responses from error analysis database
            conn = sqlite3.connect(self.error_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT original_question, corrected_answer, confidence, model_type, timestamp
                FROM error_analysis
                WHERE corrected_answer IS NOT NULL AND confidence >= 0.6
                ORDER BY timestamp DESC
                LIMIT 500
            ''')
            
            for row in cursor.fetchall():
                question, answer, confidence, model_type, timestamp = row
                training_data.questions.append(question)
                training_data.answers.append(answer)
                training_data.labels.append(f"{model_type}_corrected")
                training_data.scores.append(confidence)
                training_data.timestamps.append(timestamp)
            
            conn.close()
            
            self.logger.info(f"Collected {len(training_data.questions)} training samples")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return TrainingData([], [], [], [], [])
    
    def retrain_model(self) -> bool:
        """Retrain the model with collected data"""
        try:
            # Collect training data
            training_data = self.collect_training_data()
            
            if len(training_data.questions) < self.min_training_samples:
                self.logger.info(f"Insufficient training data: {len(training_data.questions)} < {self.min_training_samples}")
                return False
            
            # Prepare features
            X = self.vectorizer.fit_transform(training_data.questions)
            y = self.label_encoder.fit_transform(training_data.labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.classifier.score(X_train, y_train)
            test_score = self.classifier.score(X_test, y_test)
            
            # Update embeddings for similarity matching
            self.question_embeddings = X
            self.question_answers = training_data.answers
            self.question_scores = training_data.scores
            
            # Save model and embeddings
            self._save_model()
            self._save_embeddings()
            
            self.logger.info(f"Model retrained successfully - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
            return False
    
    def _save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'vectorizer': self.vectorizer,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def _save_embeddings(self):
        """Save embeddings for similarity matching"""
        try:
            embeddings_data = {
                'vectorizer': self.vectorizer,
                'embeddings': self.question_embeddings,
                'answers': self.question_answers,
                'scores': self.question_scores,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings_data, f)
            self.logger.info("Embeddings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
    
    def find_similar_answers(self, question: str) -> List[SimilarityMatch]:
        """Find similar past questions and answers using embeddings"""
        try:
            if self.question_embeddings is None or len(self.question_answers) == 0:
                return []
            
            # Vectorize the input question
            question_vector = self.vectorizer.transform([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.question_embeddings).flatten()
            
            # Find top similar questions
            similar_indices = np.argsort(similarities)[::-1][:self.max_similarity_results]
            
            similar_matches = []
            for idx in similar_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= self.similarity_threshold:
                    similar_matches.append(SimilarityMatch(
                        question=question,  # Original question
                        answer=self.question_answers[idx],
                        similarity_score=similarity_score,
                        source_type="similarity_match",
                        timestamp=datetime.now().isoformat()
                    ))
            
            self.logger.info(f"Found {len(similar_matches)} similar answers")
            return similar_matches
            
        except Exception as e:
            self.logger.error(f"Error finding similar answers: {e}")
            return []
    
    def predict_best_model(self, question: str) -> Tuple[str, float]:
        """Predict the best model to use for a given question"""
        try:
            if not hasattr(self.classifier, 'classes_'):
                return "enhanced_ai", 0.5  # Default fallback
            
            # Vectorize the question
            question_vector = self.vectorizer.transform([question])
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(question_vector)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Decode the predicted class
            predicted_model = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            self.logger.info(f"Predicted model: {predicted_model} (confidence: {confidence:.3f})")
            return predicted_model, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting best model: {e}")
            return "enhanced_ai", 0.5  # Default fallback
    
    def learn_from_interaction(self, question: str, answer: str, model_type: str, 
                             score: float, success: bool):
        """Learn from a new interaction"""
        try:
            # Add to training data for next retraining
            # This could be stored in a temporary buffer or database
            self._store_interaction(question, answer, model_type, score, success)
            
            # Update embeddings if we have enough data
            if len(self.question_answers) < 1000:  # Limit memory usage
                self._update_embeddings(question, answer, score)
            
            self.logger.info(f"Learned from interaction - Model: {model_type}, Score: {score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error learning from interaction: {e}")
    
    def _store_interaction(self, question: str, answer: str, model_type: str, 
                          score: float, success: bool):
        """Store interaction for future training"""
        try:
            conn = sqlite3.connect(self.evaluation_db_path)
            cursor = conn.cursor()
            
            # Add to a temporary learning table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO learning_buffer (question, answer, model_type, score, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (question, answer, model_type, score, success))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing interaction: {e}")
    
    def _update_embeddings(self, question: str, answer: str, score: float):
        """Update embeddings with new data"""
        try:
            # Add to existing data
            self.question_answers.append(answer)
            self.question_scores.append(score)
            
            # Recompute embeddings if we have enough data
            if len(self.question_answers) % 100 == 0:  # Update every 100 interactions
                self._recompute_embeddings()
                
        except Exception as e:
            self.logger.error(f"Error updating embeddings: {e}")
    
    def _recompute_embeddings(self):
        """Recompute embeddings with all collected data"""
        try:
            # Get all questions from database
            conn = sqlite3.connect(self.evaluation_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question FROM response_evaluations
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')
            
            questions = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if len(questions) > 0:
                # Recompute embeddings
                self.question_embeddings = self.vectorizer.fit_transform(questions)
                self._save_embeddings()
                
        except Exception as e:
            self.logger.error(f"Error recomputing embeddings: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning module statistics"""
        try:
            stats = {
                "model_info": {
                    "model_path": self.model_path,
                    "embeddings_path": self.embeddings_path,
                    "has_trained_model": hasattr(self.classifier, 'classes_'),
                    "num_classes": len(self.classifier.classes_) if hasattr(self.classifier, 'classes_') else 0
                },
                "embeddings_info": {
                    "num_embeddings": len(self.question_answers) if self.question_answers else 0,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_similarity_results
                },
                "training_info": {
                    "min_samples": self.min_training_samples,
                    "retraining_interval": f"{self.retraining_interval_hours} hours",
                    "last_retraining": self._get_last_retraining_time()
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return {}
    
    def _get_last_retraining_time(self) -> str:
        """Get the last retraining time"""
        try:
            if Path(self.model_path).exists():
                model_data = joblib.load(self.model_path)
                return model_data.get('timestamp', 'Unknown')
            return 'Never'
        except Exception as e:
            self.logger.error(f"Error getting last retraining time: {e}")
            return 'Unknown'
    
    def force_retraining(self) -> bool:
        """Force immediate retraining"""
        self.logger.info("Forcing immediate retraining...")
        return self.retrain_model()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Save current state
            if hasattr(self.classifier, 'classes_'):
                self._save_model()
            if self.question_embeddings is not None:
                self._save_embeddings()
            
            self.logger.info("Iterative learning module cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
