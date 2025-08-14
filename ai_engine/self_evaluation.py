"""
Self-Evaluation Module for AI System
Handles automatic scoring, database storage, and manual feedback
"""

import sqlite3
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import logging
from pathlib import Path

@dataclass
class ResponseEvaluation:
    """Data structure for response evaluation"""
    question: str
    answer: str
    accuracy_score: float
    relevance_score: float
    speed_score: float
    overall_score: float
    timestamp: str
    model_type: str
    response_time: float
    user_feedback: Optional[str] = None
    user_rating: Optional[int] = None
    failure_reason: Optional[str] = None
    corrected_answer: Optional[str] = None
    context_info: Optional[Dict] = None

class SelfEvaluationModule:
    """
    Self-Evaluation Module for automatic scoring and feedback collection
    """
    
    def __init__(self, db_path: str = "storage/evaluation.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Scoring weights
        self.accuracy_weight = 0.4
        self.relevance_weight = 0.3
        self.speed_weight = 0.3
        
        # Performance thresholds
        self.speed_threshold = 5.0  # seconds
        self.min_acceptable_score = 0.6
        
    def _init_database(self):
        """Initialize the evaluation database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create evaluations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS response_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_hash TEXT UNIQUE,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    accuracy_score REAL NOT NULL,
                    relevance_score REAL NOT NULL,
                    speed_score REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    user_feedback TEXT,
                    user_rating INTEGER,
                    failure_reason TEXT,
                    corrected_answer TEXT,
                    context_info TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id INTEGER,
                    feedback_type TEXT NOT NULL,
                    feedback_text TEXT,
                    rating INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evaluation_id) REFERENCES response_evaluations (id)
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    avg_accuracy REAL,
                    avg_relevance REAL,
                    avg_speed REAL,
                    avg_overall REAL,
                    total_responses INTEGER,
                    successful_responses INTEGER,
                    failed_responses INTEGER,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Evaluation database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation database: {e}")
    
    def evaluate_response(self, 
                         question: str, 
                         answer: str, 
                         response_time: float,
                         model_type: str,
                         context_info: Optional[Dict] = None) -> ResponseEvaluation:
        """
        Automatically evaluate a response and store it in the database
        """
        try:
            # Calculate individual scores
            accuracy_score = self._calculate_accuracy_score(question, answer)
            relevance_score = self._calculate_relevance_score(question, answer)
            speed_score = self._calculate_speed_score(response_time)
            
            # Calculate overall score
            overall_score = (
                accuracy_score * self.accuracy_weight +
                relevance_score * self.relevance_weight +
                speed_score * self.speed_weight
            )
            
            # Create evaluation object
            evaluation = ResponseEvaluation(
                question=question,
                answer=answer,
                accuracy_score=accuracy_score,
                relevance_score=relevance_score,
                speed_score=speed_score,
                overall_score=overall_score,
                timestamp=datetime.now().isoformat(),
                model_type=model_type,
                response_time=response_time,
                context_info=context_info
            )
            
            # Store in database
            self._store_evaluation(evaluation)
            
            # Update performance metrics
            self._update_performance_metrics(model_type, evaluation)
            
            self.logger.info(f"Response evaluated - Overall Score: {overall_score:.3f}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            # Return a default evaluation with low scores
            return ResponseEvaluation(
                question=question,
                answer=answer,
                accuracy_score=0.0,
                relevance_score=0.0,
                speed_score=0.0,
                overall_score=0.0,
                timestamp=datetime.now().isoformat(),
                model_type=model_type,
                response_time=response_time,
                failure_reason=str(e)
            )
    
    def _calculate_accuracy_score(self, question: str, answer: str) -> float:
        """
        Calculate accuracy score based on answer quality indicators
        """
        score = 0.5  # Base score
        
        # Check for empty or very short answers
        if not answer or len(answer.strip()) < 10:
            return 0.1
        
        # Check for error indicators
        error_indicators = [
            "error", "failed", "unable", "cannot", "sorry", "don't know",
            "not available", "no data", "failed to", "error occurred"
        ]
        
        answer_lower = answer.lower()
        for indicator in error_indicators:
            if indicator in answer_lower:
                score -= 0.2
        
        # Check for specific answer patterns
        if any(keyword in question.lower() for keyword in ["what", "how", "why", "when", "where"]):
            if len(answer) > 50:  # Substantial answer
                score += 0.2
            if any(word in answer.lower() for word in ["because", "since", "therefore", "thus"]):
                score += 0.1  # Explanatory answer
        
        # Check for mathematical expressions (if question seems math-related)
        if any(word in question.lower() for word in ["calculate", "solve", "math", "equation", "sum", "multiply"]):
            if re.search(r'\d+[\+\-\*\/\^]\d+', answer) or "=" in answer:
                score += 0.2
        
        # Check for current data indicators
        if any(word in question.lower() for word in ["current", "latest", "today", "now", "weather", "news"]):
            if any(word in answer.lower() for word in ["2024", "2025", "today", "current", "latest"]):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """
        Calculate relevance score based on topic matching
        """
        score = 0.5  # Base score
        
        # Extract key words from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Calculate word overlap
        if question_words:
            overlap = len(question_words.intersection(answer_words))
            overlap_ratio = overlap / len(question_words)
            score += overlap_ratio * 0.3
        
        # Check for question-answer coherence
        question_type = self._classify_question_type(question)
        if question_type == "factual" and any(word in answer.lower() for word in ["is", "are", "was", "were", "will be"]):
            score += 0.1
        elif question_type == "how_to" and any(word in answer.lower() for word in ["step", "first", "then", "finally", "process"]):
            score += 0.1
        elif question_type == "opinion" and any(word in answer.lower() for word in ["think", "believe", "opinion", "view", "perspective"]):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_speed_score(self, response_time: float) -> float:
        """
        Calculate speed score based on response time
        """
        if response_time <= 1.0:
            return 1.0
        elif response_time <= self.speed_threshold:
            return 1.0 - (response_time - 1.0) / (self.speed_threshold - 1.0) * 0.5
        else:
            return max(0.1, 0.5 - (response_time - self.speed_threshold) * 0.1)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for relevance scoring"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["how to", "steps", "process", "procedure"]):
            return "how_to"
        elif any(word in question_lower for word in ["what do you think", "opinion", "believe", "feel"]):
            return "opinion"
        elif any(word in question_lower for word in ["calculate", "solve", "math", "equation"]):
            return "mathematical"
        else:
            return "factual"
    
    def _store_evaluation(self, evaluation: ResponseEvaluation):
        """Store evaluation in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create hash for question to avoid duplicates
            question_hash = hashlib.md5(evaluation.question.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO response_evaluations 
                (question_hash, question, answer, accuracy_score, relevance_score, 
                 speed_score, overall_score, timestamp, model_type, response_time,
                 user_feedback, user_rating, failure_reason, corrected_answer, context_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                question_hash, evaluation.question, evaluation.answer,
                evaluation.accuracy_score, evaluation.relevance_score,
                evaluation.speed_score, evaluation.overall_score,
                evaluation.timestamp, evaluation.model_type, evaluation.response_time,
                evaluation.user_feedback, evaluation.user_rating,
                evaluation.failure_reason, evaluation.corrected_answer,
                json.dumps(evaluation.context_info) if evaluation.context_info else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store evaluation: {e}")
    
    def _update_performance_metrics(self, model_type: str, evaluation: ResponseEvaluation):
        """Update performance metrics for the model type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current metrics
            cursor.execute('''
                SELECT * FROM performance_metrics WHERE model_type = ?
            ''', (model_type,))
            
            current = cursor.fetchone()
            
            if current:
                # Update existing metrics
                (id, _, avg_acc, avg_rel, avg_spd, avg_overall, 
                 total, successful, failed, _) = current
                
                total += 1
                if evaluation.overall_score >= self.min_acceptable_score:
                    successful += 1
                else:
                    failed += 1
                
                # Update averages
                new_avg_acc = (avg_acc * (total - 1) + evaluation.accuracy_score) / total
                new_avg_rel = (avg_rel * (total - 1) + evaluation.relevance_score) / total
                new_avg_spd = (avg_spd * (total - 1) + evaluation.speed_score) / total
                new_avg_overall = (avg_overall * (total - 1) + evaluation.overall_score) / total
                
                cursor.execute('''
                    UPDATE performance_metrics 
                    SET avg_accuracy = ?, avg_relevance = ?, avg_speed = ?, 
                        avg_overall = ?, total_responses = ?, successful_responses = ?,
                        failed_responses = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_avg_acc, new_avg_rel, new_avg_spd, new_avg_overall,
                      total, successful, failed, id))
            else:
                # Create new metrics
                successful = 1 if evaluation.overall_score >= self.min_acceptable_score else 0
                failed = 1 if evaluation.overall_score < self.min_acceptable_score else 0
                
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (model_type, avg_accuracy, avg_relevance, avg_speed, avg_overall,
                     total_responses, successful_responses, failed_responses)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_type, evaluation.accuracy_score, evaluation.relevance_score,
                      evaluation.speed_score, evaluation.overall_score, 1, successful, failed))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    def process_feedback_command(self, command: str) -> str:
        """
        Process manual feedback commands like "feedback good/bad <reason>"
        """
        try:
            # Parse feedback command
            parts = command.strip().split(' ', 2)
            if len(parts) < 2:
                return "❌ Invalid feedback format. Use: feedback good/bad <reason>"
            
            feedback_type = parts[1].lower()
            feedback_text = parts[2] if len(parts) > 2 else ""
            
            if feedback_type not in ["good", "bad"]:
                return "❌ Invalid feedback type. Use 'good' or 'bad'"
            
            # Get the most recent evaluation
            recent_eval = self._get_most_recent_evaluation()
            if not recent_eval:
                return "❌ No recent response found to provide feedback on"
            
            # Update the evaluation with user feedback
            rating = 5 if feedback_type == "good" else 1
            self._update_evaluation_with_feedback(recent_eval['id'], feedback_type, feedback_text, rating)
            
            return f"✅ Feedback recorded: {feedback_type.upper()} - {feedback_text}"
            
        except Exception as e:
            self.logger.error(f"Error processing feedback command: {e}")
            return f"❌ Error processing feedback: {e}"
    
    def _get_most_recent_evaluation(self) -> Optional[Dict]:
        """Get the most recent evaluation from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM response_evaluations 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting recent evaluation: {e}")
            return None
    
    def _update_evaluation_with_feedback(self, eval_id: int, feedback_type: str, 
                                       feedback_text: str, rating: int):
        """Update evaluation with user feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the evaluation
            cursor.execute('''
                UPDATE response_evaluations 
                SET user_feedback = ?, user_rating = ?
                WHERE id = ?
            ''', (f"{feedback_type}: {feedback_text}", rating, eval_id))
            
            # Add to feedback table
            cursor.execute('''
                INSERT INTO user_feedback 
                (evaluation_id, feedback_type, feedback_text, rating)
                VALUES (?, ?, ?, ?)
            ''', (eval_id, feedback_type, feedback_text, rating))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating evaluation with feedback: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_responses,
                    AVG(overall_score) as avg_score,
                    AVG(response_time) as avg_response_time,
                    COUNT(CASE WHEN overall_score >= ? THEN 1 END) as successful_responses
                FROM response_evaluations
            ''', (self.min_acceptable_score,))
            
            overall_stats = cursor.fetchone()
            
            # Get model-specific stats
            cursor.execute('''
                SELECT model_type, avg_overall, total_responses, successful_responses
                FROM performance_metrics
                ORDER BY avg_overall DESC
            ''')
            
            model_stats = cursor.fetchall()
            
            # Get recent evaluations
            cursor.execute('''
                SELECT question, answer, overall_score, model_type, timestamp
                FROM response_evaluations
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            
            recent_evaluations = cursor.fetchall()
            
            conn.close()
            
            return {
                "overall": {
                    "total_responses": overall_stats[0],
                    "average_score": overall_stats[1] or 0.0,
                    "average_response_time": overall_stats[2] or 0.0,
                    "success_rate": (overall_stats[3] / overall_stats[0] * 100) if overall_stats[0] > 0 else 0.0
                },
                "models": [
                    {
                        "model_type": row[0],
                        "average_score": row[1],
                        "total_responses": row[2],
                        "successful_responses": row[3]
                    }
                    for row in model_stats
                ],
                "recent_evaluations": [
                    {
                        "question": row[0][:50] + "..." if len(row[0]) > 50 else row[0],
                        "answer": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                        "score": row[2],
                        "model_type": row[3],
                        "timestamp": row[4]
                    }
                    for row in recent_evaluations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        pass  # SQLite connections are closed after each operation
