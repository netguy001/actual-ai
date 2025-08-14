"""
Error Analysis and Self-Reflection Module
Detects failed responses, logs failures, and fetches correct answers
"""

import json
import time
import re
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import sqlite3
from urllib.parse import quote_plus
import wikipedia
from bs4 import BeautifulSoup

@dataclass
class ErrorAnalysis:
    """Data structure for error analysis"""
    original_question: str
    original_answer: str
    failure_type: str
    failure_reason: str
    corrected_answer: Optional[str] = None
    correction_source: Optional[str] = None
    confidence: float = 0.0
    timestamp: str = ""
    model_type: str = ""

class ErrorAnalysisModule:
    """
    Error Analysis Module for detecting and correcting failed responses
    """
    
    def __init__(self, db_path: str = "storage/error_analysis.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Failure detection patterns
        self.failure_indicators = {
            "no_answer": [
                "sorry", "don't know", "cannot", "unable", "error", "failed",
                "not available", "no data", "no information", "i don't have",
                "i cannot", "i'm unable", "no answer", "empty response"
            ],
            "irrelevant": [
                "music album", "song", "movie", "book", "unrelated", "off topic",
                "different topic", "not what you asked", "wrong answer"
            ],
            "incomplete": [
                "partial", "incomplete", "more information needed", "unclear",
                "vague", "not specific", "missing details"
            ],
            "outdated": [
                "old data", "outdated", "not current", "previous version",
                "historical", "no longer valid"
            ]
        }
        
        # Reliable API sources for corrections
        self.correction_sources = {
            "time": "http://worldtimeapi.org/api/ip",
            "weather": "https://wttr.in/",
            "news": "https://feeds.bbci.co.uk/news/rss.xml",
            "wikipedia": "wikipedia",
            "web_search": "https://api.duckduckgo.com/"
        }
    
    def _init_database(self):
        """Initialize the error analysis database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create error analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_question TEXT NOT NULL,
                    original_answer TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    failure_reason TEXT NOT NULL,
                    corrected_answer TEXT,
                    correction_source TEXT,
                    confidence REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create correction attempts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correction_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_id INTEGER,
                    source_used TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    result TEXT,
                    confidence REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (error_id) REFERENCES error_analysis (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Error analysis database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error analysis database: {e}")
    
    def analyze_response(self, question: str, answer: str, model_type: str) -> Optional[ErrorAnalysis]:
        """
        Analyze a response to detect failures and attempt corrections
        """
        try:
            # Detect failure type
            failure_type, failure_reason = self._detect_failure(question, answer)
            
            if not failure_type:
                return None  # No failure detected
            
            # Create error analysis object
            error_analysis = ErrorAnalysis(
                original_question=question,
                original_answer=answer,
                failure_type=failure_type,
                failure_reason=failure_reason,
                timestamp=datetime.now().isoformat(),
                model_type=model_type
            )
            
            # Attempt to get corrected answer
            corrected_answer, correction_source, confidence = self._attempt_correction(question, failure_type)
            
            if corrected_answer:
                error_analysis.corrected_answer = corrected_answer
                error_analysis.correction_source = correction_source
                error_analysis.confidence = confidence
            
            # Store in database
            self._store_error_analysis(error_analysis)
            
            self.logger.info(f"Error analysis completed - Type: {failure_type}, Correction: {bool(corrected_answer)}")
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Error in response analysis: {e}")
            return None
    
    def _detect_failure(self, question: str, answer: str) -> Tuple[Optional[str], str]:
        """
        Detect if a response has failed and determine the failure type
        """
        answer_lower = answer.lower()
        
        # Check for each failure type
        for failure_type, indicators in self.failure_indicators.items():
            for indicator in indicators:
                if indicator in answer_lower:
                    return failure_type, f"Contains failure indicator: '{indicator}'"
        
        # Check for very short or empty answers
        if len(answer.strip()) < 10:
            return "no_answer", "Answer too short or empty"
        
        # Check for question-answer mismatch
        if self._is_irrelevant_answer(question, answer):
            return "irrelevant", "Answer doesn't match question topic"
        
        # Check for current data requests that don't have current data
        if self._needs_current_data(question) and not self._has_current_data(answer):
            return "outdated", "Question requires current data but answer lacks it"
        
        return None, ""
    
    def _is_irrelevant_answer(self, question: str, answer: str) -> bool:
        """Check if answer is irrelevant to the question"""
        # Extract key words from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Calculate word overlap
        if question_words:
            overlap = len(question_words.intersection(answer_words))
            overlap_ratio = overlap / len(question_words)
            return overlap_ratio < 0.2  # Less than 20% word overlap
        
        return False
    
    def _needs_current_data(self, question: str) -> bool:
        """Check if question requires current data"""
        current_indicators = [
            "current", "latest", "today", "now", "weather", "news", 
            "time", "date", "recent", "up to date", "live"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in current_indicators)
    
    def _has_current_data(self, answer: str) -> bool:
        """Check if answer contains current data indicators"""
        current_indicators = [
            "2024", "2025", "today", "current", "latest", "now",
            "recent", "live", "real-time", "up to date"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in current_indicators)
    
    def _attempt_correction(self, question: str, failure_type: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Attempt to get a corrected answer from reliable sources
        """
        try:
            # Determine the best source based on question type
            source = self._determine_correction_source(question, failure_type)
            
            if not source:
                return None, None, 0.0
            
            # Attempt correction based on source
            if source == "time":
                return self._get_current_time()
            elif source == "weather":
                return self._get_current_weather(question)
            elif source == "news":
                return self._get_latest_news(question)
            elif source == "wikipedia":
                return self._get_wikipedia_info(question)
            elif source == "web_search":
                return self._get_web_search(question)
            
            return None, None, 0.0
            
        except Exception as e:
            self.logger.error(f"Error attempting correction: {e}")
            return None, None, 0.0
    
    def _determine_correction_source(self, question: str, failure_type: str) -> Optional[str]:
        """Determine the best source for correction based on question and failure type"""
        question_lower = question.lower()
        
        # Time-related questions
        if any(word in question_lower for word in ["time", "what time", "current time"]):
            return "time"
        
        # Weather-related questions
        if any(word in question_lower for word in ["weather", "temperature", "forecast"]):
            return "weather"
        
        # News-related questions
        if any(word in question_lower for word in ["news", "latest", "current events", "what's happening"]):
            return "news"
        
        # Factual questions (Wikipedia)
        if any(word in question_lower for word in ["what is", "who is", "definition", "explain"]):
            return "wikipedia"
        
        # General questions (web search)
        return "web_search"
    
    def _get_current_time(self) -> Tuple[str, str, float]:
        """Get current time from reliable API"""
        try:
            response = requests.get("http://worldtimeapi.org/api/ip", timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_time = data.get('datetime', '')
                timezone = data.get('timezone', '')
                
                # Format the time
                if current_time:
                    # Extract time part
                    time_part = current_time.split('T')[1][:8]  # HH:MM:SS
                    return f"Current time is {time_part} in {timezone}", "worldtimeapi", 0.9
                
        except Exception as e:
            self.logger.error(f"Error getting current time: {e}")
        
        # Fallback to local time
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"Current time is approximately {current_time}", "local_time", 0.7
    
    def _get_current_weather(self, question: str) -> Tuple[str, str, float]:
        """Get current weather information"""
        try:
            # Extract location from question or use default
            location = "London"  # Default location
            if "weather in" in question.lower():
                # Try to extract location
                parts = question.lower().split("weather in")
                if len(parts) > 1:
                    location = parts[1].strip().split()[0].title()
            
            response = requests.get(f"https://wttr.in/{location}?format=3", timeout=10)
            if response.status_code == 200:
                weather_info = response.text.strip()
                return weather_info, "wttr.in", 0.8
                
        except Exception as e:
            self.logger.error(f"Error getting weather: {e}")
        
        return "Weather information is currently unavailable", "error", 0.3
    
    def _get_latest_news(self, question: str) -> Tuple[str, str, float]:
        """Get latest news from BBC RSS feed"""
        try:
            response = requests.get("https://feeds.bbci.co.uk/news/rss.xml", timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:3]  # Get first 3 news items
                
                news_summary = "Latest news:\n"
                for i, item in enumerate(items, 1):
                    title = item.find('title').text if item.find('title') else "No title"
                    news_summary += f"{i}. {title}\n"
                
                return news_summary.strip(), "bbc_rss", 0.8
                
        except Exception as e:
            self.logger.error(f"Error getting news: {e}")
        
        return "Latest news is currently unavailable", "error", 0.3
    
    def _get_wikipedia_info(self, question: str) -> Tuple[str, str, float]:
        """Get information from Wikipedia"""
        try:
            # Extract search term from question
            search_terms = self._extract_search_terms(question)
            
            for term in search_terms:
                try:
                    # Search Wikipedia
                    search_results = wikipedia.search(term, results=1)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        summary = wikipedia.summary(search_results[0], sentences=2)
                        return summary, "wikipedia", 0.8
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error getting Wikipedia info: {e}")
        
        return "Information not found on Wikipedia", "wikipedia", 0.3
    
    def _get_web_search(self, question: str) -> Tuple[str, str, float]:
        """Get information from web search"""
        try:
            # Use DuckDuckGo Instant Answer API
            search_query = quote_plus(question)
            url = f"https://api.duckduckgo.com/?q={search_query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Abstract'):
                    return data['Abstract'], "duckduckgo", 0.7
                elif data.get('Answer'):
                    return data['Answer'], "duckduckgo", 0.7
                    
        except Exception as e:
            self.logger.error(f"Error getting web search: {e}")
        
        return "Web search information is currently unavailable", "error", 0.3
    
    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract potential search terms from a question"""
        # Remove common question words
        question_words = question.lower().split()
        stop_words = {"what", "is", "are", "who", "where", "when", "why", "how", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        search_terms = []
        for word in question_words:
            if word not in stop_words and len(word) > 2:
                search_terms.append(word)
        
        # Also try combinations
        if len(search_terms) >= 2:
            search_terms.append(" ".join(search_terms[:2]))
        
        return search_terms[:3]  # Return top 3 terms
    
    def _store_error_analysis(self, error_analysis: ErrorAnalysis):
        """Store error analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO error_analysis 
                (original_question, original_answer, failure_type, failure_reason,
                 corrected_answer, correction_source, confidence, timestamp, model_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_analysis.original_question,
                error_analysis.original_answer,
                error_analysis.failure_type,
                error_analysis.failure_reason,
                error_analysis.corrected_answer,
                error_analysis.correction_source,
                error_analysis.confidence,
                error_analysis.timestamp,
                error_analysis.model_type
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store error analysis: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error analysis statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall error stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(CASE WHEN corrected_answer IS NOT NULL THEN 1 END) as corrected_errors,
                    AVG(confidence) as avg_confidence
                FROM error_analysis
            ''')
            
            overall_stats = cursor.fetchone()
            
            # Get error types distribution
            cursor.execute('''
                SELECT failure_type, COUNT(*) as count
                FROM error_analysis
                GROUP BY failure_type
                ORDER BY count DESC
            ''')
            
            error_types = cursor.fetchall()
            
            # Get correction sources effectiveness
            cursor.execute('''
                SELECT correction_source, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM error_analysis
                WHERE corrected_answer IS NOT NULL
                GROUP BY correction_source
                ORDER BY avg_confidence DESC
            ''')
            
            correction_sources = cursor.fetchall()
            
            conn.close()
            
            return {
                "overall": {
                    "total_errors": overall_stats[0],
                    "corrected_errors": overall_stats[1],
                    "correction_rate": (overall_stats[1] / overall_stats[0] * 100) if overall_stats[0] > 0 else 0.0,
                    "average_confidence": overall_stats[2] or 0.0
                },
                "error_types": [
                    {"type": row[0], "count": row[1]}
                    for row in error_types
                ],
                "correction_sources": [
                    {"source": row[0], "count": row[1], "avg_confidence": row[2]}
                    for row in correction_sources
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting error statistics: {e}")
            return {}
    
    def get_training_data(self) -> List[Dict[str, str]]:
        """Get error-corrected data for training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT original_question, original_answer, corrected_answer, failure_type
                FROM error_analysis
                WHERE corrected_answer IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            ''')
            
            training_data = []
            for row in cursor.fetchall():
                training_data.append({
                    "question": row[0],
                    "wrong_answer": row[1],
                    "correct_answer": row[2],
                    "failure_type": row[3]
                })
            
            conn.close()
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        pass  # SQLite connections are closed after each operation
