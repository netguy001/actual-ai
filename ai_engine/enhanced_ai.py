"""
Enhanced AI Engine with Internet Access, Math, and Learning Capabilities
"""

import os
import re
import json
import math
import requests
import wikipedia
import sympy
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote_plus
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import time
import requests
import wikipedia
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from bs4 import BeautifulSoup
import sqlite3
from utils.helpers import safe_json_dumps, safe_json_loads, clean_text, classify_query_type, is_question

logger = logging.getLogger(__name__)


class EnhancedAI:
    """
    Enhanced AI with web scraping, Wikipedia, math, real-time data fetching,
    query classification, continuous learning, data persistence, auto-correct,
    and context awareness
    """
    
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize Wikipedia
        wikipedia.set_lang('en')
        
        # Math symbols and functions
        self.math_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
            'exp': math.exp, 'abs': abs, 'round': round,
            'floor': math.floor, 'ceil': math.ceil, 'pi': math.pi,
            'e': math.e
        }
        
        # Enhanced Learning and Memory System
        self.conversation_history = []
        self.knowledge_base = {}
        self.user_preferences = {}
        self.current_conversation_context = {
            'topic': None,
            'last_questions': [],
            'related_info': {},
            'conversation_chain': []
        }
        self.typo_corrections = {}
        self.data_feeds = {
            'last_news_update': None,
            'last_weather_update': None,
            'last_knowledge_update': None
        }
        
        # Load existing data
        self._load_persistent_data()
        
    def _load_persistent_data(self):
        """Load all persistent data from storage"""
        try:
            # Ensure storage directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load conversation history
            history_file = self.storage_path / "conversation_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"Loaded {len(self.conversation_history)} conversation records")
            
            # Load knowledge base
            knowledge_file = self.storage_path / "ai_knowledge_base.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            
            # Load user preferences
            prefs_file = self.storage_path / "user_preferences.json"
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    self.user_preferences = json.load(f)
                logger.info("Loaded user preferences")
            
            # Load conversation context
            context_file = self.storage_path / "conversation_context.json"
            if context_file.exists():
                with open(context_file, 'r', encoding='utf-8') as f:
                    self.current_conversation_context = json.load(f)
                logger.info("Loaded conversation context")
            
            # Load typo corrections
            typo_file = self.storage_path / "typo_corrections.json"
            if typo_file.exists():
                with open(typo_file, 'r', encoding='utf-8') as f:
                    self.typo_corrections = json.load(f)
                logger.info(f"Loaded {len(self.typo_corrections)} typo corrections")
            
            # Load data feeds
            feeds_file = self.storage_path / "data_feeds.json"
            if feeds_file.exists():
                with open(feeds_file, 'r', encoding='utf-8') as f:
                    self.data_feeds = json.load(f)
                logger.info("Loaded data feeds")
                
        except Exception as e:
            logger.error(f"Error loading persistent data: {e}")
    
    def _save_persistent_data(self):
        """Save all persistent data to storage"""
        try:
            # Ensure storage directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save conversation history
            history_file = self.storage_path / "conversation_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            
            # Save knowledge base
            knowledge_file = self.storage_path / "ai_knowledge_base.json"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
            # Save user preferences
            prefs_file = self.storage_path / "user_preferences.json"
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, indent=2, ensure_ascii=False)
            
            # Save conversation context
            context_file = self.storage_path / "conversation_context.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_conversation_context, f, indent=2, ensure_ascii=False)
            
            # Save typo corrections
            typo_file = self.storage_path / "typo_corrections.json"
            with open(typo_file, 'w', encoding='utf-8') as f:
                json.dump(self.typo_corrections, f, indent=2, ensure_ascii=False)
            
            # Save data feeds
            feeds_file = self.storage_path / "data_feeds.json"
            with open(feeds_file, 'w', encoding='utf-8') as f:
                json.dump(self.data_feeds, f, indent=2, ensure_ascii=False)
                
            logger.info("Enhanced persistent data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")
    
    def learn_from_interaction(self, query: str, response: str):
        """Learn from user interaction"""
        try:
            # Only store substantial responses, not greetings
            if len(response) > 20 and not any(greeting in query.lower() for greeting in ['hello', 'hi', 'hey']):
                self.knowledge_base[f"learned_{len(self.knowledge_base)}"] = {
                    'query': query,
                    'response': response,
                    'type': 'learned',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Keep knowledge base size manageable
                if len(self.knowledge_base) > 50:
                    # Remove oldest entries
                    keys_to_remove = list(self.knowledge_base.keys())[:10]
                    for key in keys_to_remove:
                        if key.startswith('learned_'):
                            del self.knowledge_base[key]
                
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    def _update_conversation_patterns(self, user_input: str, ai_response: str):
        """Update conversation patterns for better context understanding"""
        try:
            # Store conversation patterns
            pattern = {
                'user_input_pattern': self._extract_pattern(user_input),
                'ai_response_pattern': self._extract_pattern(ai_response),
                'timestamp': datetime.now().isoformat()
            }
            
            if 'conversation_patterns' not in self.user_preferences:
                self.user_preferences['conversation_patterns'] = []
            
            self.user_preferences['conversation_patterns'].append(pattern)
            
            # Keep only last 50 patterns
            if len(self.user_preferences['conversation_patterns']) > 50:
                self.user_preferences['conversation_patterns'] = \
                    self.user_preferences['conversation_patterns'][-50:]
                    
        except Exception as e:
            logger.error(f"Error updating conversation patterns: {e}")
    
    def _extract_pattern(self, text: str) -> str:
        """Extract conversation pattern from text"""
        try:
            # Extract key words and structure
            words = text.lower().split()
            key_words = [word for word in words if len(word) > 3]
            return ' '.join(key_words[:5])  # First 5 key words
        except Exception:
            return text.lower()
    
    def _classify_input_type(self, user_input: str) -> str:
        """Classify the type of user input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['hello', 'hi', 'hey']):
            return 'greeting'
        elif any(word in input_lower for word in ['what is', 'who is', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(word in input_lower for word in ['calculate', 'solve', 'math', '+', '-', '*', '/']):
            return 'calculation'
        elif any(word in input_lower for word in ['news', 'current', 'latest']):
            return 'news_request'
        elif any(word in input_lower for word in ['weather', 'temperature']):
            return 'weather_request'
        elif any(word in input_lower for word in ['time', 'date']):
            return 'time_request'
        else:
            return 'general'
    
    def _classify_response_type(self, ai_response: str) -> str:
        """Classify the type of AI response"""
        response_lower = ai_response.lower()
        
        if 'wikipedia' in response_lower:
            return 'factual'
        elif 'news' in response_lower:
            return 'news'
        elif 'weather' in response_lower:
            return 'weather'
        elif 'time' in response_lower or 'date' in response_lower:
            return 'time'
        elif 'result' in response_lower and ('=' in response_lower or 'calculation' in response_lower):
            return 'calculation'
        else:
            return 'conversational'
    
    def _extract_knowledge(self, user_input: str, ai_response: str):
        """Extract and store knowledge from interactions"""
        try:
            # Extract key information from factual responses
            if 'According to Wikipedia' in ai_response or 'Web search result' in ai_response:
                # Extract the main topic from user input
                topic = self._extract_topic_from_query(user_input)
                if topic:
                    self.knowledge_base[topic] = {
                        'information': ai_response,
                        'source': 'interaction',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.8
                    }
                    
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
    
    def _extract_topic_from_query(self, query: str) -> Optional[str]:
        """Extract main topic from a query"""
        # Remove common question words
        words_to_remove = ['what', 'is', 'who', 'where', 'when', 'how', 'why', 'the', 'a', 'an']
        topic = query.lower().strip()
        
        for word in words_to_remove:
            topic = topic.replace(word, '').strip()
        
        return topic if len(topic) > 2 else None
    
    def get_learned_response(self, user_input: str) -> Optional[str]:
        """Get a learned response based on previous interactions with context awareness"""
        try:
            input_lower = user_input.lower()
            
            # Check for context-aware responses first
            if self.current_conversation_context['topic']:
                topic = self.current_conversation_context['topic']
                
                # Check if this is related to current conversation topic
                if topic.lower() in input_lower or any(word in input_lower for word in topic.split()):
                    # Look for recent interactions about this topic
                    for interaction in reversed(self.conversation_history[-20:]):
                        if (interaction.get('topic') == topic and 
                            self._similar_inputs(user_input, interaction['user_input'])):
                            return f"[Context: {topic}] {interaction['ai_response']}"
            
            # Check knowledge base for direct matches
            for topic, knowledge in self.knowledge_base.items():
                if topic.lower() in input_lower or input_lower in topic.lower():
                    return knowledge['information']
            
            # Check for similar previous interactions with context
            for interaction in reversed(self.conversation_history[-50:]):
                if self._similar_inputs(user_input, interaction['user_input']):
                    # Add context if available
                    context = interaction.get('topic', '')
                    if context:
                        return f"[Related to: {context}] {interaction['ai_response']}"
                    else:
                        return interaction['ai_response']
            
            # Check conversation patterns for better responses
            if 'conversation_patterns' in self.user_preferences:
                for pattern in reversed(self.user_preferences['conversation_patterns'][-20:]):
                    if self._similar_inputs(user_input, pattern['user_input_pattern']):
                        return pattern['ai_response_pattern']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting learned response: {e}")
            return None
    
    def _similar_inputs(self, input1: str, input2: str, threshold: float = 0.7) -> bool:
        """Check if two inputs are similar"""
        try:
            # Simple similarity check based on common words
            words1 = set(input1.lower().split())
            words2 = set(input2.lower().split())
            
            if not words1 or not words2:
                return False
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union)
            return similarity >= threshold
            
        except Exception:
            return False
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics about the AI's evolution"""
        try:
            total_interactions = len(self.conversation_history)
            knowledge_entries = len(self.knowledge_base)
            user_prefs = len(self.user_preferences)
            
            return {
                'total_interactions': total_interactions,
                'knowledge_entries': knowledge_entries,
                'user_preferences': user_prefs,
                'evolution_level': self._calculate_evolution_level(total_interactions, knowledge_entries),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting evolution stats: {e}")
            return {}
    
    def _calculate_evolution_level(self, interactions: int, knowledge: int) -> str:
        """Calculate the AI's evolution level"""
        score = interactions + (knowledge * 2)
        
        if score < 50:
            return "Novice"
        elif score < 200:
            return "Beginner"
        elif score < 500:
            return "Intermediate"
        elif score < 1000:
            return "Advanced"
        else:
            return "Expert"
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query with enhanced capabilities
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary with response data
        """
        start_time = time.time()
        
        try:
            # Clean and classify query
            clean_query = clean_text(query)
            query_type = classify_query_type(clean_query)
            
            # Debug logging
            logger.info(f"Query: '{query}' -> Clean: '{clean_query}' -> Type: {query_type}")
            
            # Store in conversation history
            self.conversation_history.append({
                'query': clean_query,
                'timestamp': datetime.now().isoformat(),
                'type': query_type
            })
            
            # Process based on query type
            if query_type == "time":
                response = self._handle_time_query(clean_query)
            elif query_type == "weather":
                response = self._handle_weather_query(clean_query)
            elif query_type == "math":
                response = self._handle_math_query(clean_query)
            elif query_type == "definition":
                response = self._handle_definition_query(clean_query)
            elif query_type == "howto":
                response = self._handle_howto_query(clean_query)
            elif query_type == "question":
                response = self._handle_question_query(clean_query)
            else:
                response = self._handle_general_query(clean_query)
            
            # Add response to conversation history
            if len(self.conversation_history) > 0:
                self.conversation_history[-1]['response'] = response.get('response', '')
                self.conversation_history[-1]['type'] = response.get('type', 'general')
            
            # Save data
            self._save_persistent_data()
            
            return {
                'success': True,
                'response': response.get('response', 'I apologize, but I could not process your query.'),
                'type': response.get('type', 'general'),
                'confidence': response.get('confidence', 0.5),
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'response': f"I encountered an error while processing your query: {str(e)}",
                'type': 'error',
                'confidence': 0.0,
                'response_time': time.time() - start_time
            }
    
    def _is_math_query(self, query: str) -> bool:
        """Check if query contains mathematical expressions"""
        query_lower = query.lower().strip()
        
        # Skip if it's clearly a factual question
        if any(word in query_lower for word in ['who is', 'what is', 'where is', 'when', 'how']):
            return False
        
        # Check for actual math patterns
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic operations like 2+3
            r'sin|cos|tan|log|sqrt|exp',   # Math functions
            r'integral|derivative|limit',  # Calculus
            r'equation|solve|calculate',   # Math keywords
            r'^[\d\s\+\-\*\/\^\(\)\.]+$',  # Pure math expression
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_factual_question(self, query: str) -> bool:
        """Check if query is a factual question that needs research"""
        query_lower = query.lower().strip()
        
        # Check for question words that indicate factual research needed
        question_patterns = [
            r'what is\s+\w+',  # "what is dog"
            r'who is\s+\w+',   # "who is sunnyleone"
            r'where is\s+\w+', # "where is china"
            r'when\s+\w+',     # "when did..."
            r'how\s+\w+',      # "how does..."
            r'why\s+\w+',      # "why is..."
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def _is_current_data_request(self, query: str) -> bool:
        """Check if query is asking for current/real-time data"""
        query_lower = query.lower().strip()
        
        current_data_keywords = [
            'time', 'date', 'now', 'today', 'current',
            'weather', 'temperature', 'forecast',
            'news', 'latest', 'recent', 'breaking'
        ]
        
        return any(keyword in query_lower for keyword in current_data_keywords)
    
    def _auto_correct_query(self, query: str) -> str:
        """Auto-correct common typos and misspellings"""
        try:
            # Common typo corrections
            common_typos = {
                'wat': 'what', 'wut': 'what', 'wat is': 'what is',
                'hw': 'how', 'hw to': 'how to',
                'wen': 'when', 'were': 'where',
                'wether': 'weather', 'wheather': 'weather',
                'nuz': 'news', 'newz': 'news',
                'tym': 'time', 'tme': 'time',
                'calcuate': 'calculate', 'calclate': 'calculate',
                'maths': 'math', 'mathematics': 'math',
                'temprature': 'temperature', 'temp': 'temperature',
                'forcast': 'forecast', 'forcaste': 'forecast',
                'updte': 'update', 'updae': 'update',
                'infrmation': 'information', 'info': 'information',
                'qestion': 'question', 'quesion': 'question',
                'ansr': 'answer', 'answr': 'answer',
                'explain': 'explain', 'explane': 'explain',
                'defin': 'define', 'defne': 'define',
                'search': 'search', 'serch': 'search',
                'find': 'find', 'fnd': 'find',
                'tell': 'tell', 'tel': 'tell',
                'show': 'show', 'shw': 'show',
                'help': 'help', 'hlp': 'help'
            }
            
            corrected_query = query.lower()
            
            # Apply common typo corrections
            for typo, correction in common_typos.items():
                corrected_query = corrected_query.replace(typo, correction)
            
            # Fix common word boundary issues
            corrected_query = re.sub(r'\b(\w+)\s+\1\b', r'\1', corrected_query)  # Remove duplicates
            corrected_query = re.sub(r'\s+', ' ', corrected_query)  # Fix multiple spaces
            
            # Store the correction for learning
            if corrected_query != query.lower():
                self.typo_corrections[query.lower()] = corrected_query
            
            return corrected_query.strip()
            
        except Exception as e:
            logger.error(f"Auto-correct error: {e}")
            return query
    
    def _update_conversation_context(self, query: str):
        """Update conversation context for memory and continuity"""
        try:
            # Extract topic from query
            topic = self._extract_topic_from_query(query)
            
            # Update conversation chain
            self.current_conversation_context['conversation_chain'].append({
                'query': query,
                'topic': topic,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 interactions for context
            if len(self.current_conversation_context['conversation_chain']) > 10:
                self.current_conversation_context['conversation_chain'] = \
                    self.current_conversation_context['conversation_chain'][-10:]
            
            # Update current topic
            if topic:
                self.current_conversation_context['topic'] = topic
            
            # Update last questions
            self.current_conversation_context['last_questions'].append(query)
            if len(self.current_conversation_context['last_questions']) > 5:
                self.current_conversation_context['last_questions'] = \
                    self.current_conversation_context['last_questions'][-5:]
                    
        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
    
    def _handle_context_aware_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle context-aware follow-up questions"""
        try:
            query_lower = query.lower()
            
            # Check for follow-up indicators
            follow_up_indicators = [
                'what about', 'how about', 'tell me more', 'more details',
                'explain', 'elaborate', 'what else', 'anything else',
                'related', 'similar', 'other', 'different',
                'why', 'how', 'when', 'where', 'who else'
            ]
            
            is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
            
            if is_follow_up and self.current_conversation_context['topic']:
                # This is a follow-up question, provide context-aware response
                topic = self.current_conversation_context['topic']
                
                # Search for more information about the current topic
                enhanced_result = self._search_enhanced_info(topic, query)
                if enhanced_result:
                    return {
                        'success': True,
                        'response': f"Following up on {topic}:\n\n{enhanced_result}",
                        'type': 'context_aware',
                        'topic': topic
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling context-aware query: {e}")
            return None
    
    def _handle_math_query(self, query: str) -> Dict[str, Any]:
        """Handle mathematical queries"""
        try:
            # Extract mathematical expression
            # Remove common words and keep numbers and operators
            math_expr = re.sub(r'[a-zA-Z\s]', '', query)
            math_expr = re.sub(r'[^0-9+\-*/().]', '', math_expr)
            
            if math_expr:
                try:
                    result = eval(math_expr)
                    response = f"Result: {math_expr} = {result}"
                    confidence = 0.9
                except:
                    response = "I couldn't evaluate that mathematical expression."
                    confidence = 0.0
            else:
                response = "Please provide a mathematical expression to calculate."
                confidence = 0.0
            
            return {
                'response': response,
                'type': 'math',
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error handling math query: {e}")
            return {
                'response': "I couldn't process the mathematical query.",
                'type': 'math',
                'confidence': 0.0
            }
    
    def _extract_math_expression(self, query: str) -> str:
        """Extract mathematical expression from query"""
        # Remove common words and keep math symbols
        words_to_remove = ['calculate', 'solve', 'what is', 'compute', 'evaluate', 'find']
        expr = query.lower()
        
        for word in words_to_remove:
            expr = expr.replace(word, '').strip()
        
        # Clean up the expression
        expr = re.sub(r'[^\d\s\+\-\*\/\^\(\)\.\w]', '', expr)
        return expr.strip()
    
    def _evaluate_math_expression(self, expr: str) -> Dict[str, Any]:
        """Evaluate mathematical expression using multiple methods"""
        try:
            # Method 1: Simple eval (safe for basic operations)
            if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expr):
                result = eval(expr)
                return {
                    'success': True,
                    'result': result,
                    'method': 'basic_eval'
                }
            
            # Method 2: SymPy for advanced math
            try:
                x = sympy.Symbol('x')
                result = sympy.sympify(expr)
                if result.is_number:
                    return {
                        'success': True,
                        'result': float(result),
                        'method': 'sympy'
                    }
                else:
                    return {
                        'success': True,
                        'result': str(result),
                        'method': 'sympy_symbolic'
                    }
            except:
                pass
            
            # Method 3: Custom function evaluation
            for func_name, func in self.math_functions.items():
                if func_name in expr:
                    # Replace function calls
                    expr_clean = expr.replace(func_name, f'math.{func_name}')
                    try:
                        result = eval(expr_clean, {'math': math})
                        return {
                            'success': True,
                            'result': result,
                            'method': 'custom_functions'
                        }
                    except:
                        continue
            
            return {
                'success': False,
                'error': 'Expression not recognized'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _handle_news_query(self, query: str) -> Dict[str, Any]:
        """Handle news and current events queries using free sources"""
        try:
            # Extract topic from query
            topic = self._extract_topic_from_query(query)
            
            # Use a free news source - BBC News RSS
            try:
                url = "https://feeds.bbci.co.uk/news/rss.xml"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                if items:
                    news_summary = f"Here are the latest news headlines:\n\n"
                    
                    for i, item in enumerate(items[:5], 1):
                        title = item.find('title').text if item.find('title') else 'No title'
                        description = item.find('description').text if item.find('description') else ''
                        news_summary += f"{i}. {title}\n"
                        if description:
                            news_summary += f"   {description[:100]}...\n\n"
                    
                    return {
                        'success': True,
                        'response': news_summary,
                        'type': 'news',
                        'source': 'BBC News RSS'
                    }
                else:
                    return {
                        'success': False,
                        'response': "I couldn't fetch the latest news at the moment.",
                        'type': 'news'
                    }
                    
            except Exception as e:
                logger.error(f"News fetch error: {e}")
                return {
                    'success': False,
                    'response': f"Error fetching news: {str(e)}",
                    'type': 'news'
                }
                
        except Exception as e:
            logger.error(f"News query error: {e}")
            return {
                'success': False,
                'response': f"Error processing news query: {str(e)}",
                'type': 'news'
            }
    
    def _handle_factual_query(self, query: str) -> Dict[str, Any]:
        """Handle factual questions using Wikipedia and web search"""
        try:
            # Extract the main topic from the query
            topic = self._extract_topic_from_query(query)
            
            # Try Wikipedia first with the extracted topic
            wiki_result = self._search_wikipedia(topic)
            if wiki_result['success']:
                return wiki_result
            
            # Try web search with the original query
            web_result = self._web_search(query)
            if web_result['success']:
                return web_result
            
            # Try a broader search
            broader_result = self._web_search(topic)
            if broader_result['success']:
                return broader_result
            
            return {
                'success': False,
                'response': f"I couldn't find specific information about '{topic}'. Try rephrasing your question or being more specific.",
                'type': 'factual'
            }
            
        except Exception as e:
            logger.error(f"Factual query error: {e}")
            return {
                'success': False,
                'response': f"Error processing factual query: {str(e)}",
                'type': 'factual'
            }
    
    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for information"""
        try:
            # Clean the query
            clean_query = query.strip()
            
            # Search for pages
            search_results = wikipedia.search(clean_query, results=5)
            
            if not search_results:
                return {'success': False}
            
            # Try to get the most relevant result
            for result in search_results:
                try:
                    # Get the page
                    page = wikipedia.page(result, auto_suggest=False)
                    
                    # Extract summary
                    summary = wikipedia.summary(result, sentences=4)
                    
                    # Check if the summary is relevant to the query
                    if self._is_relevant_summary(summary, clean_query):
                        return {
                            'success': True,
                            'response': f"According to Wikipedia:\n\n{summary}\n\nSource: {page.url}",
                            'type': 'wikipedia',
                            'title': result,
                            'url': page.url
                        }
                except wikipedia.DisambiguationError as e:
                    # Handle disambiguation
                    options = e.options[:5]
                    response = f"'{clean_query}' could refer to several things:\n\n"
                    for i, option in enumerate(options, 1):
                        response += f"{i}. {option}\n"
                    response += "\nPlease be more specific."
                    
                    return {
                        'success': True,
                        'response': response,
                        'type': 'wikipedia_disambiguation'
                    }
                except Exception:
                    continue
            
            # If no relevant result found, return the first one
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                summary = wikipedia.summary(search_results[0], sentences=3)
                
                return {
                    'success': True,
                    'response': f"According to Wikipedia:\n\n{summary}\n\nSource: {page.url}",
                    'type': 'wikipedia',
                    'title': search_results[0],
                    'url': page.url
                }
            except Exception:
                return {'success': False}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _is_relevant_summary(self, summary: str, query: str) -> bool:
        """Check if a summary is relevant to the query"""
        query_words = set(query.lower().split())
        summary_words = set(summary.lower().split())
        
        # Check for word overlap
        overlap = query_words.intersection(summary_words)
        return len(overlap) > 0
    
    def _search_enhanced_info(self, topic: str, query: str) -> Optional[str]:
        """Search for enhanced information with multiple sources"""
        try:
            # Try Wikipedia first
            wiki_result = self._search_wikipedia(topic)
            if wiki_result['success']:
                return wiki_result['response']
            
            # Try web search
            web_result = self._web_search(f"{topic} {query}")
            if web_result['success']:
                return web_result['response']
            
            # Try broader search
            broader_result = self._web_search(topic)
            if broader_result['success']:
                return broader_result['response']
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            return None
    
    def _continuous_data_feed(self):
        """Continuously feed data to keep AI updated"""
        try:
            now = datetime.now()
            
            # Update news every 30 minutes
            if (not self.data_feeds['last_news_update'] or 
                (now - self.data_feeds['last_news_update']).seconds > 1800):
                self._update_news_feed()
                self.data_feeds['last_news_update'] = now
            
            # Update weather data every 15 minutes
            if (not self.data_feeds['last_weather_update'] or 
                (now - self.data_feeds['last_weather_update']).seconds > 900):
                self._update_weather_feed()
                self.data_feeds['last_weather_update'] = now
            
            # Update knowledge base every hour
            if (not self.data_feeds['last_knowledge_update'] or 
                (now - self.data_feeds['last_knowledge_update']).seconds > 3600):
                self._update_knowledge_base()
                self.data_feeds['last_knowledge_update'] = now
                
        except Exception as e:
            logger.error(f"Continuous data feed error: {e}")
    
    def _update_news_feed(self):
        """Update news feed with latest information"""
        try:
            # Fetch latest news and store in knowledge base
            news_result = self._handle_news_query("latest news")
            if news_result['success']:
                self.knowledge_base['latest_news'] = {
                    'information': news_result['response'],
                    'source': 'news_feed',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.9
                }
                logger.info("News feed updated")
        except Exception as e:
            logger.error(f"News feed update error: {e}")
    
    def _update_weather_feed(self):
        """Update weather feed with current conditions"""
        try:
            # Update weather for common locations
            common_locations = ['London', 'New York', 'Tokyo', 'Paris', 'Sydney']
            for location in common_locations:
                weather_result = self._handle_weather_query(f"weather in {location}")
                if weather_result['success']:
                    self.knowledge_base[f'weather_{location.lower()}'] = {
                        'information': weather_result['response'],
                        'source': 'weather_feed',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.8
                    }
            logger.info("Weather feed updated")
        except Exception as e:
            logger.error(f"Weather feed update error: {e}")
    
    def _update_knowledge_base(self):
        """Update knowledge base with trending topics"""
        try:
            # Search for trending topics and update knowledge
            trending_topics = ['artificial intelligence', 'technology', 'science', 'health']
            for topic in trending_topics:
                wiki_result = self._search_wikipedia(topic)
                if wiki_result['success']:
                    self.knowledge_base[f'trending_{topic}'] = {
                        'information': wiki_result['response'],
                        'source': 'knowledge_feed',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.7
                    }
            logger.info("Knowledge base updated")
        except Exception as e:
            logger.error(f"Knowledge base update error: {e}")
    
    def _web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search using DuckDuckGo instant answer API"""
        try:
            # Use DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Abstract'):
                return {
                    'success': True,
                    'response': f"Web search result:\n\n{data['Abstract']}\n\nSource: {data.get('AbstractURL', 'Unknown')}",
                    'type': 'web_search',
                    'source': data.get('AbstractURL')
                }
            else:
                return {'success': False}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_weather_query(self, query: str) -> Dict[str, Any]:
        """Handle weather queries using free weather API"""
        try:
            # Extract location from query
            location = self._extract_location_from_query(query)
            
            if not location:
                return {
                    'success': False,
                    'response': 'Please specify a location for weather information.',
                    'type': 'weather'
                }
            
            # Use wttr.in (free weather service)
            try:
                url = f"https://wttr.in/{location}?format=3"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                weather_info = response.text.strip()
                
                if weather_info and not weather_info.startswith('ERROR'):
                    return {
                        'success': True,
                        'response': f"Weather information:\n\n{weather_info}",
                        'type': 'weather',
                        'location': location
                    }
                else:
                    return {
                        'success': False,
                        'response': f"Weather information not available for {location}.",
                        'type': 'weather'
                    }
                    
            except Exception as e:
                logger.error(f"Weather API error: {e}")
                return {
                    'success': False,
                    'response': f"Error fetching weather information: {str(e)}",
                    'type': 'weather'
                }
                
        except Exception as e:
            logger.error(f"Weather query error: {e}")
            return {
                'success': False,
                'response': f"Error processing weather query: {str(e)}",
                'type': 'weather'
            }
    
    def _handle_time_query(self, query: str) -> Dict[str, Any]:
        """Handle time and date queries"""
        try:
            now = datetime.now()
            query_lower = query.lower()
            
            # More specific time queries
            if 'current time' in query_lower or 'what time' in query_lower:
                time_str = now.strftime("%I:%M:%S %p")
                response = f"The current time is {time_str}"
            elif 'time' in query_lower:
                time_str = now.strftime("%I:%M %p")
                response = f"The current time is {time_str}"
            elif 'date' in query_lower and 'time' not in query_lower:
                date_str = now.strftime("%A, %B %d, %Y")
                response = f"Today is {date_str}"
            elif 'today' in query_lower:
                date_str = now.strftime("%A, %B %d, %Y")
                response = f"Today is {date_str}"
            else:
                # Default: show both date and time
                datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
                response = f"Current date and time: {datetime_str}"
            
            return {
                'success': True,
                'response': response,
                'type': 'time',
                'timestamp': now.isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"Error getting time information: {str(e)}",
                'type': 'time'
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries"""
        try:
            # Check if it's a greeting
            greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
            if query.lower() in greetings:
                response = f"Hello! I'm your AI assistant. How can I help you today?"
                confidence = 0.9
            else:
                # Try to search for information about the query
                search_results = self._multi_search(query)
                if search_results:
                    response = self._format_multi_source_response(query, search_results)
                    confidence = 0.8
                else:
                    response = f"I understand you're asking about '{query}'. Let me provide you with the most relevant and current information available."
                    confidence = 0.6
            
            return {
                'response': response,
                'type': 'general',
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            return {
                'response': "I couldn't process your query.",
                'type': 'general',
                'confidence': 0.0
            }
    
    def _handle_question_query(self, query: str) -> Dict[str, Any]:
        """Handle general question queries"""
        try:
            # Try to find answer in knowledge base first
            keywords = query.lower().split()
            best_match = None
            best_score = 0
            
            for key, value in self.knowledge_base.items():
                if isinstance(value, dict) and 'value' in value:
                    key_words = key.lower().split()
                    score = len(set(keywords) & set(key_words))
                    if score > best_score:
                        best_score = score
                        best_match = value
            
            if best_match and best_score > 0:
                response = f"Based on my knowledge base: {best_match['value']}"
                confidence = 0.7
            else:
                response = f"I understand you're asking about '{query}'. Let me provide you with the most relevant and current information available."
                confidence = 0.6
            
            return {
                'response': response,
                'type': 'question',
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error handling question query: {e}")
            return {
                'response': "I couldn't process the question.",
                'type': 'question',
                'confidence': 0.0
            }
    
    def _handle_definition_query(self, query: str) -> Dict[str, Any]:
        """Handle definition queries using multiple search engines"""
        try:
            # Extract search term
            search_term = query.lower()
            for prefix in ['what is ', 'who is ', 'define ', 'meaning of ']:
                if search_term.startswith(prefix):
                    search_term = search_term[len(prefix):].strip()
                    break
            
            if not search_term:
                return {
                    'response': "Please specify what you'd like me to search for.",
                    'type': 'search',
                    'confidence': 0.0
                }
            
            # Use multiple search engines
            search_results = self._multi_search(search_term)
            
            if search_results:
                # Combine results from multiple sources
                response = self._format_multi_source_response(search_term, search_results)
                confidence = 0.9
            else:
                response = f"I couldn't find information about '{search_term}' from multiple sources."
                confidence = 0.0
            
            return {
                'response': response,
                'type': 'multi_search',
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error handling definition query: {e}")
            return {
                'response': "I couldn't process the search query.",
                'type': 'search',
                'confidence': 0.0
            }
    
    def _multi_search(self, query: str) -> List[Dict]:
        """Search multiple engines for information"""
        results = []
        
        try:
            # 1. DuckDuckGo Instant Answer
            ddg_result = self._search_duckduckgo(query)
            if ddg_result:
                results.append(ddg_result)
            
            # 2. Google Search
            google_results = self._search_google(query)
            if google_results:
                results.extend(google_results[:2])
            
            # 3. Bing Search
            bing_results = self._search_bing(query)
            if bing_results:
                results.extend(bing_results[:2])
            
            # 4. Web search via DuckDuckGo
            web_results = self._search_web(query)
            if web_results:
                results.extend(web_results[:2])
            
            # 5. Wikipedia as backup
            wiki_result = self._search_wikipedia(query)
            if wiki_result:
                results.append(wiki_result)
            
            # 6. News search
            news_results = self._search_news(query)
            if news_results:
                results.extend(news_results[:1])
            
            # 7. Stack Overflow for technical queries
            if any(word in query.lower() for word in ['code', 'programming', 'error', 'bug', 'function', 'api']):
                stack_results = self._search_stackoverflow(query)
                if stack_results:
                    results.extend(stack_results[:1])
            
        except Exception as e:
            logger.error(f"Error in multi-search: {e}")
        
        return results
    
    def _search_duckduckgo(self, query: str) -> Dict:
        """Search DuckDuckGo for instant answers"""
        try:
            import requests
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Abstract'):
                    return {
                        'source': 'DuckDuckGo',
                        'title': data.get('AbstractSource', 'DuckDuckGo'),
                        'content': data['Abstract'],
                        'url': data.get('AbstractURL', ''),
                        'type': 'instant_answer'
                    }
                
                if data.get('Answer'):
                    return {
                        'source': 'DuckDuckGo',
                        'title': 'Direct Answer',
                        'content': data['Answer'],
                        'url': data.get('AnswerURL', ''),
                        'type': 'direct_answer'
                    }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return None
    
    def _search_web(self, query: str) -> List[Dict]:
        """Search web for general information"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus
            
            # Use a search aggregator
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('div', class_='result')[:3]:
                    title_elem = result.find('h2')
                    snippet_elem = result.find('div', class_='snippet')
                    link_elem = result.find('a')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'source': 'Web Search',
                            'title': title_elem.get_text().strip(),
                            'content': snippet_elem.get_text().strip(),
                            'url': link_elem.get('href', '') if link_elem else '',
                            'type': 'web_result'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
        
        return []
    
    def _search_wikipedia(self, query: str) -> Dict:
        """Search Wikipedia as backup"""
        try:
            import wikipedia
            
            # Search Wikipedia
            search_results = wikipedia.search(query, results=1)
            if search_results:
                page_title = search_results[0]
                summary = wikipedia.summary(page_title, sentences=2)
                
                return {
                    'source': 'Wikipedia',
                    'title': page_title,
                    'content': summary,
                    'url': f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                    'type': 'wikipedia'
                }
                
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
        
        return None
    
    def _search_news(self, query: str) -> List[Dict]:
        """Search for recent news"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus
            
            # Search Google News
            news_url = f"https://news.google.com/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(news_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract news articles
                for article in soup.find_all('article')[:2]:
                    title_elem = article.find('h3')
                    if title_elem:
                        results.append({
                            'source': 'News',
                            'title': title_elem.get_text().strip(),
                            'content': f"Recent news about {query}",
                            'url': '',
                            'type': 'news'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"News search error: {e}")
        
        return []
    
    def _search_google(self, query: str) -> List[Dict]:
        """Search Google for information"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus
            
            # Use a Google search proxy
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract Google search results
                for result in soup.find_all('div', class_='g')[:3]:
                    title_elem = result.find('h3')
                    snippet_elem = result.find('div', class_='VwiC3b')
                    link_elem = result.find('a')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'source': 'Google',
                            'title': title_elem.get_text().strip(),
                            'content': snippet_elem.get_text().strip(),
                            'url': link_elem.get('href', '') if link_elem else '',
                            'type': 'google_result'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        return []
    
    def _search_bing(self, query: str) -> List[Dict]:
        """Search Bing for information"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus
            
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract Bing search results
                for result in soup.find_all('li', class_='b_algo')[:3]:
                    title_elem = result.find('h2')
                    snippet_elem = result.find('p')
                    link_elem = result.find('a')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'source': 'Bing',
                            'title': title_elem.get_text().strip(),
                            'content': snippet_elem.get_text().strip(),
                            'url': link_elem.get('href', '') if link_elem else '',
                            'type': 'bing_result'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Bing search error: {e}")
        
        return []
    
    def _search_stackoverflow(self, query: str) -> List[Dict]:
        """Search Stack Overflow for technical information"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote_plus
            
            search_url = f"https://stackoverflow.com/search?q={quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract Stack Overflow results
                for result in soup.find_all('div', class_='question-summary')[:2]:
                    title_elem = result.find('h3')
                    excerpt_elem = result.find('div', class_='excerpt')
                    
                    if title_elem:
                        results.append({
                            'source': 'Stack Overflow',
                            'title': title_elem.get_text().strip(),
                            'content': excerpt_elem.get_text().strip() if excerpt_elem else f"Technical discussion about {query}",
                            'url': '',
                            'type': 'stackoverflow_result'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Stack Overflow search error: {e}")
        
        return []
    
    def _format_multi_source_response(self, query: str, results: List[Dict]) -> str:
        """Format response from multiple sources"""
        if not results:
            return f"I couldn't find information about '{query}' from multiple search engines."
        
        # Get unique sources used
        sources_used = list(set(r['source'] for r in results))
        
        # Start with main answer from best source
        best_result = results[0]
        response = f"{best_result['content']}"
        
        # Add source attribution
        if best_result['url']:
            response += f"\n\nSource: {best_result['url']}"
        
        # Add additional sources if available (but keep it concise)
        if len(results) > 1:
            response += f"\n\nAdditional sources: {', '.join(sources_used[1:3])}"
        
        # Show which search engines were used
        response += f"\n\n*Searched: {', '.join(sources_used)}*"
        
        return response
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract main topic from a query"""
        query_lower = query.lower().strip()
        
        # Remove question words and common words
        words_to_remove = [
            'what', 'is', 'who', 'where', 'when', 'how', 'why',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'news', 'latest', 'current', 'recent', 'about'
        ]
        
        # Split into words and filter out common words
        words = query_lower.split()
        filtered_words = [word for word in words if word not in words_to_remove and len(word) > 2]
        
        # Join the remaining words
        topic = ' '.join(filtered_words)
        
        # If no meaningful words found, return the original query
        if not topic or len(topic) < 2:
            return query_lower
        
        return topic
    
    def _extract_location_from_query(self, query: str) -> str:
        """Extract location from weather query"""
        # Remove common words
        words_to_remove = ['weather', 'in', 'at', 'the', 'what', 'is', 'like', 'for']
        location = query.lower()
        
        for word in words_to_remove:
            location = location.replace(word, '').strip()
        
        return location if location else ''
    
    def cleanup(self):
        """Cleanup and save all data before shutdown"""
        self._save_persistent_data()
        logger.info("Enhanced AI cleanup completed")
