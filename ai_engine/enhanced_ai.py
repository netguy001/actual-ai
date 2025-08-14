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

logger = logging.getLogger(__name__)


class EnhancedAI:
    """
    Enhanced AI with internet access, math capabilities, and learning features
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
    
    def learn_from_interaction(self, user_input: str, ai_response: str, feedback: Optional[Dict] = None):
        """Learn from each interaction to improve future responses"""
        try:
            # Record the interaction with enhanced context
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'ai_response': ai_response,
                'feedback': feedback,
                'input_type': self._classify_input_type(user_input),
                'response_type': self._classify_response_type(ai_response),
                'conversation_context': self.current_conversation_context.copy(),
                'topic': self.current_conversation_context.get('topic'),
                'related_questions': self.current_conversation_context.get('last_questions', [])[-3:]
            }
            
            self.conversation_history.append(interaction)
            
            # Extract knowledge from the interaction
            self._extract_knowledge(user_input, ai_response)
            
            # Update conversation patterns
            self._update_conversation_patterns(user_input, ai_response)
            
            # Run continuous data feed
            self._continuous_data_feed()
            
            # Save data periodically (every 3 interactions for better memory)
            if len(self.conversation_history) % 3 == 0:
                self._save_persistent_data()
                
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
        Process user query and determine the best response method
        """
        # Auto-correct the query first
        corrected_query = self._auto_correct_query(query)
        original_query = query
        
        if corrected_query != query:
            logger.info(f"Auto-corrected: '{query}' -> '{corrected_query}'")
            query = corrected_query
        
        query_lower = query.lower().strip()
        
        # Update conversation context
        self._update_conversation_context(query)
        
        # Check for context-aware follow-up questions
        context_response = self._handle_context_aware_query(query)
        if context_response:
            return context_response
        
        # Check for time/date FIRST (highest priority for current data)
        if any(word in query_lower for word in ['time', 'date', 'today', 'now', 'current time', 'what time']):
            return self._handle_time_query(query)
        
        # Check for weather
        if 'weather' in query_lower:
            return self._handle_weather_query(query)
        
        # Check for current events/news
        if any(word in query_lower for word in ['news', 'current', 'latest', 'today', 'recent']):
            return self._handle_news_query(query)
        
        # Check for math expressions
        if self._is_math_query(query):
            return self._handle_math_query(query)
        
        # Check for factual questions (but be more specific)
        if self._is_factual_question(query):
            return self._handle_factual_query(query)
        
        # Check for learned responses last (to avoid overriding current data)
        learned_response = self.get_learned_response(query)
        if learned_response and not self._is_current_data_request(query):
            return {
                'success': True,
                'response': f"[Learned Response] {learned_response}",
                'type': 'learned',
                'source': 'knowledge_base'
            }
        
        # Default to general response
        return self._handle_general_query(query)
    
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
            # Extract math expression
            math_expr = self._extract_math_expression(query)
            
            if not math_expr:
                return {
                    'success': False,
                    'response': 'I couldn\'t identify a mathematical expression in your query.',
                    'type': 'math'
                }
            
            # Try different math approaches
            result = self._evaluate_math_expression(math_expr)
            
            if result['success']:
                return {
                    'success': True,
                    'response': f"The result of {math_expr} = {result['result']}",
                    'type': 'math',
                    'expression': math_expr,
                    'result': result['result'],
                    'method': result['method']
                }
            else:
                return {
                    'success': False,
                    'response': f"I couldn't evaluate the expression '{math_expr}'. {result['error']}",
                    'type': 'math'
                }
                
        except Exception as e:
            logger.error(f"Math query error: {e}")
            return {
                'success': False,
                'response': f"Error processing mathematical query: {str(e)}",
                'type': 'math'
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
        """Handle general queries with intelligent responses"""
        try:
            # Try to provide a helpful response based on query content
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['hello', 'hi', 'hey']):
                response = "Hello! I'm your enhanced AI assistant. I can help you with math, news, weather, factual questions, and much more. What would you like to know?"
            
            elif any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
                response = "I'm an enhanced AI assistant with the following capabilities:\n\n" + \
                          "🔢 **Math & Calculations**: Complex mathematical expressions, equations, calculus\n" + \
                          "📰 **Current News**: Latest news from BBC and other sources\n" + \
                          "🌍 **Factual Information**: Wikipedia searches and web research\n" + \
                          "🌤️ **Weather**: Current weather for any location\n" + \
                          "⏰ **Time & Date**: Current time and date information\n" + \
                          "🧠 **General Knowledge**: Answer questions and provide insights\n" + \
                          "📚 **Learning & Evolution**: I learn from every interaction and continuously improve!\n\n" + \
                          "Just ask me anything! For example:\n" + \
                          "- \"Calculate 2^10 + sqrt(144)\"\n" + \
                          "- \"What's the latest news?\"\n" + \
                          "- \"What is quantum computing?\"\n" + \
                          "- \"Weather in New York\"\n" + \
                          "- \"What time is it?\""
            
            elif '?' in query:
                response = f"That's an interesting question about '{query}'. Let me search for the most current and accurate information for you."
                # Try to get factual information
                factual_result = self._handle_factual_query(query)
                if factual_result['success']:
                    response = factual_result['response']
            
            else:
                response = f"I understand you're asking about '{query}'. Let me provide you with the most relevant and current information available."
            
            return {
                'success': True,
                'response': response,
                'type': 'general'
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"Error processing query: {str(e)}",
                'type': 'general'
            }
    
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
