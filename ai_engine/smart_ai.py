"""
Smart AI with Auto-Correct and Advanced Features
GPT-4/5 level capabilities with intelligent decision making
"""

import re
import json
import requests
import wikipedia
import sympy
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from collections import defaultdict, deque
import time
from difflib import get_close_matches

logger = logging.getLogger(__name__)

class SmartAI:
    """Smart AI with auto-correct and advanced capabilities"""
    
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Initialize Wikipedia
        wikipedia.set_lang('en')
        
        # Smart data structures
        self.conversation_memory = deque(maxlen=1000)
        self.knowledge_base = {}
        self.user_preferences = {}
        self.typo_corrections = {}
        
        # Context management
        self.current_context = {
            'topic': None,
            'conversation_chain': deque(maxlen=50),
            'user_intent': None,
            'response_style': 'balanced'
        }
        
        # Auto-correct dictionary
        self.common_words = self._load_common_words()
        self.programming_terms = self._load_programming_terms()
        
        # Load existing data
        self._load_smart_data()
    
    def _load_common_words(self) -> set:
        """Load common English words for auto-correct"""
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us', 'code', 'program', 'function', 'class', 'algorithm', 'data', 'system', 'computer', 'software', 'hardware',
            'network', 'database', 'api', 'web', 'internet', 'cloud', 'security', 'performance', 'optimization', 'development', 'testing', 'debugging'
        }
        return common_words
    
    def _load_programming_terms(self) -> set:
        """Load programming-specific terms"""
        programming_terms = {
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'typescript',
            'html', 'css', 'sql', 'mongodb', 'mysql', 'postgresql', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'laravel', 'git', 'github',
            'algorithm', 'data structure', 'object oriented', 'functional', 'procedural', 'declarative', 'imperative',
            'recursion', 'iteration', 'sorting', 'searching', 'graph', 'tree', 'stack', 'queue', 'linked list',
            'array', 'hash table', 'binary search', 'quick sort', 'merge sort', 'bubble sort', 'selection sort'
        }
        return programming_terms
    
    def _load_smart_data(self):
        """Load smart persistent data"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load conversation memory
            memory_file = self.storage_path / "smart_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_memory = deque(data, maxlen=1000)
                logger.info(f"Loaded {len(self.conversation_memory)} conversation records")
            
            # Load knowledge base
            knowledge_file = self.storage_path / "smart_knowledge.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            
            # Load typo corrections
            typo_file = self.storage_path / "typo_corrections.json"
            if typo_file.exists():
                with open(typo_file, 'r', encoding='utf-8') as f:
                    self.typo_corrections = json.load(f)
                logger.info(f"Loaded {len(self.typo_corrections)} typo corrections")
                
        except Exception as e:
            logger.error(f"Error loading smart data: {e}")
    
    def process_smart_query(self, query: str) -> Dict[str, Any]:
        """Process query with smart features including auto-correct"""
        start_time = time.time()
        
        try:
            # Auto-correct the query
            corrected_query = self._auto_correct_query(query)
            original_query = query
            
            # Analyze intent
            query_analysis = self._analyze_smart_intent(corrected_query)
            
            # Check conversation memory for context
            context_response = self._check_conversation_memory(corrected_query)
            if context_response:
                return {
                    'success': True,
                    'response': context_response,
                    'type': 'context_memory',
                    'confidence': 0.9,
                    'auto_corrected': corrected_query != original_query,
                    'original_query': original_query,
                    'corrected_query': corrected_query
                }
            
            # Execute smart search
            search_results = self._execute_smart_search(corrected_query, query_analysis)
            
            # Generate smart response
            response = self._generate_smart_response(search_results, query_analysis)
            
            # Update memory
            self._update_conversation_memory(original_query, corrected_query, response, query_analysis)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'response': response,
                'type': query_analysis['type'],
                'confidence': query_analysis['confidence'],
                'auto_corrected': corrected_query != original_query,
                'original_query': original_query,
                'corrected_query': corrected_query,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Smart query processing error: {e}")
            return {
                'success': False,
                'response': f"I encountered an error: {str(e)}",
                'type': 'error'
            }
    
    def _auto_correct_query(self, query: str) -> str:
        """Auto-correct spelling mistakes in query"""
        try:
            words = query.split()
            corrected_words = []
            
            for word in words:
                # Skip if it's a common word or programming term
                if word.lower() in self.common_words or word.lower() in self.programming_terms:
                    corrected_words.append(word)
                    continue
                
                # Check for common programming typos
                if word.lower() in self.typo_corrections:
                    corrected_words.append(self.typo_corrections[word.lower()])
                    continue
                
                # Try to find close matches
                all_words = self.common_words.union(self.programming_terms)
                matches = get_close_matches(word.lower(), all_words, n=1, cutoff=0.8)
                
                if matches:
                    corrected_words.append(matches[0])
                else:
                    corrected_words.append(word)
            
            corrected_query = ' '.join(corrected_words)
            
            # Store the correction for future use
            if corrected_query != query:
                self.typo_corrections[query.lower()] = corrected_query.lower()
            
            return corrected_query
            
        except Exception as e:
            logger.error(f"Auto-correct error: {e}")
            return query
    
    def _analyze_smart_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent with smart features"""
        analysis = {
            'type': 'general',
            'confidence': 0.5,
            'complexity_level': 'medium',
            'intent': 'information_seeking',
            'domain': 'general',
            'requires_code': False,
            'requires_math': False
        }
        
        query_lower = query.lower()
        
        # Programming detection
        if any(term in query_lower for term in ['code', 'program', 'function', 'class', 'algorithm', 'api', 'database']):
            analysis['type'] = 'programming'
            analysis['domain'] = 'software_development'
            analysis['confidence'] = 0.8
            analysis['requires_code'] = True
        
        # Math detection
        elif any(term in query_lower for term in ['calculate', 'solve', 'equation', 'math', 'formula', 'compute']):
            analysis['type'] = 'mathematical'
            analysis['domain'] = 'mathematics'
            analysis['confidence'] = 0.9
            analysis['requires_math'] = True
        
        # News detection
        elif any(term in query_lower for term in ['news', 'current', 'latest', 'recent', 'breaking']):
            analysis['type'] = 'news'
            analysis['domain'] = 'current_events'
            analysis['confidence'] = 0.7
        
        # Weather detection
        elif any(term in query_lower for term in ['weather', 'temperature', 'forecast', 'climate']):
            analysis['type'] = 'weather'
            analysis['domain'] = 'weather'
            analysis['confidence'] = 0.8
        
        # Time detection
        elif any(term in query_lower for term in ['time', 'date', 'schedule', 'calendar']):
            analysis['type'] = 'time'
            analysis['domain'] = 'time'
            analysis['confidence'] = 0.9
        
        # Question detection
        elif '?' in query:
            analysis['type'] = 'question'
            analysis['intent'] = 'information_seeking'
            analysis['confidence'] = 0.6
        
        # Complexity analysis
        word_count = len(query.split())
        if word_count > 15:
            analysis['complexity_level'] = 'high'
        elif word_count < 5:
            analysis['complexity_level'] = 'low'
        
        return analysis
    
    def _check_conversation_memory(self, query: str) -> Optional[str]:
        """Check conversation memory for context-aware responses"""
        try:
            if not self.conversation_memory:
                return None
            
            # Get recent conversations
            recent_conversations = list(self.conversation_memory)[-10:]
            
            # Check for related topics
            query_words = set(query.lower().split())
            
            for conv in recent_conversations:
                if 'query' in conv and 'response' in conv:
                    conv_words = set(conv['query'].lower().split())
                    
                    # Check for word overlap
                    overlap = query_words.intersection(conv_words)
                    if len(overlap) >= 2:  # At least 2 words in common
                        return f"Based on our previous conversation about {conv['query']}, here's additional information: {conv['response']}"
            
            return None
            
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return None
    
    def _execute_smart_search(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Execute smart search based on query type"""
        results = {
            'sources': {},
            'total_results': 0
        }
        
        try:
            # Wikipedia search
            if query_analysis['type'] in ['general', 'question']:
                wiki_results = self._search_wikipedia_smart(query)
                if wiki_results['success']:
                    results['sources']['wikipedia'] = wiki_results
                    results['total_results'] += len(wiki_results.get('data', []))
            
            # Web search
            web_results = self._search_web_smart(query)
            if web_results['success']:
                results['sources']['web'] = web_results
                results['total_results'] += len(web_results.get('data', []))
            
            # News search
            if query_analysis['type'] == 'news':
                news_results = self._search_news_smart(query)
                if news_results['success']:
                    results['sources']['news'] = news_results
                    results['total_results'] += len(news_results.get('data', []))
            
            # Math calculation
            if query_analysis['requires_math']:
                math_results = self._calculate_math_smart(query)
                if math_results['success']:
                    results['sources']['math'] = math_results
                    results['total_results'] += 1
            
        except Exception as e:
            logger.error(f"Smart search error: {e}")
        
        return results
    
    def _search_wikipedia_smart(self, query: str) -> Dict[str, Any]:
        """Smart Wikipedia search"""
        try:
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return {'success': False, 'data': []}
            
            data = []
            for result in search_results:
                try:
                    summary = wikipedia.summary(result, sentences=2)
                    data.append({
                        'title': result,
                        'content': summary,
                        'relevance': 0.8
                    })
                except Exception:
                    continue
            
            return {
                'success': True,
                'data': data,
                'source': 'wikipedia'
            }
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return {'success': False, 'data': []}
    
    def _search_web_smart(self, query: str) -> Dict[str, Any]:
        """Smart web search"""
        try:
            # Use DuckDuckGo Instant Answer API
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
            
            results = []
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Web Result'),
                    'content': data['Abstract'],
                    'relevance': 0.9
                })
            
            return {
                'success': True,
                'data': results,
                'source': 'web'
            }
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {'success': False, 'data': []}
    
    def _search_news_smart(self, query: str) -> Dict[str, Any]:
        """Smart news search"""
        try:
            # Use BBC News RSS
            url = "https://feeds.bbci.co.uk/news/rss.xml"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            data = []
            for item in items[:3]:
                title = item.find('title').text if item.find('title') else 'No title'
                description = item.find('description').text if item.find('description') else ''
                
                data.append({
                    'title': title,
                    'content': description,
                    'relevance': 0.7
                })
            
            return {
                'success': True,
                'data': data,
                'source': 'news'
            }
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return {'success': False, 'data': []}
    
    def _calculate_math_smart(self, query: str) -> Dict[str, Any]:
        """Smart math calculation"""
        try:
            # Extract mathematical expressions
            math_pattern = r'[\d\+\-\*\/\(\)\^\.\s]+'
            math_expressions = re.findall(math_pattern, query)
            
            if not math_expressions:
                return {'success': False, 'data': []}
            
            # Try to evaluate the expression
            for expr in math_expressions:
                try:
                    # Clean the expression
                    clean_expr = re.sub(r'[^\d\+\-\*\/\(\)\^\.\s]', '', expr).strip()
                    if clean_expr:
                        result = eval(clean_expr)
                        return {
                            'success': True,
                            'data': [{
                                'title': 'Mathematical Calculation',
                                'content': f"Result: {clean_expr} = {result}",
                                'relevance': 0.9
                            }],
                            'source': 'math'
                        }
                except Exception:
                    continue
            
            return {'success': False, 'data': []}
            
        except Exception as e:
            logger.error(f"Math calculation error: {e}")
            return {'success': False, 'data': []}
    
    def _generate_smart_response(self, search_results: Dict, query_analysis: Dict) -> str:
        """Generate smart, context-aware response"""
        try:
            if not search_results['sources']:
                return "I couldn't find specific information about your query. Could you please rephrase or provide more details?"
            
            response_parts = []
            
            # Add auto-correct notification if needed
            if query_analysis.get('auto_corrected'):
                response_parts.append("**Note:** I've auto-corrected your query for better results.\n")
            
            # Generate response based on type
            if query_analysis['type'] == 'programming':
                response_parts.append(self._generate_programming_response_smart(search_results))
            elif query_analysis['type'] == 'mathematical':
                response_parts.append(self._generate_math_response_smart(search_results))
            elif query_analysis['type'] == 'news':
                response_parts.append(self._generate_news_response_smart(search_results))
            else:
                response_parts.append(self._generate_general_response_smart(search_results))
            
            # Add confidence indicator
            if query_analysis['confidence'] > 0.8:
                response_parts.append("\n\n✅ *High confidence response*")
            elif query_analysis['confidence'] < 0.6:
                response_parts.append("\n\n⚠️ *Please verify this information*")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble generating a response. Please try again."
    
    def _generate_programming_response_smart(self, search_results: Dict) -> str:
        """Generate smart programming response"""
        response_parts = []
        response_parts.append("**Programming Solution:**\n")
        
        for source, results in search_results['sources'].items():
            if results['success'] and 'data' in results:
                for item in results['data'][:2]:
                    response_parts.append(f"• {item['content']}\n")
        
        response_parts.append("\n**Best Practices:**\n")
        response_parts.append("• Test your code thoroughly\n")
        response_parts.append("• Follow coding standards\n")
        response_parts.append("• Consider performance and security\n")
        
        return '\n'.join(response_parts)
    
    def _generate_math_response_smart(self, search_results: Dict) -> str:
        """Generate smart math response"""
        response_parts = []
        response_parts.append("**Mathematical Solution:**\n")
        
        for source, results in search_results['sources'].items():
            if results['success'] and 'data' in results:
                for item in results['data']:
                    response_parts.append(f"• {item['content']}\n")
        
        return '\n'.join(response_parts)
    
    def _generate_news_response_smart(self, search_results: Dict) -> str:
        """Generate smart news response"""
        response_parts = []
        response_parts.append("**Latest News:**\n")
        
        for source, results in search_results['sources'].items():
            if results['success'] and 'data' in results:
                for i, item in enumerate(results['data'][:3], 1):
                    response_parts.append(f"{i}. {item['content']}\n")
        
        return '\n'.join(response_parts)
    
    def _generate_general_response_smart(self, search_results: Dict) -> str:
        """Generate smart general response"""
        response_parts = []
        response_parts.append("**Answer:**\n")
        
        for source, results in search_results['sources'].items():
            if results['success'] and 'data' in results:
                for item in results['data'][:2]:
                    response_parts.append(f"• {item['content']}\n")
        
        return '\n'.join(response_parts)
    
    def _update_conversation_memory(self, original_query: str, corrected_query: str, response: str, query_analysis: Dict):
        """Update conversation memory"""
        try:
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'original_query': original_query,
                'corrected_query': corrected_query,
                'response': response,
                'analysis': query_analysis,
                'context': self.current_context.copy()
            }
            
            self.conversation_memory.append(memory_entry)
            
            # Update knowledge base
            if query_analysis['confidence'] > 0.7:
                self.knowledge_base[corrected_query] = {
                    'response': response,
                    'analysis': query_analysis,
                    'timestamp': datetime.now().isoformat(),
                    'usage_count': 1
                }
            
            # Save periodically
            if len(self.conversation_memory) % 10 == 0:
                self._save_smart_data()
                
        except Exception as e:
            logger.error(f"Memory update error: {e}")
    
    def _save_smart_data(self):
        """Save smart data"""
        try:
            # Save conversation memory
            memory_file = self.storage_path / "smart_memory.json"
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.conversation_memory), f, indent=2, ensure_ascii=False)
            
            # Save knowledge base
            knowledge_file = self.storage_path / "smart_knowledge.json"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
            # Save typo corrections
            typo_file = self.storage_path / "typo_corrections.json"
            with open(typo_file, 'w', encoding='utf-8') as f:
                json.dump(self.typo_corrections, f, indent=2, ensure_ascii=False)
                
            logger.info("Smart data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving smart data: {e}")
    
    def get_smart_stats(self) -> Dict[str, Any]:
        """Get smart AI statistics"""
        try:
            return {
                'total_conversations': len(self.conversation_memory),
                'knowledge_entries': len(self.knowledge_base),
                'typo_corrections': len(self.typo_corrections),
                'current_context': self.current_context,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting smart stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup and save all data"""
        self._save_smart_data()
        logger.info("Smart AI cleanup completed")
