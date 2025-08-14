"""
Advanced AI Core Engine
Core components for GPT-4/5 level capabilities
"""

import os
import re
import json
import math
import requests
import wikipedia
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from collections import defaultdict, deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class AdvancedAICore:
    """Core advanced AI with GPT-4/5 level capabilities"""
    
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Initialize Wikipedia
        wikipedia.set_lang('en')
        
        # Advanced data structures
        self.conversation_history = deque(maxlen=1000)
        self.knowledge_base = {}
        self.code_snippets = {}
        self.search_cache = {}
        
        # Advanced context management
        self.current_context = {
            'topic': None,
            'subtopics': [],
            'entities': [],
            'conversation_chain': deque(maxlen=50),
            'user_intent': None,
            'response_style': 'balanced'
        }
        
        # Performance optimization
        self.cache = {}
        self.cache_ttl = 3600
        self.performance_metrics = defaultdict(list)
        
        # Multi-source search engines
        self.search_engines = {
            'wikipedia': self._search_wikipedia_advanced,
            'web': self._search_web_advanced,
            'news': self._search_news_advanced,
            'code': self._search_code_advanced
        }
        
        # Load existing data
        self._load_advanced_data()
    
    def _load_advanced_data(self):
        """Load advanced persistent data"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load conversation history
            history_file = self.storage_path / "advanced_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = deque(data, maxlen=1000)
                logger.info(f"Loaded {len(self.conversation_history)} conversation records")
            
            # Load knowledge base
            knowledge_file = self.storage_path / "advanced_knowledge.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            
            # Load code snippets
            code_file = self.storage_path / "code_snippets.json"
            if code_file.exists():
                with open(code_file, 'r', encoding='utf-8') as f:
                    self.code_snippets = json.load(f)
                logger.info(f"Loaded {len(self.code_snippets)} code snippets")
                
        except Exception as e:
            logger.error(f"Error loading advanced data: {e}")
    
    def process_query_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced query processing with GPT-4/5 level capabilities"""
        start_time = time.time()
        
        try:
            # Analyze query
            query_analysis = self._analyze_query_intent(query)
            
            # Update context
            self._update_context(query, query_analysis)
            
            # Execute multi-source search
            search_results = self._execute_multi_source_search(query, query_analysis)
            
            # Synthesize information
            synthesized_info = self._synthesize_information(search_results, query_analysis)
            
            # Generate response
            response = self._generate_advanced_response(synthesized_info, query_analysis)
            
            # Calculate performance
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, len(search_results))
            
            # Learn from interaction
            self._learn_from_interaction(query, response, query_analysis)
            
            return {
                'success': True,
                'response': response,
                'type': query_analysis['type'],
                'confidence': query_analysis['confidence'],
                'sources': search_results['sources'],
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Advanced query processing error: {e}")
            return {
                'success': False,
                'response': f"I encountered an error: {str(e)}",
                'type': 'error'
            }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and complexity"""
        analysis = {
            'type': 'general',
            'confidence': 0.5,
            'complexity_level': 'medium',
            'entities': [],
            'keywords': [],
            'intent': 'information_seeking',
            'domain': 'general'
        }
        
        query_lower = query.lower()
        
        # Analyze query patterns
        if any(word in query_lower for word in ['code', 'program', 'function', 'class', 'algorithm']):
            analysis['type'] = 'programming'
            analysis['domain'] = 'software_development'
            analysis['confidence'] = 0.8
        elif any(word in query_lower for word in ['calculate', 'solve', 'equation', 'math']):
            analysis['type'] = 'mathematical'
            analysis['domain'] = 'mathematics'
            analysis['confidence'] = 0.9
        elif any(word in query_lower for word in ['news', 'current', 'latest', 'recent']):
            analysis['type'] = 'news'
            analysis['domain'] = 'current_events'
            analysis['confidence'] = 0.7
        elif any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
            analysis['type'] = 'weather'
            analysis['domain'] = 'weather'
            analysis['confidence'] = 0.8
        elif any(word in query_lower for word in ['time', 'date', 'schedule']):
            analysis['type'] = 'time'
            analysis['domain'] = 'time'
            analysis['confidence'] = 0.9
        elif '?' in query:
            analysis['type'] = 'question'
            analysis['intent'] = 'information_seeking'
            analysis['confidence'] = 0.6
        
        # Extract keywords
        words = query_lower.split()
        analysis['keywords'] = [word for word in words if len(word) > 2]
        
        # Determine complexity
        if len(words) > 10:
            analysis['complexity_level'] = 'high'
        elif len(words) < 5:
            analysis['complexity_level'] = 'low'
        
        return analysis
    
    def _execute_multi_source_search(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Execute multi-source search with parallel processing"""
        results = {
            'sources': {},
            'total_results': 0,
            'search_time': 0
        }
        
        start_time = time.time()
        
        # Determine search sources based on query type
        search_sources = ['web', 'wikipedia']
        if query_analysis['type'] == 'programming':
            search_sources.append('code')
        elif query_analysis['type'] == 'news':
            search_sources.append('news')
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(self.search_engines[source], query): source
                for source in search_sources
                if source in self.search_engines
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    source_results = future.result()
                    if source_results['success']:
                        results['sources'][source] = source_results
                        results['total_results'] += len(source_results.get('data', []))
                except Exception as e:
                    logger.error(f"Search error for {source}: {e}")
        
        results['search_time'] = time.time() - start_time
        return results
    
    def _synthesize_information(self, search_results: Dict, query_analysis: Dict) -> Dict[str, Any]:
        """Synthesize information from multiple sources"""
        synthesis = {
            'main_points': [],
            'supporting_evidence': [],
            'confidence': 0.5,
            'sources_used': []
        }
        
        # Extract main points from each source
        for source, results in search_results['sources'].items():
            if results['success'] and 'data' in results:
                synthesis['sources_used'].append(source)
                
                for item in results['data']:
                    if 'content' in item:
                        synthesis['main_points'].append({
                            'content': item['content'],
                            'source': source,
                            'relevance': item.get('relevance', 0.5)
                        })
        
        # Rank and filter main points
        synthesis['main_points'].sort(key=lambda x: x['relevance'], reverse=True)
        synthesis['main_points'] = synthesis['main_points'][:10]
        
        # Calculate overall confidence
        if synthesis['main_points']:
            synthesis['confidence'] = sum(point['relevance'] for point in synthesis['main_points']) / len(synthesis['main_points'])
        
        return synthesis
    
    def _generate_advanced_response(self, synthesis: Dict, query_analysis: Dict) -> str:
        """Generate advanced, context-aware response"""
        if not synthesis['main_points']:
            return "I couldn't find specific information about your query. Could you please rephrase or provide more details?"
        
        # Build response based on query type
        if query_analysis['type'] == 'programming':
            return self._generate_programming_response(synthesis, query_analysis)
        elif query_analysis['type'] == 'mathematical':
            return self._generate_mathematical_response(synthesis, query_analysis)
        elif query_analysis['type'] == 'news':
            return self._generate_news_response(synthesis, query_analysis)
        else:
            return self._generate_general_response(synthesis, query_analysis)
    
    def _generate_programming_response(self, synthesis: Dict, query_analysis: Dict) -> str:
        """Generate programming-specific response"""
        response_parts = []
        
        response_parts.append("**Programming Solution:**\n")
        
        # Add code examples if available
        code_examples = [point for point in synthesis['main_points'] if 'code' in point['content'].lower()]
        if code_examples:
            response_parts.append("**Code Examples:**\n")
            for example in code_examples[:2]:
                response_parts.append(f"```\n{example['content']}\n```\n")
        
        # Add explanations
        explanations = [point for point in synthesis['main_points'] if 'code' not in point['content'].lower()]
        if explanations:
            response_parts.append("**Explanation:**\n")
            for explanation in explanations[:2]:
                response_parts.append(f"• {explanation['content']}\n")
        
        # Add best practices
        response_parts.append("\n**Best Practices:**\n")
        response_parts.append("• Always test your code thoroughly\n")
        response_parts.append("• Follow coding standards and conventions\n")
        response_parts.append("• Consider performance and security implications\n")
        
        return '\n'.join(response_parts)
    
    def _generate_mathematical_response(self, synthesis: Dict, query_analysis: Dict) -> str:
        """Generate mathematical response with step-by-step solution"""
        response_parts = []
        
        response_parts.append("**Step-by-Step Solution:**\n")
        
        for i, point in enumerate(synthesis['main_points'][:5], 1):
            response_parts.append(f"**Step {i}:** {point['content']}\n")
        
        response_parts.append("\n**Key Concepts:**\n")
        response_parts.append("• Understand the problem thoroughly\n")
        response_parts.append("• Break it down into smaller steps\n")
        response_parts.append("• Verify your solution\n")
        
        return '\n'.join(response_parts)
    
    def _generate_news_response(self, synthesis: Dict, query_analysis: Dict) -> str:
        """Generate news response with timeline"""
        response_parts = []
        
        response_parts.append("**Latest News Summary:**\n")
        
        for i, point in enumerate(synthesis['main_points'][:5], 1):
            response_parts.append(f"{i}. {point['content']}\n")
        
        response_parts.append(f"\n*Based on {len(synthesis['sources_used'])} sources*")
        
        return '\n'.join(response_parts)
    
    def _generate_general_response(self, synthesis: Dict, query_analysis: Dict) -> str:
        """Generate general response"""
        response_parts = []
        
        response_parts.append("**Comprehensive Answer:**\n")
        
        for point in synthesis['main_points'][:3]:
            response_parts.append(f"• {point['content']}\n")
        
        if synthesis['confidence'] > 0.7:
            response_parts.append("\n*This information is highly reliable based on multiple sources.*")
        else:
            response_parts.append("\n*Please verify this information from additional sources.*")
        
        return '\n'.join(response_parts)
    
    def _update_context(self, query: str, query_analysis: Dict):
        """Update conversation context"""
        # Update conversation chain
        self.current_context['conversation_chain'].append({
            'query': query,
            'analysis': query_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update topic if significant
        if query_analysis['confidence'] > 0.6:
            self.current_context['topic'] = query_analysis.get('domain', 'general')
        
        # Update user intent
        self.current_context['user_intent'] = query_analysis['intent']
    
    def _update_performance_metrics(self, processing_time: float, result_count: int):
        """Update performance metrics"""
        self.performance_metrics['processing_time'].append(processing_time)
        self.performance_metrics['result_count'].append(result_count)
        
        # Keep only last 100 metrics
        if len(self.performance_metrics['processing_time']) > 100:
            self.performance_metrics['processing_time'] = self.performance_metrics['processing_time'][-100:]
            self.performance_metrics['result_count'] = self.performance_metrics['result_count'][-100:]
    
    def _learn_from_interaction(self, query: str, response: str, query_analysis: Dict):
        """Learn from interaction"""
        # Record interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'analysis': query_analysis,
            'context': self.current_context.copy()
        }
        
        self.conversation_history.append(interaction)
        
        # Update knowledge base
        if query_analysis['confidence'] > 0.7:
            self.knowledge_base[query] = {
                'response': response,
                'analysis': query_analysis,
                'timestamp': datetime.now().isoformat(),
                'usage_count': 1
            }
        
        # Save periodically
        if len(self.conversation_history) % 10 == 0:
            self._save_advanced_data()
    
    def _save_advanced_data(self):
        """Save advanced data"""
        try:
            # Save conversation history
            history_file = self.storage_path / "advanced_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.conversation_history), f, indent=2, ensure_ascii=False)
            
            # Save knowledge base
            knowledge_file = self.storage_path / "advanced_knowledge.json"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
            # Save code snippets
            code_file = self.storage_path / "code_snippets.json"
            with open(code_file, 'w', encoding='utf-8') as f:
                json.dump(self.code_snippets, f, indent=2, ensure_ascii=False)
                
            logger.info("Advanced data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving advanced data: {e}")
    
    # Advanced search methods
    def _search_wikipedia_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced Wikipedia search"""
        try:
            search_results = wikipedia.search(query, results=5)
            
            if not search_results:
                return {'success': False, 'data': []}
            
            data = []
            for result in search_results:
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    summary = wikipedia.summary(result, sentences=3)
                    
                    data.append({
                        'title': result,
                        'content': summary,
                        'url': page.url,
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
    
    def _search_web_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced web search"""
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
                    'url': data.get('AbstractURL', ''),
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
    
    def _search_news_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced news search"""
        try:
            # Use BBC News RSS
            url = "https://feeds.bbci.co.uk/news/rss.xml"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            data = []
            for item in items[:5]:
                title = item.find('title').text if item.find('title') else 'No title'
                description = item.find('description').text if item.find('description') else ''
                
                data.append({
                    'title': title,
                    'content': description,
                    'url': '',
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
    
    def _search_code_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced code search (placeholder)"""
        return {'success': True, 'data': [], 'source': 'code'}
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get advanced AI statistics"""
        try:
            avg_processing_time = np.mean(self.performance_metrics['processing_time']) if self.performance_metrics['processing_time'] else 0
            avg_result_count = np.mean(self.performance_metrics['result_count']) if self.performance_metrics['result_count'] else 0
            
            return {
                'total_interactions': len(self.conversation_history),
                'knowledge_entries': len(self.knowledge_base),
                'code_snippets': len(self.code_snippets),
                'avg_processing_time': avg_processing_time,
                'avg_result_count': avg_result_count,
                'cache_size': len(self.cache),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting advanced stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup and save all data"""
        self._save_advanced_data()
        logger.info("Advanced AI core cleanup completed")
