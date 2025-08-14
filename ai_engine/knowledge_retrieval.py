"""
Knowledge Retrieval Module for AI System
Handles web search, Wikipedia, APIs, and other knowledge sources
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from urllib.parse import quote_plus, urlparse
import wikipedia
from bs4 import BeautifulSoup
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class KnowledgeSource:
    """Data structure for knowledge source"""
    name: str
    url: str
    content: str
    confidence: float
    timestamp: str
    source_type: str

@dataclass
class SearchResult:
    """Data structure for search result"""
    query: str
    sources: List[KnowledgeSource]
    best_answer: Optional[str] = None
    confidence: float = 0.0
    search_time: float = 0.0

class KnowledgeRetrievalModule:
    """
    Knowledge Retrieval Module for unified access to various knowledge sources
    """
    
    def __init__(self, cache_db_path: str = "storage/knowledge_cache.db"):
        self.cache_db_path = cache_db_path
        self.logger = logging.getLogger(__name__)
        self._init_cache_database()
        
        # Knowledge sources configuration with MULTIPLE search engines
        self.sources = {
            # Primary Search Engines
            "google_search": {
                "enabled": True,
                "priority": 1,  # Highest priority - Google Search
                "timeout": 15,
                "max_results": 5,
                "quality": "high",
                "requires_api_key": True
            },
            "bing_search": {
                "enabled": True,
                "priority": 1,  # Highest priority - Bing Search
                "timeout": 15,
                "max_results": 5,
                "quality": "high",
                "requires_api_key": True
            },
            "duckduckgo": {
                "enabled": True,
                "priority": 2,  # High priority - No API key needed
                "timeout": 10,
                "max_results": 5,
                "quality": "high",
                "requires_api_key": False
            },
            "wikipedia": {
                "enabled": True,
                "priority": 3,  # Medium priority with cross-verification
                "timeout": 10,
                "max_results": 3,
                "quality": "medium",
                "requires_api_key": False
            },
            
            # News Sources
            "news": {
                "enabled": True,
                "priority": 4,
                "timeout": 10,
                "max_results": 3,
                "quality": "high",
                "requires_api_key": False
            },
            "news_api": {
                "enabled": True,
                "priority": 4,
                "timeout": 10,
                "max_results": 3,
                "quality": "high",
                "requires_api_key": True
            },
            
            # Academic Sources
            "arxiv": {
                "enabled": True,
                "priority": 5,
                "timeout": 15,
                "max_results": 3,
                "quality": "high",
                "requires_api_key": False
            },
            "pubmed": {
                "enabled": True,
                "priority": 5,
                "timeout": 15,
                "max_results": 3,
                "quality": "high",
                "requires_api_key": False
            },
            
            # Developer/Technical Sources
            "stackoverflow": {
                "enabled": True,
                "priority": 6,
                "timeout": 10,
                "max_results": 3,
                "quality": "medium",
                "requires_api_key": False
            },
            "github": {
                "enabled": True,
                "priority": 6,
                "timeout": 10,
                "max_results": 3,
                "quality": "medium",
                "requires_api_key": False
            },
            
            # Real-time Data
            "weather": {
                "enabled": True,
                "priority": 7,
                "timeout": 5,
                "max_results": 1,
                "quality": "high",
                "requires_api_key": False
            },
            "time": {
                "enabled": True,
                "priority": 7,
                "timeout": 5,
                "max_results": 1,
                "quality": "high",
                "requires_api_key": False
            },
            
            # Social/Community Sources
            "reddit": {
                "enabled": True,
                "priority": 8,
                "timeout": 10,
                "max_results": 3,
                "quality": "low",
                "requires_api_key": False
            }
        }
        
        # API endpoints for MULTIPLE search engines
        self.api_endpoints = {
            # Search Engines
            "google_search": "https://www.googleapis.com/customsearch/v1",
            "bing_search": "https://api.bing.microsoft.com/v7.0/search",
            "duckduckgo": "https://api.duckduckgo.com/",
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/",
            
            # Specialized APIs
            "weather": "https://wttr.in/",
            "time": "http://worldtimeapi.org/api/ip",
            "news": "https://feeds.bbci.co.uk/news/rss.xml",
            "news_api": "https://newsapi.org/v2/",
            "arxiv": "http://export.arxiv.org/api/query",
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            
            # Additional sources
            "stackoverflow": "https://api.stackexchange.com/2.3/",
            "github": "https://api.github.com/search/",
            "reddit": "https://www.reddit.com/r/",
        }
        
        # Cache settings
        self.cache_duration_hours = 24
        self.max_cache_size = 1000
    
    def _init_cache_database(self):
        """Initialize the knowledge cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    query TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    best_answer TEXT,
                    confidence REAL DEFAULT 0.0,
                    search_time REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create source cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Knowledge cache database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge cache database: {e}")
    
    def retrieve_knowledge(self, query: str, source_types: Optional[List[str]] = None) -> SearchResult:
        """
        Retrieve knowledge from multiple sources based on query type
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self._get_cached_result(query)
            if cached_result:
                self.logger.info(f"Retrieved cached result for query: {query}")
                return cached_result
            
            # Determine which sources to use
            if source_types is None:
                source_types = self._determine_sources_for_query(query)
            
            # Retrieve from sources in parallel
            sources = self._retrieve_from_sources(query, source_types)
            
            # Synthesize best answer
            best_answer, confidence = self._synthesize_answer(query, sources)
            
            # Create search result
            search_result = SearchResult(
                query=query,
                sources=sources,
                best_answer=best_answer,
                confidence=confidence,
                search_time=time.time() - start_time
            )
            
            # Cache the result
            self._cache_result(search_result)
            
            self.logger.info(f"Knowledge retrieval completed - Sources: {len(sources)}, Confidence: {confidence:.3f}")
            return search_result
            
        except Exception as e:
            self.logger.error(f"Error in knowledge retrieval: {e}")
            return SearchResult(
                query=query,
                sources=[],
                best_answer="Knowledge retrieval failed",
                confidence=0.0,
                search_time=time.time() - start_time
            )
    
    def _determine_sources_for_query(self, query: str) -> List[str]:
        """Determine which knowledge sources to use based on query type"""
        query_lower = query.lower()
        
        sources = []
        
        # ALWAYS start with multiple search engines for comprehensive coverage
        sources.extend(["duckduckgo", "google_search", "bing_search"])
        
        # Time-related queries
        if any(word in query_lower for word in ["time", "what time", "current time"]):
            sources.append("time")
        
        # Weather-related queries
        if any(word in query_lower for word in ["weather", "temperature", "forecast", "climate"]):
            sources.append("weather")
        
        # News-related queries
        if any(word in query_lower for word in ["news", "latest", "current events", "what's happening", "recent"]):
            sources.extend(["news", "news_api"])
        
        # Academic/Scientific queries
        if any(word in query_lower for word in ["research", "study", "paper", "academic", "scientific", "journal"]):
            sources.extend(["arxiv", "pubmed"])
        
        # Programming/Technical queries
        if any(word in query_lower for word in ["code", "programming", "github", "stackoverflow", "error", "bug", "api"]):
            sources.extend(["stackoverflow", "github"])
        
        # Factual queries (Wikipedia)
        if any(word in query_lower for word in ["what is", "who is", "definition", "explain", "meaning"]):
            sources.append("wikipedia")
        
        # Community/Opinion queries
        if any(word in query_lower for word in ["opinion", "discussion", "reddit", "community", "forum"]):
            sources.append("reddit")
        
        # Always include Wikipedia for factual questions if not already included
        if "wikipedia" not in sources and any(word in query_lower for word in ["what", "who", "which"]):
            sources.append("wikipedia")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return unique_sources[:8]  # Allow more sources for comprehensive coverage
    
    def _retrieve_from_sources(self, query: str, source_types: List[str]) -> List[KnowledgeSource]:
        """Retrieve knowledge from multiple sources in parallel"""
        sources = []
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=len(source_types)) as executor:
            future_to_source = {
                executor.submit(self._retrieve_from_source, query, source_type): source_type
                for source_type in source_types
                if self.sources[source_type]["enabled"]
            }
            
            for future in as_completed(future_to_source):
                source_type = future_to_source[future]
                try:
                    source_result = future.result()
                    if source_result:
                        sources.extend(source_result)
                except Exception as e:
                    self.logger.error(f"Error retrieving from {source_type}: {e}")
        
        # Sort by confidence and priority
        sources.sort(key=lambda x: (x.confidence, -self.sources.get(x.source_type, {}).get("priority", 0)), reverse=True)
        
        return sources
    
    def _retrieve_from_source(self, query: str, source_type: str) -> List[KnowledgeSource]:
        """Retrieve knowledge from a specific source"""
        try:
            # Primary Search Engines
            if source_type == "google_search":
                return self._search_google(query)
            elif source_type == "bing_search":
                return self._search_bing(query)
            elif source_type == "duckduckgo":
                return self._search_duckduckgo(query)
            elif source_type == "wikipedia":
                return self._search_wikipedia(query)
            
            # News Sources
            elif source_type == "news":
                return self._search_news(query)
            elif source_type == "news_api":
                return self._search_news_api(query)
            
            # Academic Sources
            elif source_type == "arxiv":
                return self._search_arxiv(query)
            elif source_type == "pubmed":
                return self._search_pubmed(query)
            
            # Developer/Technical Sources
            elif source_type == "stackoverflow":
                return self._search_stackoverflow(query)
            elif source_type == "github":
                return self._search_github(query)
            
            # Real-time Data
            elif source_type == "weather":
                return self._search_weather(query)
            elif source_type == "time":
                return self._search_time(query)
            
            # Social/Community Sources
            elif source_type == "reddit":
                return self._search_reddit(query)
            
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving from {source_type}: {e}")
            return []
    
    def _search_google(self, query: str) -> List[KnowledgeSource]:
        """Search Google for information"""
        try:
            # Note: Google Custom Search API requires API key and Custom Search Engine ID
            # For now, we'll use a fallback approach
            search_query = quote_plus(query)
            
            # Use a simple web scraping approach as fallback
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try to get search results from a search aggregator
            url = f"https://www.google.com/search?q={search_query}"
            
            response = requests.get(url, headers=headers, timeout=self.sources["google_search"]["timeout"])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                sources = []
                # Extract search results (this is a simplified approach)
                search_results = soup.find_all('div', class_='g')[:self.sources["google_search"]["max_results"]]
                
                for result in search_results:
                    title_elem = result.find('h3')
                    link_elem = result.find('a')
                    snippet_elem = result.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href', '')
                        snippet = snippet_elem.get_text() if snippet_elem else title
                        
                        sources.append(KnowledgeSource(
                            name=f"Google: {title}",
                            url=link,
                            content=snippet,
                            confidence=0.9,
                            timestamp=datetime.now().isoformat(),
                            source_type="google_search"
                        ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in Google search: {e}")
            return []

    def _search_bing(self, query: str) -> List[KnowledgeSource]:
        """Search Bing for information"""
        try:
            # Note: Bing Search API requires API key
            # For now, we'll use a fallback approach
            search_query = quote_plus(query)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            url = f"https://www.bing.com/search?q={search_query}"
            
            response = requests.get(url, headers=headers, timeout=self.sources["bing_search"]["timeout"])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                sources = []
                # Extract search results
                search_results = soup.find_all('li', class_='b_algo')[:self.sources["bing_search"]["max_results"]]
                
                for result in search_results:
                    title_elem = result.find('h2')
                    link_elem = result.find('a')
                    snippet_elem = result.find('p')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href', '')
                        snippet = snippet_elem.get_text() if snippet_elem else title
                        
                        sources.append(KnowledgeSource(
                            name=f"Bing: {title}",
                            url=link,
                            content=snippet,
                            confidence=0.9,
                            timestamp=datetime.now().isoformat(),
                            source_type="bing_search"
                        ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in Bing search: {e}")
            return []

    def _search_duckduckgo(self, query: str) -> List[KnowledgeSource]:
        """Search DuckDuckGo for information"""
        try:
            search_query = quote_plus(query)
            url = f"{self.api_endpoints['duckduckgo']}?q={search_query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, timeout=self.sources["duckduckgo"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                
                sources = []
                
                # Add abstract if available
                if data.get('Abstract'):
                    sources.append(KnowledgeSource(
                        name="DuckDuckGo Abstract",
                        url=data.get('AbstractURL', ''),
                        content=data['Abstract'],
                        confidence=0.8,
                        timestamp=datetime.now().isoformat(),
                        source_type="duckduckgo"
                    ))
                
                # Add answer if available
                if data.get('Answer'):
                    sources.append(KnowledgeSource(
                        name="DuckDuckGo Answer",
                        url=data.get('AnswerURL', ''),
                        content=data['Answer'],
                        confidence=0.9,
                        timestamp=datetime.now().isoformat(),
                        source_type="duckduckgo"
                    ))
                
                # Add related topics
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:2]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            sources.append(KnowledgeSource(
                                name="DuckDuckGo Related",
                                url=topic.get('FirstURL', ''),
                                content=topic['Text'],
                                confidence=0.7,
                                timestamp=datetime.now().isoformat(),
                                source_type="duckduckgo"
                            ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in DuckDuckGo search: {e}")
            return []

    def _search_wikipedia(self, query: str) -> List[KnowledgeSource]:
        """Search Wikipedia for information"""
        try:
            # Extract search terms
            search_terms = self._extract_search_terms(query)
            
            sources = []
            for term in search_terms[:self.sources["wikipedia"]["max_results"]]:
                try:
                    # Search Wikipedia
                    search_results = wikipedia.search(term, results=1)
                    if search_results:
                        page_title = search_results[0]
                        
                        # Get summary
                        summary = wikipedia.summary(page_title, sentences=3)
                        
                        # Get page URL
                        page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                        
                        sources.append(KnowledgeSource(
                            name=f"Wikipedia: {page_title}",
                            url=page_url,
                            content=summary,
                            confidence=0.8,
                            timestamp=datetime.now().isoformat(),
                            source_type="wikipedia"
                        ))
                        
                except Exception as e:
                    self.logger.error(f"Error searching Wikipedia for '{term}': {e}")
                    continue
            
            return sources
            
        except Exception as e:
            self.logger.error(f"Error in Wikipedia search: {e}")
            return []
    
    def _search_news_api(self, query: str) -> List[KnowledgeSource]:
        """Search news using News API"""
        try:
            # Note: News API requires API key
            # For now, we'll use a fallback approach
            search_query = quote_plus(query)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Use a news aggregator as fallback
            url = f"https://news.google.com/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, headers=headers, timeout=self.sources["news_api"]["timeout"])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                sources = []
                # Extract news articles
                articles = soup.find_all('article')[:self.sources["news_api"]["max_results"]]
                
                for article in articles:
                    title_elem = article.find('h3')
                    link_elem = article.find('a')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href', '')
                        
                        sources.append(KnowledgeSource(
                            name=f"News: {title}",
                            url=link,
                            content=title,
                            confidence=0.7,
                            timestamp=datetime.now().isoformat(),
                            source_type="news_api"
                        ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in News API search: {e}")
            return []

    def _search_arxiv(self, query: str) -> List[KnowledgeSource]:
        """Search arXiv for academic papers"""
        try:
            search_query = quote_plus(query)
            url = f"{self.api_endpoints['arxiv']}?search_query=all:{search_query}&start=0&max_results={self.sources['arxiv']['max_results']}"
            
            response = requests.get(url, timeout=self.sources["arxiv"]["timeout"])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                
                sources = []
                entries = soup.find_all('entry')[:self.sources["arxiv"]["max_results"]]
                
                for entry in entries:
                    title = entry.find('title').get_text() if entry.find('title') else "No title"
                    summary = entry.find('summary').get_text() if entry.find('summary') else title
                    link = entry.find('id').get_text() if entry.find('id') else ""
                    
                    sources.append(KnowledgeSource(
                        name=f"arXiv: {title}",
                        url=link,
                        content=summary,
                        confidence=0.8,
                        timestamp=datetime.now().isoformat(),
                        source_type="arxiv"
                    ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in arXiv search: {e}")
            return []

    def _search_pubmed(self, query: str) -> List[KnowledgeSource]:
        """Search PubMed for medical/scientific papers"""
        try:
            search_query = quote_plus(query)
            url = f"{self.api_endpoints['pubmed']}esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax={self.sources['pubmed']['max_results']}"
            
            response = requests.get(url, timeout=self.sources["pubmed"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                
                sources = []
                if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                    for pmid in data['esearchresult']['idlist'][:self.sources["pubmed"]["max_results"]]:
                        # Get article details
                        detail_url = f"{self.api_endpoints['pubmed']}esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                        detail_response = requests.get(detail_url, timeout=5)
                        
                        if detail_response.status_code == 200:
                            detail_data = detail_response.json()
                            if 'result' in detail_data and pmid in detail_data['result']:
                                article = detail_data['result'][pmid]
                                title = article.get('title', 'No title')
                                abstract = article.get('abstract', title)
                                
                                sources.append(KnowledgeSource(
                                    name=f"PubMed: {title}",
                                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                    content=abstract,
                                    confidence=0.8,
                                    timestamp=datetime.now().isoformat(),
                                    source_type="pubmed"
                                ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in PubMed search: {e}")
            return []

    def _search_stackoverflow(self, query: str) -> List[KnowledgeSource]:
        """Search Stack Overflow for programming questions"""
        try:
            search_query = quote_plus(query)
            url = f"{self.api_endpoints['stackoverflow']}search/advanced?order=desc&sort=activity&q={search_query}&site=stackoverflow"
            
            response = requests.get(url, timeout=self.sources["stackoverflow"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                
                sources = []
                if 'items' in data:
                    for item in data['items'][:self.sources["stackoverflow"]["max_results"]]:
                        title = item.get('title', 'No title')
                        excerpt = item.get('excerpt', title)
                        link = item.get('link', '')
                        
                        sources.append(KnowledgeSource(
                            name=f"Stack Overflow: {title}",
                            url=link,
                            content=excerpt,
                            confidence=0.7,
                            timestamp=datetime.now().isoformat(),
                            source_type="stackoverflow"
                        ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in Stack Overflow search: {e}")
            return []

    def _search_github(self, query: str) -> List[KnowledgeSource]:
        """Search GitHub for code repositories"""
        try:
            search_query = quote_plus(query)
            url = f"{self.api_endpoints['github']}repositories?q={search_query}&sort=stars&order=desc"
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AI-System/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=self.sources["github"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                
                sources = []
                for item in data.get('items', [])[:self.sources["github"]["max_results"]]:
                    name = item.get('full_name', 'No name')
                    description = item.get('description', name)
                    url = item.get('html_url', '')
                    
                    sources.append(KnowledgeSource(
                        name=f"GitHub: {name}",
                        url=url,
                        content=description,
                        confidence=0.7,
                        timestamp=datetime.now().isoformat(),
                        source_type="github"
                    ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in GitHub search: {e}")
            return []

    def _search_reddit(self, query: str) -> List[KnowledgeSource]:
        """Search Reddit for community discussions"""
        try:
            search_query = quote_plus(query)
            url = f"https://www.reddit.com/search.json?q={search_query}&sort=relevance&t=all"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=self.sources["reddit"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                
                sources = []
                if 'data' in data and 'children' in data['data']:
                    for child in data['data']['children'][:self.sources["reddit"]["max_results"]]:
                        post = child['data']
                        title = post.get('title', 'No title')
                        selftext = post.get('selftext', title)
                        url = f"https://reddit.com{post.get('permalink', '')}"
                        
                        sources.append(KnowledgeSource(
                            name=f"Reddit: {title}",
                            url=url,
                            content=selftext,
                            confidence=0.5,
                            timestamp=datetime.now().isoformat(),
                            source_type="reddit"
                        ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in Reddit search: {e}")
            return []
    
    def _search_news(self, query: str) -> List[KnowledgeSource]:
        """Search for latest news"""
        try:
            response = requests.get(self.api_endpoints["news"], timeout=self.sources["news"]["timeout"])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:self.sources["news"]["max_results"]]
                
                sources = []
                for i, item in enumerate(items):
                    title = item.find('title').text if item.find('title') else "No title"
                    description = item.find('description').text if item.find('description') else title
                    link = item.find('link').text if item.find('link') else ""
                    
                    sources.append(KnowledgeSource(
                        name=f"BBC News: {title}",
                        url=link,
                        content=description,
                        confidence=0.6,
                        timestamp=datetime.now().isoformat(),
                        source_type="news"
                    ))
                
                return sources
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in news search: {e}")
            return []
    
    def _search_weather(self, query: str) -> List[KnowledgeSource]:
        """Search for weather information"""
        try:
            # Extract location from query
            location = "London"  # Default location
            if "weather in" in query.lower():
                parts = query.lower().split("weather in")
                if len(parts) > 1:
                    location = parts[1].strip().split()[0].title()
            
            response = requests.get(f"{self.api_endpoints['weather']}{location}?format=3", 
                                 timeout=self.sources["weather"]["timeout"])
            if response.status_code == 200:
                weather_info = response.text.strip()
                
                return [KnowledgeSource(
                    name=f"Weather: {location}",
                    url=f"{self.api_endpoints['weather']}{location}",
                    content=weather_info,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    source_type="weather"
                )]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in weather search: {e}")
            return []
    
    def _search_time(self, query: str) -> List[KnowledgeSource]:
        """Search for current time"""
        try:
            response = requests.get(self.api_endpoints["time"], timeout=self.sources["time"]["timeout"])
            if response.status_code == 200:
                data = response.json()
                current_time = data.get('datetime', '')
                timezone = data.get('timezone', '')
                
                if current_time:
                    time_part = current_time.split('T')[1][:8]  # HH:MM:SS
                    time_info = f"Current time is {time_part} in {timezone}"
                    
                    return [KnowledgeSource(
                        name="Current Time",
                        url=self.api_endpoints["time"],
                        content=time_info,
                        confidence=0.9,
                        timestamp=datetime.now().isoformat(),
                        source_type="time"
                    )]
            
            # Fallback to local time
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M:%S")
            time_info = f"Current time is approximately {current_time}"
            
            return [KnowledgeSource(
                name="Local Time",
                url="",
                content=time_info,
                confidence=0.7,
                timestamp=datetime.now().isoformat(),
                source_type="time"
            )]
            
        except Exception as e:
            self.logger.error(f"Error in time search: {e}")
            return []
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract potential search terms from a query"""
        # Remove common question words
        question_words = query.lower().split()
        stop_words = {"what", "is", "are", "who", "where", "when", "why", "how", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        search_terms = []
        for word in question_words:
            if word not in stop_words and len(word) > 2:
                search_terms.append(word)
        
        # Also try combinations
        if len(search_terms) >= 2:
            search_terms.append(" ".join(search_terms[:2]))
        
        return search_terms[:3]  # Return top 3 terms
    
    def _synthesize_answer(self, query: str, sources: List[KnowledgeSource]) -> Tuple[Optional[str], float]:
        """Synthesize the best answer from multiple sources following Master Retrieval Protocol"""
        if not sources:
            return None, 0.0

        # Sort sources by quality and confidence
        high_quality_sources = [s for s in sources if s.confidence > 0.7]
        medium_quality_sources = [s for s in sources if 0.5 <= s.confidence <= 0.7]
        
        # Cross-verify information between sources
        verified_facts = self._cross_verify_sources(sources)
        
        # Determine confidence level based on source agreement
        confidence = self._calculate_confidence_level(sources, verified_facts)
        
        # Format response according to Master Retrieval Protocol
        formatted_answer = self._format_master_retrieval_response(
            query, sources, verified_facts, confidence
        )
        
        return formatted_answer, confidence

    def _cross_verify_sources(self, sources: List[KnowledgeSource]) -> Dict[str, Any]:
        """Cross-verify facts between multiple sources"""
        verified_facts = {}
        
        if len(sources) < 2:
            return verified_facts
        
        # Extract key facts from each source
        source_facts = {}
        for source in sources:
            key_facts = self._extract_key_facts(source.content)
            source_facts[source.name] = key_facts
        
        # Find facts that appear in multiple sources
        fact_counts = {}
        for source_name, facts in source_facts.items():
            for fact in facts:
                if fact not in fact_counts:
                    fact_counts[fact] = []
                fact_counts[fact].append(source_name)
        
        # Mark facts as verified if they appear in multiple sources
        for fact, source_list in fact_counts.items():
            if len(source_list) >= 2:
                verified_facts[fact] = {
                    'sources': source_list,
                    'confidence': len(source_list) / len(sources)
                }
        
        return verified_facts

    def _extract_key_facts(self, content: str) -> List[str]:
        """Extract key factual statements from content"""
        # Simple fact extraction - can be enhanced with NLP
        sentences = content.split('.')
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have', 'had']):
                facts.append(sentence)
        
        return facts[:5]  # Limit to top 5 facts

    def _calculate_confidence_level(self, sources: List[KnowledgeSource], verified_facts: Dict[str, Any]) -> float:
        """Calculate confidence level based on source agreement and quality"""
        if not sources:
            return 0.0
        
        # Base confidence on source quality and agreement
        total_confidence = 0.0
        source_weights = 0.0
        
        for source in sources:
            weight = 1.0
            if source.confidence > 0.8:
                weight = 1.5  # High confidence sources weighted more
            elif source.confidence < 0.5:
                weight = 0.5  # Low confidence sources weighted less
            
            total_confidence += source.confidence * weight
            source_weights += weight
        
        base_confidence = total_confidence / source_weights if source_weights > 0 else 0.0
        
        # Boost confidence if facts are cross-verified
        verification_boost = len(verified_facts) * 0.1
        
        return min(1.0, base_confidence + verification_boost)

    def _format_master_retrieval_response(self, query: str, sources: List[KnowledgeSource], 
                                       verified_facts: Dict[str, Any], confidence: float) -> str:
        """Format response according to Master Retrieval Protocol"""
        if not sources:
            return "No reliable information found from multiple sources."
        
        # Get the best source for primary answer
        best_source = max(sources, key=lambda x: x.confidence)
        
        # Format the response
        response_parts = []
        
        # Direct Answer
        response_parts.append(f"[DIRECT ANSWER]")
        response_parts.append(f"{best_source.content}")
        
        # Reasoning
        response_parts.append(f"\n[REASONING]")
        response_parts.append(f"I synthesized this answer from {len(sources)} independent sources, ")
        response_parts.append(f"cross-verifying {len(verified_facts)} key facts between multiple sources.")
        
        # Sources
        response_parts.append(f"\n[SOURCES]")
        for i, source in enumerate(sources[:3], 1):  # Top 3 sources
            response_parts.append(f"- Source {i}: {source.name} ({source.url}) - Confidence: {source.confidence:.2f}")
        
        # Confidence Level
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        response_parts.append(f"\n[CONFIDENCE]")
        response_parts.append(f"{confidence_level} confidence ({confidence:.2f}) based on source agreement and cross-verification.")
        
        # Additional Context
        if verified_facts:
            response_parts.append(f"\n[ADDITIONAL CONTEXT]")
            response_parts.append(f"Cross-verified facts: {len(verified_facts)} key facts confirmed across multiple sources.")
        
        return "\n".join(response_parts)
    
    def _get_cached_result(self, query: str) -> Optional[SearchResult]:
        """Get cached result for a query"""
        try:
            query_hash = self._hash_query(query)
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sources, best_answer, confidence, search_time, timestamp
                FROM knowledge_cache
                WHERE query_hash = ? AND 
                      datetime(timestamp) > datetime('now', '-{} hours')
            '''.format(self.cache_duration_hours), (query_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                sources_data = json.loads(row[0])
                sources = [KnowledgeSource(**source_data) for source_data in sources_data]
                
                return SearchResult(
                    query=query,
                    sources=sources,
                    best_answer=row[1],
                    confidence=row[2],
                    search_time=row[3]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached result: {e}")
            return None
    
    def _cache_result(self, search_result: SearchResult):
        """Cache a search result"""
        try:
            query_hash = self._hash_query(search_result.query)
            
            # Convert sources to JSON-serializable format
            sources_data = []
            for source in search_result.sources:
                sources_data.append({
                    'name': source.name,
                    'url': source.url,
                    'content': source.content,
                    'confidence': source.confidence,
                    'timestamp': source.timestamp,
                    'source_type': source.source_type
                })
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_cache 
                (query_hash, query, sources, best_answer, confidence, search_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_hash,
                search_result.query,
                json.dumps(sources_data),
                search_result.best_answer,
                search_result.confidence,
                search_result.search_time,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error caching result: {e}")
    
    def _hash_query(self, query: str) -> str:
        """Create a hash for the query"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get knowledge retrieval statistics"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Get cache statistics
            cursor.execute('''
                SELECT COUNT(*) as total_cached,
                       AVG(confidence) as avg_confidence,
                       AVG(search_time) as avg_search_time
                FROM knowledge_cache
            ''')
            
            cache_stats = cursor.fetchone()
            
            # Get source usage statistics
            cursor.execute('''
                SELECT source_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM source_cache
                GROUP BY source_type
                ORDER BY count DESC
            ''')
            
            source_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                "cache": {
                    "total_cached": cache_stats[0],
                    "average_confidence": cache_stats[1] or 0.0,
                    "average_search_time": cache_stats[2] or 0.0
                },
                "sources": [
                    {"source": row[0], "count": row[1], "avg_confidence": row[2]}
                    for row in source_stats
                ],
                "enabled_sources": [
                    source for source, config in self.sources.items() 
                    if config["enabled"]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting retrieval statistics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the knowledge cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM knowledge_cache')
            cursor.execute('DELETE FROM source_cache')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Knowledge cache cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clean old cache entries
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM knowledge_cache 
                WHERE datetime(timestamp) < datetime('now', '-{} hours')
            '''.format(self.cache_duration_hours * 2))
            
            cursor.execute('''
                DELETE FROM source_cache 
                WHERE datetime(timestamp) < datetime('now', '-{} hours')
            '''.format(self.cache_duration_hours * 2))
            
            conn.commit()
            conn.close()
            
            self.logger.info("Knowledge retrieval module cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
