"""Web search tool for online information retrieval"""

from typing import Optional, Dict, Any, List
from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from tavily import TavilyClient

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()


class SearchTool:
    """Web search tool supporting multiple search APIs"""
    
    def __init__(
        self,
        search_provider: Optional[str] = None,
        max_results: int = 5
    ):
        """Initialize search tool
        
        Args:
            search_provider: Search provider to use ('serper' or 'tavily')
            max_results: Maximum number of search results
        """
        self.max_results = max_results
        self.search_provider = search_provider or self._determine_provider()
        self._tool = None
        
        logger.info(f"Search tool initialized with provider: {self.search_provider}")
    
    def _determine_provider(self) -> str:
        """Determine which search provider to use based on available API keys
        
        Returns:
            Provider name ('serper' or 'tavily')
            
        Raises:
            ValueError: If no search API is configured
        """
        if settings.serper_api_key:
            return "serper"
        elif settings.tavily_api_key:
            return "tavily"
        else:
            raise ValueError(
                "No search API key configured. Please set SERPER_API_KEY or "
                "TAVILY_API_KEY in your .env file."
            )
    
    def search(self, query: str) -> str:
        """Perform a web search
        
        Args:
            query: Search query
            
        Returns:
            Search results as a formatted string
        """
        logger.info(f"Performing web search: '{query}'")
        
        try:
            if self.search_provider == "serper":
                return self._search_serper(query)
            elif self.search_provider == "tavily":
                return self._search_tavily(query)
            else:
                raise ValueError(f"Unknown search provider: {self.search_provider}")
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return f"Error performing search: {str(e)}"
    
    def _search_serper(self, query: str) -> str:
        """Search using Google Serper API
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            search = GoogleSerperAPIWrapper(
                serper_api_key=settings.serper_api_key,
                k=self.max_results
            )
            results = search.run(query)
            logger.info(f"Serper search completed successfully")
            return results
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            raise
    
    def _search_tavily(self, query: str) -> str:
        """Search using Tavily API
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query, max_results=self.max_results)
            
            # Format results
            results = []
            for result in response.get('results', []):
                results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {result.get('content', 'N/A')}\n"
                )
            
            formatted_results = "\n---\n".join(results)
            logger.info(f"Tavily search completed successfully")
            return formatted_results if formatted_results else "No results found."
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            raise
    
    def get_langchain_tool(self) -> Tool:
        """Get LangChain Tool wrapper
        
        Returns:
            LangChain Tool instance
        """
        if self._tool is None:
            self._tool = Tool(
                name="web_search",
                func=self.search,
                description=(
                    "Useful for searching the internet for current information, "
                    "recent events, or facts that may not be in the knowledge base. "
                    "Input should be a search query string."
                )
            )
        return self._tool
    
    def search_with_metadata(self, query: str) -> Dict[str, Any]:
        """Perform search and return structured results with metadata
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results and metadata
        """
        logger.info(f"Performing search with metadata: '{query}'")
        
        try:
            if self.search_provider == "tavily":
                client = TavilyClient(api_key=settings.tavily_api_key)
                response = client.search(query, max_results=self.max_results)
                
                return {
                    "query": query,
                    "provider": "tavily",
                    "results": response.get('results', []),
                    "num_results": len(response.get('results', []))
                }
            else:
                # For Serper, return simpler format
                results = self._search_serper(query)
                return {
                    "query": query,
                    "provider": "serper",
                    "results": results,
                    "num_results": 1  # Serper returns pre-formatted text
                }
        except Exception as e:
            logger.error(f"Error in search_with_metadata: {e}")
            return {
                "query": query,
                "provider": self.search_provider,
                "results": [],
                "error": str(e)
            }
