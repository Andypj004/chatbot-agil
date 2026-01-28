"""Main chatbot agent with RAG and search capabilities"""

from typing import Optional, Dict, Any, List
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

from src.llm.base import BaseLLMProvider
from src.rag.retriever import RAGRetriever
from src.tools.search_tool import SearchTool
from src.core.logger import get_logger
from src.core.config import settings

logger = get_logger()


class ChatbotAgent:
    """Intelligent chatbot agent with RAG and web search capabilities"""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        rag_retriever: Optional[RAGRetriever] = None,
        search_tool: Optional[SearchTool] = None,
        enable_memory: bool = True
    ):
        """Initialize chatbot agent
        
        Args:
            llm_provider: LLM provider instance
            rag_retriever: Optional RAG retriever for knowledge base
            search_tool: Optional search tool for online information
            enable_memory: Whether to enable conversation memory
        """
        self.llm_provider = llm_provider
        self.rag_retriever = rag_retriever
        self.search_tool = search_tool
        self.enable_memory = enable_memory
        
        # Initialize memory if enabled
        self.memory = None
        if self.enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        logger.info(
            f"Chatbot agent initialized with {len(self.tools)} tools, "
            f"memory={'enabled' if enable_memory else 'disabled'}"
        )
    
    def _setup_tools(self) -> List[Tool]:
        """Set up available tools for the agent
        
        Returns:
            List of Tool instances
        """
        tools = []
        
        # Add RAG tool if retriever is available
        if self.rag_retriever:
            rag_tool = Tool(
                name="knowledge_base",
                func=self._query_knowledge_base,
                description=(
                    "Useful for answering questions using the internal knowledge base. "
                    "Use this when the question is about topics that might be in uploaded documents. "
                    "Input should be a clear question."
                )
            )
            tools.append(rag_tool)
            logger.info("Added knowledge base tool")
        
        # Add search tool if available
        if self.search_tool:
            tools.append(self.search_tool.get_langchain_tool())
            logger.info("Added web search tool")
        
        return tools
    
    def _query_knowledge_base(self, query: str) -> str:
        """Query the RAG knowledge base
        
        Args:
            query: Query string
            
        Returns:
            Answer from knowledge base
        """
        try:
            result = self.rag_retriever.query(query, return_sources=False)
            return result["answer"]
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return f"Error accessing knowledge base: {str(e)}"
    
    def _create_agent(self):
        """Create the LangChain agent
        
        Returns:
            Initialized agent executor
        """
        llm = self.llm_provider.get_llm()
        
        # If no tools are available, we'll use the agent without tools
        if not self.tools:
            logger.warning("No tools available for agent")
            return None
        
        agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agent
    
    def chat(
        self,
        message: str,
        use_rag: bool = True,
        use_search: bool = True
    ) -> Dict[str, Any]:
        """Send a message to the chatbot
        
        Args:
            message: User message
            use_rag: Whether to use RAG for this query
            use_search: Whether to allow web search
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing chat message: '{message}'")
        logger.info(f"Options: use_rag={use_rag}, use_search={use_search}")
        
        try:
            # If agent is available and tools should be used
            if self.agent and (use_rag or use_search):
                # Temporarily disable tools if needed
                active_tools = []
                if use_rag and self.rag_retriever:
                    active_tools.append(self.tools[0])  # RAG tool
                if use_search and self.search_tool:
                    # Find search tool
                    search_tools = [t for t in self.tools if t.name == "web_search"]
                    active_tools.extend(search_tools)
                
                if active_tools:
                    # Use agent with selected tools
                    temp_agent = initialize_agent(
                        tools=active_tools,
                        llm=self.llm_provider.get_llm(),
                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                        memory=self.memory,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=5,
                        early_stopping_method="generate"
                    )
                    response = temp_agent.run(message)
                else:
                    # No tools, just use LLM directly
                    llm = self.llm_provider.get_llm()
                    response = llm.predict(message)
            else:
                # Use LLM directly without tools
                llm = self.llm_provider.get_llm()
                response = llm.predict(message)
            
            logger.info("Chat response generated successfully")
            
            return {
                "response": response,
                "provider": self.llm_provider.get_provider_name(),
                "model": self.llm_provider.get_default_model(),
                "used_rag": use_rag and self.rag_retriever is not None,
                "used_search": use_search and self.search_tool is not None
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "provider": self.llm_provider.get_provider_name(),
                "model": self.llm_provider.get_default_model()
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_memory_messages(self) -> List[Dict[str, str]]:
        """Get conversation history from memory
        
        Returns:
            List of message dictionaries
        """
        if not self.memory:
            return []
        
        messages = []
        if hasattr(self.memory, 'chat_memory'):
            for msg in self.memory.chat_memory.messages:
                messages.append({
                    "role": msg.type,
                    "content": msg.content
                })
        
        return messages
