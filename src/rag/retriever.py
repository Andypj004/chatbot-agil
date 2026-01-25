"""RAG retriever for combining vector search with LLM"""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.rag.vector_store import VectorStore
from src.llm.base import BaseLLMProvider
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()


class RAGRetriever:
    """Retrieval-Augmented Generation system"""
    
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: BaseLLMProvider,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        """Initialize RAG retriever
        
        Args:
            vector_store: VectorStore instance
            llm_provider: LLM provider instance
            prompt_template: Custom prompt template
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k or settings.top_k_results
        
        # Set up prompt
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info(f"RAG retriever initialized with top_k={self.top_k}")
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides default)
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        k = k or self.top_k
        logger.info(f"Retrieving documents for query: '{query}'")
        
        documents = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Retrieve relevant documents with relevance scores
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, score)
        """
        k = k or self.top_k
        logger.info(f"Retrieving documents with scores for query: '{query}'")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """Query the RAG system
        
        Args:
            question: Question to answer
            k: Number of documents to retrieve
            filter: Optional metadata filter
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing RAG query: '{question}'")
        
        # Retrieve relevant documents
        documents = self.retrieve_documents(query=question, k=k, filter=filter)
        
        if not documents:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [] if return_sources else None
            }
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Generate answer using LLM
        llm = self.llm_provider.get_llm()
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        logger.info(f"Generating answer using {self.llm_provider.get_provider_name()}")
        response = llm.predict(formatted_prompt)
        
        result = {
            "answer": response.strip(),
            "num_sources": len(documents)
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        
        logger.info("RAG query completed successfully")
        return result
    
    def create_qa_chain(self, chain_type: str = "stuff") -> RetrievalQA:
        """Create a LangChain RetrievalQA chain
        
        Args:
            chain_type: Type of chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
            
        Returns:
            RetrievalQA chain
        """
        logger.info(f"Creating QA chain with type: {chain_type}")
        
        retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_provider.get_llm(),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        return qa_chain
