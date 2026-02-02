"""Chat endpoint for interacting with the chatbot"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models import ChatRequest, ChatResponse
from src.api.dependencies import get_llm_provider, get_chatbot_agent
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat(request: ChatRequest):
    """Send a message to the chatbot and get a response
    
    The chatbot can use:
    - RAG (Retrieval-Augmented Generation) for knowledge base queries
    - Online search for real-time information
    - Multiple LLM providers (OpenAI, Claude, Gemini, Deepseek)
    
    Args:
        request: Chat request with message and options
        
    Returns:
        Chat response with answer and metadata
    """
    logger.info(f"Received chat request: {request.message[:100]}...")
    
    try:
        # Get LLM provider
        llm_provider = get_llm_provider(
            provider_name=request.llm_provider,
            model_name=request.model_name,
            temperature=request.temperature
        )
        
        # Get chatbot agent
        agent = get_chatbot_agent(
            llm_provider=llm_provider,
            use_rag=request.use_rag,
            use_search=request.use_online_search
        )
        
        # Process message
        result = agent.chat(
            message=request.message,
            use_rag=request.use_rag,
            use_search=request.use_online_search
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your message: {str(e)}"
        )
