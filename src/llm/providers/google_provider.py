"""Google (Gemini) LLM provider implementation"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models.base import BaseChatModel

from src.llm.base import BaseLLMProvider
from src.core.logger import get_logger

logger = get_logger()


class GoogleProvider(BaseLLMProvider):
    """Google LLM provider using Gemini models"""
    
    def get_llm(self) -> BaseChatModel:
        """Get Google Gemini chat model instance
        
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        if self._llm is None:
            logger.info(f"Initializing Google provider with model: {self.model_name}")
            self._llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **self.kwargs
            )
        return self._llm
    
    def get_provider_name(self) -> str:
        """Get provider name
        
        Returns:
            Provider name string
        """
        return "google"
    
    def get_default_model(self) -> str:
        """Get default model for Google
        
        Returns:
            Default model name
        """
        return "gemini-pro"
