"""OpenAI LLM provider implementation"""

from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from src.llm.base import BaseLLMProvider
from src.core.logger import get_logger

logger = get_logger()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using GPT models"""
    
    def get_llm(self) -> BaseChatModel:
        """Get OpenAI chat model instance
        
        Returns:
            Configured ChatOpenAI instance
        """
        if self._llm is None:
            logger.info(f"Initializing OpenAI provider with model: {self.model_name}")
            self._llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
        return self._llm
    
    def get_provider_name(self) -> str:
        """Get provider name
        
        Returns:
            Provider name string
        """
        return "openai"
    
    def get_default_model(self) -> str:
        """Get default model for OpenAI
        
        Returns:
            Default model name
        """
        return "gpt-4-turbo-preview"
