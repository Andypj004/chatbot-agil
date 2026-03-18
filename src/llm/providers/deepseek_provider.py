"""Deepseek LLM provider implementation"""

from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from src.llm.base import BaseLLMProvider
from src.core.logger import get_logger

logger = get_logger()


class DeepseekProvider(BaseLLMProvider):
    """Deepseek LLM provider using OpenAI-compatible API"""
    
    def get_llm(self) -> BaseChatModel:
        """Get Deepseek chat model instance
        
        Deepseek uses an OpenAI-compatible API, so we use ChatOpenAI
        with a custom base URL.
        
        Returns:
            Configured ChatOpenAI instance for Deepseek
        """
        if self._llm is None:
            logger.info(f"Initializing Deepseek provider with model: {self.model_name}")
            self._llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                base_url="https://api.deepseek.com/v1",
                **self.kwargs
            )
        return self._llm
    
    def get_provider_name(self) -> str:
        """Get provider name
        
        Returns:
            Provider name string
        """
        return "deepseek"
    
    def get_default_model(self) -> str:
        """Get default model for Deepseek
        
        Returns:
            Default model name
        """
        return "deepseek-chat"
