"""Anthropic (Claude) LLM provider implementation"""

from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel

from src.llm.base import BaseLLMProvider
from src.core.logger import get_logger

logger = get_logger()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider using Claude models"""
    
    def get_llm(self) -> BaseChatModel:
        """Get Anthropic chat model instance
        
        Returns:
            Configured ChatAnthropic instance
        """
        if self._llm is None:
            logger.info(f"Initializing Anthropic provider with model: {self.model_name}")
            self._llm = ChatAnthropic(
                anthropic_api_key=self.api_key,
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
        return "anthropic"
    
    def get_default_model(self) -> str:
        """Get default model for Anthropic
        
        Returns:
            Default model name
        """
        return "claude-3-opus-20240229"
