"""Application configuration management using Pydantic settings"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    ollama_api_key: Optional[str] = Field(default=None, alias="OLLAMA_API_KEY")
    
    # Application Settings
    default_llm_provider: str = Field(default="openai", alias="DEFAULT_LLM_PROVIDER")
    default_model: str = Field(default="gpt-4-turbo-preview", alias="DEFAULT_MODEL")
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_tokens: int = Field(default=2000, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="CORS_ORIGINS"
    )
    
    # RAG Configuration
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=300, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=8, alias="TOP_K_RESULTS")
    rag_context_max_chars: int = Field(default=10000, alias="RAG_CONTEXT_MAX_CHARS")
    rag_max_chunks_per_file: int = Field(default=5, alias="RAG_MAX_CHUNKS_PER_FILE")

    # Conversation history / memory
    conversation_db_path: str = Field(
        default="./data/conversations.db",
        alias="CONVERSATION_DB_PATH"
    )
    conversation_context_messages: int = Field(
        default=12,
        alias="CONVERSATION_CONTEXT_MESSAGES"
    )
    conversation_list_limit: int = Field(
        default=50,
        alias="CONVERSATION_LIST_LIMIT"
    )

    # Startup / lifecycle
    warmup_vector_store_on_startup: bool = Field(
        default=True,
        alias="WARMUP_VECTOR_STORE_ON_STARTUP"
    )
    warmup_default_provider_on_startup: bool = Field(
        default=False,
        alias="WARMUP_DEFAULT_PROVIDER_ON_STARTUP"
    )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific LLM provider"""
        key_mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "claude": self.anthropic_api_key,
            "google": self.google_api_key,
            "gemini": self.google_api_key,
            "deepseek": self.deepseek_api_key,
            # Ollama is usually local and does not require an API key.
            "ollama": self.ollama_api_key or "ollama-local",
        }
        return key_mapping.get(provider.lower())


# Global settings instance
settings = Settings()
