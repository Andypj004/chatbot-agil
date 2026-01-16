"""
Global configuration settings for the chatbot application.
Loads environment variables and defines system parameters.
"""

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"
    VECTOR_STORE_PATH: Path = DATA_DIR / "vector_store"
    LOG_DIR: Path = BASE_DIR / "logs"

    # LLM Configuration
    OPENAI_API_KEY: str = "your-api-key-here"
    MODEL_NAME: str = "gpt-4-turbo-preview"
    MODEL_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000

    # Vector Store Configuration (RF1)
    CHROMA_PERSIST_DIRECTORY: str = "./data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME: str = "agile_knowledge_base"

    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # API Configuration (RNF2)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    DEBUG_MODE: bool = True

    # Security (RNF5)
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate Limiting (RNF3)
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_CONCURRENT_REQUESTS: int = 10

    # Session Management (RF4)
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_CONVERSATION_HISTORY: int = 20

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/chatbot.log"

    # Evaluation Metrics (RNF1)
    HALLUCINATION_THRESHOLD: float = 0.85
    MIN_ACCURACY_TARGET: float = 0.95

    # Frontend Configuration (RNF4)
    FRONTEND_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    # Performance (RNF3)
    MAX_RESPONSE_TIME_SECONDS: int = 8
    REQUEST_TIMEOUT_SECONDS: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
