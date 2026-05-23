"""FastAPI application routes"""

from src.api.routes import chat, documents, config, health, forms, sessions

__all__ = ["chat", "documents", "config", "health", "forms", "sessions"]
