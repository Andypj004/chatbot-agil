"""Legacy entrypoint wrapper for the active FastAPI application."""

from src.main import app


if __name__ == "__main__":
    import uvicorn

    from src.core.config import settings
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
