"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, documents, config, health
from src.core.config import settings
from src.core.logger import get_logger
from src import __version__

logger = get_logger()

# Create FastAPI application
app = FastAPI(
    title="Agile Chatbot API",
    description="A production-ready chatbot with multi-LLM support, RAG, and online search capabilities",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("=" * 60)
    logger.info("Starting Agile Chatbot API")
    logger.info(f"Version: {__version__}")
    logger.info(f"Default LLM Provider: {settings.default_llm_provider}")
    logger.info(f"Default Model: {settings.default_model}")
    logger.info(f"RAG Enabled: Yes")
    logger.info(f"Search Enabled: {settings.has_search_capability()}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("Shutting down Agile Chatbot API")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Agile Chatbot API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
