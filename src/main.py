"""Main FastAPI application"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import get_llm_provider, get_vector_store
from src.api.routes import chat, documents, config, health, sessions
from src.core.config import settings
from src.core.logger import get_logger
from src import __version__

logger = get_logger()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_FILE = FRONTEND_DIR / "templates" / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("Starting Agile Chatbot API")
    logger.info(f"Version: {__version__}")
    logger.info(f"Default LLM Provider: {settings.default_llm_provider}")
    logger.info(f"Default Model: {settings.default_model}")
    logger.info("RAG Enabled: Yes")

    if settings.warmup_vector_store_on_startup:
        try:
            vector_store = get_vector_store()
            logger.info(
                f"Vector store warmup complete with "
                f"{vector_store.get_collection_count()} indexed documents"
            )
        except Exception as exc:
            logger.warning(f"Vector store warmup skipped: {exc}")

    if settings.warmup_default_provider_on_startup:
        try:
            provider = get_llm_provider()
            provider.get_llm()
            logger.info(
                f"Default provider warmup complete: "
                f"{provider.get_provider_name()}/{provider.model_name}"
            )
        except Exception as exc:
            logger.warning(f"Default provider warmup skipped: {exc}")

    logger.info("=" * 60)

    try:
        yield
    finally:
        logger.info("Shutting down Agile Chatbot API")

# Create FastAPI application
app = FastAPI(
    title="Agile Chatbot API",
    description="A production-ready chatbot with multi-LLM support and RAG capabilities",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Agile Chatbot API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
        "frontend": "/app"
    }


@app.get("/app", tags=["frontend"])
async def frontend_app():
    """Serve the web frontend application"""
    if not INDEX_FILE.exists():
        return {
            "message": "Frontend not found",
            "hint": "Ensure src/frontend/templates/index.html exists"
        }
    return FileResponse(INDEX_FILE)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
