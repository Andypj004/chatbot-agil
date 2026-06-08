"""Main FastAPI application"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import (
    get_llm_provider,
    get_vector_store,
    get_document_processor,
)
from src.api.routes import (
    admin,
    auth,
    chat,
    documents,
    config,
    health,
    sessions,
    forms,
    debug,
)
from src.core.config import settings
from src.core.logger import get_logger
from src import __version__

logger = get_logger()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_FILE = FRONTEND_DIR / "templates" / "index.html"
UPLOADS_DIR = PROJECT_DIR / "data" / "uploads"


def _reindex_global_uploads(vector_store) -> int:
    """Rebuild the active vector collection from persisted global uploads when empty."""
    uploads_dir = UPLOADS_DIR / "global"
    if not uploads_dir.exists():
        return 0

    doc_processor = get_document_processor()
    supported_extensions = {".pdf", ".txt", ".docx", ".doc", ".md", ".markdown"}
    indexed_files: List[Path] = []
    total_chunks = 0

    for file_path in sorted(uploads_dir.glob("*")):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            chunks = doc_processor.process_file(
                file_path=str(file_path), scope="global_rag", session_id=None
            )
            if not chunks:
                continue

            base_id = f"global_rag:global:{file_path.stem}"
            doc_ids = [f"{base_id}:{index}" for index in range(len(chunks))]
            vector_store.add_documents(chunks, ids=doc_ids)
            indexed_files.append(file_path)
            total_chunks += len(chunks)
        except Exception as exc:
            logger.warning(f"Skipping reindex for {file_path.name}: {exc}")

    if indexed_files:
        logger.info(
            f"Reindexed {len(indexed_files)} global files into {vector_store.collection_name} "
            f"({total_chunks} chunks)"
        )

    return total_chunks


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
            count = vector_store.get_collection_count()
            if count == 0:
                count = _reindex_global_uploads(vector_store)
            logger.info(f"Vector store warmup complete with {count} indexed documents")
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
app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(forms.router, prefix="/api/v1")
app.include_router(debug.router, prefix="/api/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Agile Chatbot API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
        "frontend": "/app",
    }


@app.get("/app", tags=["frontend"])
async def frontend_app():
    """Serve the web frontend application"""
    if not INDEX_FILE.exists():
        return {
            "message": "Frontend not found",
            "hint": "Ensure src/frontend/templates/index.html exists",
        }
    return FileResponse(INDEX_FILE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
