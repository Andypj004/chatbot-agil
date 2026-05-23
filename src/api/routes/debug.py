"""Debug routes for inspecting RAG retrieval results"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends

from src.api.models import APIBaseModel
from src.api.dependencies import get_rag_retriever
from src.rag.retriever import RAGRetriever
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/debug", tags=["debug"])


class DebugRAGRequest(APIBaseModel):
    query: str
    session_id: Optional[str] = None
    k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None


class DebugRAGItem(APIBaseModel):
    score: float
    excerpt: str
    metadata: Dict[str, Any]


class DebugRAGResponse(APIBaseModel):
    total_documents: int
    retrieved: List[DebugRAGItem]


class DebugCombinedRAGRequest(APIBaseModel):
    query: str
    session_id: Optional[str] = None
    k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None


@router.post("/rag/combined", response_model=DebugRAGResponse, summary="Debug: retrieve combined RAG documents")
def debug_rag_combined(
    request: DebugCombinedRAGRequest,
    retriever: RAGRetriever = Depends(get_rag_retriever),
):
    """Return the documents selected by the combined RAG retriever flow.

    This is closer to the actual chat path than the raw similarity endpoint.
    """
    logger.info(f"Debug combined RAG query: {request.query[:80]}")
    k = request.k or retriever.top_k or 10

    documents = retriever.retrieve_documents(
        query=request.query,
        k=k,
        filter=request.filter,
        session_id=request.session_id,
    )

    items = []
    for doc in documents:
        excerpt = (doc.page_content or "")[:800]
        items.append(DebugRAGItem(score=0.0, excerpt=excerpt, metadata=doc.metadata or {}))

    total = retriever.vector_store.get_collection_count()
    return DebugRAGResponse(total_documents=total, retrieved=items)


@router.post("/rag", response_model=DebugRAGResponse, summary="Debug: retrieve RAG documents with scores")
def debug_rag(
    request: DebugRAGRequest,
    retriever: RAGRetriever = Depends(get_rag_retriever),
):
    """Return documents retrieved by the RAG retriever with relevance scores.

    Useful to debug which documents are considered for a query and their metadata.
    """
    logger.info(f"Debug RAG query: {request.query[:80]}")

    # Determine k
    k = request.k or retriever.top_k or 10

    try:
        results = retriever.retrieve_with_scores(query=request.query, k=k, filter=request.filter)
    except Exception as e:
        logger.error(f"Error in retrieve_with_scores: {e}")
        results = []

    items = []
    for doc, score in results:
        excerpt = (doc.page_content or "")[:800]
        items.append(DebugRAGItem(score=score, excerpt=excerpt, metadata=doc.metadata or {}))

    total = retriever.vector_store.get_collection_count()

    return DebugRAGResponse(total_documents=total, retrieved=items)
