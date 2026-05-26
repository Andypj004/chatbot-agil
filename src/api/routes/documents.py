"""Document management endpoints"""

from typing import List, Optional
import hashlib
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from pathlib import Path

from src.api.models import DocumentUploadResponse, DocumentListResponse, DocumentInfo
from src.api.dependencies import (
    get_vector_store,
    get_document_processor,
    get_session_manager,
    get_current_user_optional,
)
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor
from src.memory.session_manager import SessionManager
from src.utils.file_utils import save_uploaded_file
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])

GLOBAL_SCOPE = "global_rag"
SESSION_SCOPE = "session_chat"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _build_document_id(file_path: str, scope: str, session_id: Optional[str]) -> str:
    stem = Path(file_path).stem
    owner = session_id or "global"
    return f"{scope}:{owner}:{stem}"


def _calculate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _list_uploaded_files_fallback(
    uploads_dir: Path,
    scope: str,
    session_id: Optional[str] = None,
) -> List[DocumentInfo]:
    """Fallback metadata list from persisted uploaded files."""
    if not uploads_dir.exists():
        return []

    items: List[DocumentInfo] = []
    for file_path in sorted(uploads_dir.glob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue

        file_type = file_path.suffix.lower().lstrip(".") or "unknown"
        items.append(
            DocumentInfo(
                document_id=f"{scope}:{session_id or 'global'}:{file_path.stem}",
                filename=file_path.name,
                file_type=file_type,
                source=str(file_path),
                file_hash=file_path.stem,
                scope=scope,
                session_id=session_id,
            )
        )

    return items


def _clear_uploaded_files(uploads_dir: Path) -> None:
    """Remove uploaded files from disk, keeping hidden sentinel files."""
    if not uploads_dir.exists():
        return

    for file_path in uploads_dir.glob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        file_path.unlink(missing_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    vector_store: VectorStore = Depends(get_vector_store),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
):
    """Upload a document to the knowledge base
    
    Supported formats: PDF, TXT, DOCX, DOC, MD, PNG, JPG, JPEG, WEBP
    
    The document will be:
    1. Saved to disk
    2. Processed and chunked
    3. Added to the vector database for RAG
    
    Args:
        file: Document file
        vector_store: Vector store instance
        doc_processor: Document processor instance
        
    Returns:
        Upload confirmation with document ID
    """
    logger.info(f"Received document upload: {file.filename}")
    
    try:
        extension = Path(file.filename or "").suffix.lower()
        if extension in IMAGE_EXTENSIONS:
            raise ValueError(
                "Las imagenes no se indexan en RAG global sin OCR. "
                "Subelas como adjuntos de sesion para analisis multimodal del modelo."
            )

        # Save uploaded file
        file_path = await save_uploaded_file(file, destination_dir="data/uploads/global")
        
        # Process document
        logger.info(f"Processing document: {file_path}")
        chunks = doc_processor.process_file(file_path, scope=GLOBAL_SCOPE, session_id=None)
        
        # Generate document IDs
        base_id = _build_document_id(file_path, scope=GLOBAL_SCOPE, session_id=None)
        doc_ids = [f"{base_id}:{i}" for i in range(len(chunks))]
        
        # Add to vector store
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        vector_store.add_documents(chunks, ids=doc_ids)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            document_id=base_id,
            chunks_created=len(chunks),
            scope=GLOBAL_SCOPE,
            session_id=None,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post(
    "/sessions/{session_id}/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a session-scoped document",
)
async def upload_session_document(
    session_id: str,
    file: UploadFile = File(..., description="Session document file to upload"),
    vector_store: VectorStore = Depends(get_vector_store),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    session_manager: SessionManager = Depends(get_session_manager),
    current_user=Depends(get_current_user_optional),
):
    """Upload a document restricted to one chat session."""
    logger.info(f"Received session document upload for session={session_id}: {file.filename}")

    try:
        owner_id = current_user.get("user_id") if current_user else None
        existing_session = session_manager.get_session_record(session_id)
        if existing_session is not None and existing_session.get("user_id") not in (None, owner_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        session_manager.create_session(session_id, user_id=owner_id)
        destination = f"data/uploads/sessions/{session_id}"
        file_path = await save_uploaded_file(file, destination_dir=destination)
        base_id = _build_document_id(file_path, scope=SESSION_SCOPE, session_id=session_id)
        extension = Path(file_path).suffix.lower()
        file_hash = _calculate_file_hash(file_path)
        file_type = extension.lstrip(".") or "unknown"

        chunks_created = 0
        if extension not in IMAGE_EXTENSIONS:
            chunks = doc_processor.process_file(file_path, scope=SESSION_SCOPE, session_id=session_id)
            doc_ids = [f"{base_id}:{i}" for i in range(len(chunks))]
            vector_store.add_documents(chunks, ids=doc_ids)
            chunks_created = len(chunks)

        session_manager.add_session_document(
            session_id=session_id,
            document_id=base_id,
            filename=file.filename or Path(file_path).name,
            source=file_path,
            file_type=file_type or "unknown",
            file_hash=file_hash,
        )

        return DocumentUploadResponse(
            message="Session document uploaded and processed successfully",
            filename=file.filename,
            document_id=base_id,
            chunks_created=chunks_created,
            scope=SESSION_SCOPE,
            session_id=session_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error uploading session document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}",
        )


@router.get("", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """List all documents in the knowledge base
    
    Returns:
        List of documents with metadata
    """
    logger.info("Listing documents")
    
    try:
        # Metadata list deduplicated at file-level for global scope.
        documents = [
            DocumentInfo(**item)
            for item in vector_store.list_indexed_documents(metadata_filter={"scope": GLOBAL_SCOPE})
        ]
        if not documents:
            documents = _list_uploaded_files_fallback(Path("data/uploads/global"), scope=GLOBAL_SCOPE)

        return DocumentListResponse(
            total_documents=len(documents),
            documents=documents,
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=DocumentListResponse, summary="List session documents")
async def list_session_documents(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    current_user=Depends(get_current_user_optional),
):
    """List session-scoped documents attached to one chat session."""
    logger.info(f"Listing session documents for session={session_id}")
    try:
        owner_id = current_user.get("user_id") if current_user else None
        session = session_manager.get_session_record(session_id)
        if session is None or session.get("user_id") not in (None, owner_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")

        documents = [DocumentInfo(**item) for item in session_manager.list_session_documents(session_id)]
        if not documents:
            documents = _list_uploaded_files_fallback(
                Path(f"data/uploads/sessions/{session_id}"),
                scope=SESSION_SCOPE,
                session_id=session_id,
            )

        return DocumentListResponse(total_documents=len(documents), documents=documents)
    except Exception as e:
        logger.error(f"Error listing session documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing session documents: {str(e)}",
        )


@router.delete("/{document_id}", summary="Delete a document")
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a document from the knowledge base
    
    Args:
        document_id: ID of the document to delete
        vector_store: Vector store instance
        
    Returns:
        Deletion confirmation
    """
    logger.info(f"Deleting document: {document_id}")
    
    try:
        success = vector_store.delete_by_metadata({"scope": GLOBAL_SCOPE, "file_hash": document_id})
        if not success:
            success = vector_store.delete_documents([document_id])
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.delete("/sessions/{session_id}/{document_id}", summary="Delete one session document")
async def delete_session_document(
    session_id: str,
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
    session_manager: SessionManager = Depends(get_session_manager),
    current_user=Depends(get_current_user_optional),
):
    """Delete a session-scoped document by document_id."""
    logger.info(f"Deleting session document: session={session_id}, document={document_id}")

    try:
        owner_id = current_user.get("user_id") if current_user else None
        session = session_manager.get_session_record(session_id)
        if session is None or session.get("user_id") not in (None, owner_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in session {session_id}",
            )

        session_docs = session_manager.list_session_documents(session_id)
        match = next((item for item in session_docs if item.get("document_id") == document_id), None)
        if match is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in session {session_id}",
            )

        file_hash = match.get("file_hash")
        if file_hash:
            vector_store.delete_by_metadata(
                {"scope": SESSION_SCOPE, "session_id": session_id, "file_hash": file_hash}
            )

        source = match.get("source")
        if source:
            Path(source).unlink(missing_ok=True)

        session_manager.remove_session_document(session_id=session_id, document_id=document_id)
        return {"message": f"Session document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session document: {str(e)}",
        )


@router.delete("/sessions/{session_id}", summary="Clear all session documents")
async def clear_session_documents(
    session_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
    session_manager: SessionManager = Depends(get_session_manager),
    current_user=Depends(get_current_user_optional),
):
    """Clear all session-scoped documents for one chat session."""
    logger.warning(f"Clearing session documents: session={session_id}")
    try:
        owner_id = current_user.get("user_id") if current_user else None
        session = session_manager.get_session_record(session_id)
        if session is None or session.get("user_id") not in (None, owner_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")

        vector_store.delete_by_metadata({"scope": SESSION_SCOPE, "session_id": session_id})

        session_docs = session_manager.list_session_documents(session_id)
        for item in session_docs:
            source = item.get("source")
            if source:
                Path(source).unlink(missing_ok=True)
        removed_count = session_manager.clear_session_documents(session_id)

        _clear_uploaded_files(Path(f"data/uploads/sessions/{session_id}"))
        return {"message": "Session documents cleared successfully", "removed": removed_count}
    except Exception as e:
        logger.error(f"Error clearing session documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session documents: {str(e)}",
        )


@router.delete("", summary="Clear all documents")
async def clear_documents(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Clear all documents from the knowledge base
    
    ⚠️ Warning: This action cannot be undone!
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        Confirmation message
    """
    logger.warning("Clearing all documents")
    
    try:
        success = vector_store.delete_by_metadata({"scope": GLOBAL_SCOPE})
        _clear_uploaded_files(Path("data/uploads/global"))
        
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear documents"
            )
            
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing documents: {str(e)}"
        )
