"""Document management endpoints"""

from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from pathlib import Path

from src.api.models import DocumentUploadResponse, DocumentListResponse, DocumentInfo
from src.api.dependencies import get_vector_store, get_document_processor
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor
from src.utils.file_utils import save_uploaded_file
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    vector_store: VectorStore = Depends(get_vector_store),
    doc_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Upload a document to the knowledge base
    
    Supported formats: PDF, TXT, DOCX, DOC, MD
    
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
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        
        # Process document
        logger.info(f"Processing document: {file_path}")
        chunks = doc_processor.process_file(file_path)
        
        # Generate document IDs
        doc_ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        
        # Add to vector store
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        vector_store.add_documents(chunks, ids=doc_ids)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            document_id=Path(file_path).stem,
            chunks_created=len(chunks)
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
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
        # Get collection count
        total_docs = vector_store.get_collection_count()
        
        # For now, return summary info
        # In production, you might want to maintain a separate metadata store
        return DocumentListResponse(
            total_documents=total_docs,
            documents=[]
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
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
        # In a production system, you'd track document IDs properly
        # For now, this is a placeholder
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
        success = vector_store.clear_collection()
        
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
