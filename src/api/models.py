"""Pydantic models for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message", min_length=1)
    use_rag: bool = Field(default=True, description="Use RAG for knowledge base search")
    use_online_search: bool = Field(default=False, description="Allow online search")
    llm_provider: Optional[str] = Field(default=None, description="LLM provider to use")
    model_name: Optional[str] = Field(default=None, description="Model name to use")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot response")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    used_rag: bool = Field(..., description="Whether RAG was used")
    used_search: bool = Field(..., description="Whether online search was used")
    error: Optional[str] = Field(default=None, description="Error message if any")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    document_id: str = Field(..., description="Document ID in vector store")
    chunks_created: int = Field(..., description="Number of chunks created")


class DocumentInfo(BaseModel):
    """Information about a document"""
    filename: str
    file_type: str
    source: str
    file_hash: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    total_documents: int = Field(..., description="Total number of documents")
    documents: List[DocumentInfo] = Field(..., description="List of documents")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration"""
    llm_provider: Optional[str] = Field(default=None, description="Default LLM provider")
    model_name: Optional[str] = Field(default=None, description="Default model name")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class ConfigResponse(BaseModel):
    """Response model for configuration"""
    llm_provider: str
    model_name: str
    temperature: float
    max_tokens: int
    available_providers: List[str]
    rag_enabled: bool
    search_enabled: bool


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    llm_providers: List[str] = Field(..., description="Available LLM providers")
    rag_status: str = Field(..., description="RAG system status")
    search_status: str = Field(..., description="Search system status")
    vector_store_documents: int = Field(..., description="Number of documents in vector store")
