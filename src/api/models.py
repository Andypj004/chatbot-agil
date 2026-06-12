"""Pydantic models for API requests and responses"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class APIBaseModel(BaseModel):
    """Base model for API schemas with relaxed protected namespace checks."""

    model_config = ConfigDict(protected_namespaces=())


class SessionAttachment(APIBaseModel):
    """Attachment metadata stored with a chat message."""

    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    source_url: Optional[str] = None


class QuestionnaireAnswer(APIBaseModel):
    """Single answer in the agile adoption questionnaire."""

    question_number: int = Field(..., ge=1, le=5)
    answer: Literal["a", "b", "c", "d"]


class UserRegistrationRequest(APIBaseModel):
    """Request model for registering a new user."""

    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=3, max_length=120)
    account_type: Literal["Profesor", "Estudiante"]
    knowledge_level: int = Field(..., ge=1, le=4)
    questionnaire_answers: List[QuestionnaireAnswer] = Field(
        default_factory=list,
        description="Optional agile adoption questionnaire answers",
    )


class UserLoginRequest(APIBaseModel):
    """Request model for logging in a user."""

    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class UserProfileResponse(APIBaseModel):
    """Response model for authenticated user profile data."""

    user_id: str
    email: str
    full_name: str
    account_type: str
    knowledge_level: int
    agile_adoption_level: int
    agile_adoption_label: str
    is_admin: bool = False
    created_at: str
    updated_at: str
    last_login_at: Optional[str] = None


class AuthResponse(APIBaseModel):
    """Response model for register/login endpoints."""

    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


class UserListResponse(APIBaseModel):
    """Response model for admin user listing."""

    total: int
    users: List[UserProfileResponse]


class ChatRequest(APIBaseModel):
    """Request model for chat endpoint"""

    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(
        default=None, description="Conversation session id"
    )
    use_rag: bool = Field(default=True, description="Use RAG for knowledge base search")
    stream: bool = Field(default=False, description="Stream response tokens")
    llm_provider: Optional[str] = Field(default=None, description="LLM provider to use")
    model_name: Optional[str] = Field(default=None, description="Model name to use")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    session_document_ids: List[str] = Field(
        default_factory=list,
        description="Optional session-scoped document IDs attached to this chat turn",
    )
    session_attachments: List[SessionAttachment] = Field(
        default_factory=list,
        description="Optional attachment metadata to persist with the chat turn",
    )


class CurrentUserInfo(APIBaseModel):
    """Compact authenticated user payload used internally by dependencies."""

    user_id: str
    email: str
    full_name: str
    account_type: str
    knowledge_level: int
    agile_adoption_level: int
    agile_adoption_label: str


class SourceCitation(APIBaseModel):
    """Source citation metadata for RAG responses."""

    document_id: Optional[str] = None
    filename: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    scope: Optional[str] = None
    session_id: Optional[str] = None
    relevance: Optional[float] = None
    excerpt: str = Field(default="", description="Short source excerpt")


class ChatResponse(APIBaseModel):
    """Response model for chat endpoint"""

    session_id: str = Field(..., description="Conversation session id")
    response: str = Field(..., description="Chatbot response")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    used_rag: bool = Field(..., description="Whether RAG was used")
    sources: Optional[List[SourceCitation]] = Field(
        default=None, description="Sources used for the response"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class ConversationMessage(APIBaseModel):
    """Single message in a stored conversation."""

    id: int
    session_id: str
    role: str
    text: str
    created_at: str
    provider: Optional[str] = None
    model: Optional[str] = None
    used_rag: Optional[bool] = None
    sources: List[SourceCitation] = Field(default_factory=list)
    attachments: List[SessionAttachment] = Field(default_factory=list)


class SessionSummary(APIBaseModel):
    """Conversation session summary."""

    session_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    created_at: str
    updated_at: str
    message_count: int
    last_message: str


class SessionTitleUpdateRequest(APIBaseModel):
    """Request model for updating a session title."""

    title: str = Field(..., min_length=1, max_length=80)


class SessionListResponse(APIBaseModel):
    """Response model for listing sessions."""

    total_sessions: int
    sessions: List[SessionSummary]


class SessionHistoryResponse(APIBaseModel):
    """Response model for session history."""

    session_id: str
    total_messages: int
    offset: int
    limit: int
    messages: List[ConversationMessage]


class DocumentUploadResponse(APIBaseModel):
    """Response model for document upload"""

    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    document_id: str = Field(..., description="Document ID in vector store")
    chunks_created: int = Field(..., description="Number of chunks created")
    scope: str = Field(..., description="Document scope: global_rag or session_chat")
    session_id: Optional[str] = Field(
        default=None, description="Owner session for session-scoped uploads"
    )


class DocumentInfo(APIBaseModel):
    """Information about a document"""

    document_id: Optional[str] = None
    filename: str
    file_type: str
    source: str
    file_hash: str
    scope: str = "global_rag"
    session_id: Optional[str] = None


class DocumentListResponse(APIBaseModel):
    """Response model for listing documents"""

    total_documents: int = Field(..., description="Total number of documents")
    documents: List[DocumentInfo] = Field(..., description="List of documents")


class ConfigUpdateRequest(APIBaseModel):
    """Request model for updating configuration"""

    llm_provider: Optional[str] = Field(
        default=None, description="Default LLM provider"
    )
    model_name: Optional[str] = Field(default=None, description="Default model name")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class ConfigResponse(APIBaseModel):
    """Response model for configuration"""

    llm_provider: str
    model_name: str
    temperature: float
    max_tokens: int
    available_providers: List[str]
    available_models: Dict[str, List[str]]
    rag_enabled: bool


class HealthResponse(APIBaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    llm_providers: List[str] = Field(..., description="Available LLM providers")
    rag_status: str = Field(..., description="RAG system status")
    vector_store_documents: int = Field(
        ..., description="Number of documents in vector store"
    )
