"""
API data models for request/response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents with support for custom fields."""

    doc_link: str = Field(
        ...,
        description="Link to original document. Recommended to use accessible link; "
        "if not available, you can use custom unique address. Must ensure uniqueness.",
    )
    topic_name: str = Field(..., description="Topic name for knowledge graph building")
    database_uri: Optional[str] = Field(
        None, description="Database connection string for storing the data"
    )
    custom_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom metadata fields that users can define. Any additional key-value pairs for document context.",
    )


class ProcessedDocument(BaseModel):
    """Information about a processed document."""

    id: str
    name: str
    file_path: str
    doc_link: Optional[str] = None
    file_type: str
    status: str = "processed"


class DocumentInfo(BaseModel):
    """Document information."""

    id: str
    name: str
    doc_link: Optional[str] = None
    file_type: str
    content_preview: str
    created_at: str
    updated_at: str
    build_statuses: List[Dict[str, Any]] = []
    graph_elements: Dict[str, List[Dict[str, str]]] = {
        "entities": [],
        "relationships": [],
    }


class TopicSummary(BaseModel):
    """Summary information for a topic."""

    topic_name: str
    total_documents: int
    uploaded_count: int
    pending_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    latest_update: Optional[str] = None
    database_uri: str = Field(
        default="", description="External database URI, empty for local"
    )


class APIResponse(BaseModel):
    """Standard API response."""

    status: str = "success"
    data: Any
    message: Optional[str] = None
