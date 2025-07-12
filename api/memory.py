"""
Memory API endpoints for personal chat history.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memory_system import PersonalMemorySystem
from api.models import APIResponse
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding
from setting.db import db_manager
from setting.base import LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


class MemoryRetrieveRequest(BaseModel):
    """Request model for retrieving memory."""

    query: str
    user_id: str
    memory_types: Optional[List[str]] = Field(
        ["conversation", "insights"], description="Types of memory to search"
    )
    time_range: Optional[Dict[str, str]] = Field(
        None, description="Time range filter with 'start' and 'end' keys"
    )
    top_k: int = Field(10, description="Number of results to return")


def _get_memory_system() -> PersonalMemorySystem:
    """Get initialized PersonalMemorySystem instance."""
    llm_client = LLMInterface(LLM_PROVIDER, LLM_MODEL)
    return PersonalMemorySystem(
        llm_client=llm_client,
        embedding_func=get_text_embedding,
        session_factory=db_manager.get_session_factory(),
    )


@router.post("/retrieve", response_model=APIResponse)
async def retrieve_memory(request: MemoryRetrieveRequest) -> JSONResponse:
    """
    Retrieve user memory based on semantic query.

    ## Overview
    This endpoint provides semantic search across a user's personal memory, including:

    1. **Conversation Summaries** - Previous chat interactions and their insights
    2. **User Insights** - Personal knowledge, interests, and behavioral patterns

    ## Search Capabilities

    **Vector Similarity Search**: Uses embeddings for semantic matching beyond keyword search
    **Time Range Filtering**: Filter results by creation date range
    **Memory Type Filtering**: Choose between conversations, insights, or both
    **User Isolation**: Each user's memory is completely private and isolated

    ## Memory Types

    - **`conversation`**: Searchable summaries of past chat interactions
      - Includes conversation topics, user queries, and key outcomes
      - Searchable by content, topics discussed, and temporal context
      - Useful for "What did we discuss about X?" type queries

    - **`insights`**: Personal knowledge and behavioral patterns
      - User interests, expertise domains, and learning patterns
      - Goals, aspirations, and personal development tracking
      - Communication preferences and problem-solving approaches
      - Useful for "What are my interests in X?" type queries

    ## Query Examples

    - `"Python programming"` - Find conversations and insights about Python
    - `"machine learning projects"` - Discover ML-related discussions and interests
    - `"career goals"` - Search for career-related insights and conversations
    - `"learning patterns"` - Find information about how the user learns

    ## Time Range Filtering

    ```json
    {
        "time_range": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-31T23:59:59Z"
        }
    }
    ```

    ## Example Request

    ```json
    {
        "query": "Python async programming",
        "user_id": "user_456",
        "memory_types": ["conversation", "insights"],
        "time_range": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-31T23:59:59Z"
        },
        "top_k": 5
    }
    ```

    ## Example Response

    ```json
    {
        "status": "success",
        "message": "Found 3 memory items for query: Python async programming",
        "data": {
            "query": "Python async programming",
            "user_id": "user_456",
            "results": {
                "conversations": [
                    {
                        "id": "kb_101112",
                        "name": "Chat Summary - user_456 - 2024-01-15",
                        "content": "Conversation about async programming...",
                        "created_at": "2024-01-15T10:32:00Z",
                        "attributes": {
                            "domains": ["programming", "Python"],
                            "facets": ["async/await", "concurrency"]
                        }
                    }
                ],
                "insights": [
                    {
                        "id": "insight_789",
                        "name": "Python Programming Interest",
                        "description": "Strong interest in Python async programming",
                        "created_at": "2024-01-15T10:32:00Z",
                        "attributes": {
                            "confidence_level": "high",
                            "insight_category": "technical_interest"
                        }
                    }
                ]
            },
            "total_found": 2
        }
    }
    ```

    Args:
        request: Memory retrieval request with query and filters

    Returns:
        JSON response with retrieved memory results

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        memory_system = _get_memory_system()

        # Retrieve memory without exposing topic_name concept
        results = memory_system.retrieve_user_memory(
            query=request.query,
            user_id=request.user_id,
            memory_types=request.memory_types,
            time_range=request.time_range,
            top_k=request.top_k,
        )

        response = APIResponse(
            status="success",
            message=f"Found {results['total_found']} memory items for query: {request.query}",
            data=results,
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(
            f"Error retrieving memory for user {request.user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}",
        )
