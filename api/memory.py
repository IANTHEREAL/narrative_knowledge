"""
Memory API endpoints for personal chat history.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
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


class ChatMessage(BaseModel):
    """Chat message model."""

    message_content: str
    session_id: str
    conversation_title: str
    date: str = Field(..., description="ISO format datetime string")
    role: str = Field(..., description="user or assistant")


class MemoryStoreRequest(BaseModel):
    """Request model for storing memory."""

    chat_messages: List[ChatMessage]
    user_id: str
    force_reprocess: bool = Field(False, description="Force reprocessing existing data")


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


@router.post("/store", response_model=APIResponse)
async def store_memory(request: MemoryStoreRequest) -> JSONResponse:
    """
    Store batch chat messages as personal memory.

    ## Overview
    This endpoint processes chat message batches into a comprehensive personal memory system that includes:

    1. **Conversation Summaries** - Stored as structured knowledge blocks
    2. **Personal Analysis Blueprints** - Automatically evolving user profiles
    3. **User Insights** - Graph-based personal knowledge extraction

    ## Processing Pipeline

    The system automatically:
    - Analyzes conversation content to generate comprehensive summaries
    - Maintains personal analysis blueprints that evolve with user interactions
    - Extracts insights about user interests, knowledge domains, and behavioral patterns
    - Handles deduplication and conflict resolution for overlapping insights
    - Stores everything in a searchable vector database for future retrieval

    ## Input Format

    **Chat Messages** should include:
    - `message_content`: The actual message text
    - `session_id`: Unique identifier for the chat session
    - `conversation_title`: Title/topic of the conversation
    - `date`: ISO format timestamp
    - `role`: Either "user" or "assistant"

    ## Response Data

    Returns processing results including:
    - `source_id`: Database ID of the stored conversation data
    - `knowledge_block_id`: ID of the created summary knowledge block
    - `summary`: Generated conversation summary with facets and insights
    - `insights_generated`: Number of personal insights extracted
    - `blueprint_updated`: Timestamp of personal analysis blueprint update

    ## Example Request

    ```json
    {
        "chat_messages": [
            {
                "message_content": "How do I implement async/await in Python?",
                "session_id": "session_123",
                "conversation_title": "Learning Python Async Programming",
                "date": "2024-01-15T10:30:00Z",
                "role": "user"
            },
            {
                "message_content": "Here's how to use async/await in Python...",
                "session_id": "session_123",
                "conversation_title": "Learning Python Async Programming",
                "date": "2024-01-15T10:31:00Z",
                "role": "assistant"
            }
        ],
        "user_id": "user_456",
        "force_reprocess": false
    }
    ```

    ## Example Response

    ```json
    {
        "status": "success",
        "message": "Successfully processed 2 messages and generated 3 insights",
        "data": {
            "status": "success",
            "source_id": "source_789",
            "knowledge_block_id": "kb_101112",
            "summary": {
                "main_summary": "User learning Python async programming concepts",
                "user_queries": ["How to implement async/await"],
                "topics_discussed": ["async programming", "Python"],
                "facets": ["syntax", "best practices"],
                "user_interests": ["backend development"],
                "knowledge_domains": ["programming", "Python"],
                "key_outcomes": ["learned async/await syntax"]
            },
            "insights_generated": 3,
            "blueprint_updated": "2024-01-15T10:32:00Z"
        }
    }
    ```

    Args:
        request: Memory store request with chat messages and user info

    Returns:
        JSON response with storage results and insights generated

    Raises:
        HTTPException: If processing fails
    """
    try:
        memory_system = _get_memory_system()

        # Convert Pydantic models to dicts
        chat_messages = [msg.dict() for msg in request.chat_messages]

        # Process the chat batch
        result = memory_system.process_chat_batch(
            chat_messages=chat_messages, user_id=request.user_id
        )

        response = APIResponse(
            status="success",
            message=f"Successfully processed {len(chat_messages)} messages and generated {result['insights_generated']} insights",
            data=result,
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(
            f"Error storing memory for user {request.user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}",
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
