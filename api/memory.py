"""
Memory API endpoints for personal chat history.
"""
import json
import hashlib
import uuid
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memory_system import PersonalMemorySystem
from api.models import APIResponse
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding
from setting.db import db_manager, SessionLocal
from setting.base import LLM_MODEL, LLM_PROVIDER
from knowledge_graph.models import RawDataSource, SourceData, ContentStore, BackgroundTask
from tools.route_wrapper import ToolsRouteWrapper


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


def generate_topic_name_for_personal_memory(user_id: str) -> str:
    """Generate a topic name for the user."""
    return f"The personal information of {user_id}"


async def store_chat_batch(
    chat_messages: List[Dict[str, Any]],
    user_id: str,
    process_strategy: Optional[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Store chat batch as SourceData without processing - Phase 1.
    
    Args:
        chat_messages: List of chat messages
        user_id: User identifier
        
    Returns:
        Dict with storage confirmation including source_id
    """
    topic_name = generate_topic_name_for_personal_memory(user_id)
    
    logger.info(
        f"Storing chat batch for user '{user_id}' with topic name: '{topic_name}'; {len(chat_messages)} messages"
    )
    
    # Get the latest timestamp for unique identifier
    if chat_messages and any(msg.get("date") for msg in chat_messages):
        latest_timestamp_str = max(
            msg["date"] for msg in chat_messages if msg.get("date")
        )
        batch_timestamp = latest_timestamp_str
    else:
        batch_timestamp = datetime.now().isoformat()
    
    chat_link = f"memory://{user_id}/chat_batch/{batch_timestamp}"
    
    # Prepare attributes
    attributes = {
        "user_id": user_id,
        "topic_name": topic_name,
        "batch_type": "chat_messages",
        "message_count": len(chat_messages),
        "session_id": metadata.get("session_id", "") if chat_messages else "",
        "conversation_title": chat_messages[0].get("conversation_title", "") if chat_messages else "",
        "last_message_date": batch_timestamp,
        "processing_status": "pending",  # Mark as pending for processing
    }
    
    # Serialize content data
    content_json = json.dumps(chat_messages, ensure_ascii=False, indent=2)
    content_hash = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
    
    SessionLocal = db_manager.get_session_factory()
    
    with SessionLocal() as db:
        # Check if source data already exists by content hash
        existing_source = (
            db.query(SourceData)
            .filter(SourceData.link == chat_link)
            .first()
        )
        
        if existing_source:
            logger.info(
                f"Chat batch already exists with link: {chat_link}, reusing: {existing_source.id}"
            )
            return {
                "status": "already_exists",
                "source_id": str(existing_source.id),
                "message": "Chat batch already stored",
                "topic_name": topic_name,
                "user_id": user_id,
                "message_count": len(chat_messages),
            }
        
        # Check if content already exists
        content_store = (
            db.query(ContentStore).filter_by(content_hash=content_hash).first()
        )
        
        if not content_store:
            content_store = ContentStore(
                content_hash=content_hash,
                content=content_json,
                content_size=len(content_json.encode("utf-8")),
                content_type="application/json",
                name=f"chat_batch_{user_id}_{batch_timestamp}",
                link=chat_link,
            )
            db.add(content_store)
            logger.info(
                    f"Created new content store entry with hash: {content_hash[:8]}..."
                )
        else:
            logger.info(
                f"Reusing existing content store entry with hash: {content_hash[:8]}..."
            )
        
        # Create source data & raw data source
        source_data = SourceData(
            name=f"chat_batch_{user_id}_{batch_timestamp}",
            link=chat_link,
            topic_name=topic_name,
            source_type="application/json",
            content_hash=content_store.content_hash,
            attributes=attributes,
        )
        
        db.add(source_data)
        db.commit()
        db.refresh(source_data)

        raw_data_source = RawDataSource(
            topic_name=topic_name,
            target_type= "personal_memory",
            process_strategy=process_strategy,
            build_id=source_data.id,
            file_path=chat_link,
            file_hash=content_hash,
            original_filename=f"chat_batch_{user_id}_{batch_timestamp}",
            raw_data_source_metadata={
                "chat_messages": chat_messages,
                "metadata": metadata,
            },
            status="uploaded",
        )
        
        db.add(raw_data_source)
        db.commit()
        db.refresh(raw_data_source)
        db.refresh(content_store)
        
        logger.info(f"Stored chat batch as SourceData: {source_data.id} and RawDataSource: {raw_data_source.id}")
        
        return {
            "status": "uploaded",
            "source_id": str(source_data.id),
            "message": "Chat batch stored successfully. Processing will begin shortly.",
            "topic_name": topic_name,
            "user_id": user_id,
            "message_count": len(chat_messages),
        }


# Create wrapper instance at module level
tools_wrapper = ToolsRouteWrapper()

# Use database-based task tracking for persistence across workers


def register_memory_background_task(task_id: str, source_id: str, user_id: str, topic_name:str, message_count: int) -> None:
    """Register a new background task immediately."""
    with SessionLocal() as db:
        task = BackgroundTask(
            id=task_id,
            task_type="memory_processing",
            source_id=source_id,
            user_id=user_id,
            topic_name=topic_name,
            message_count=message_count,
            status="processing"
        )
        db.add(task)
        db.commit()

async def memory_background_processing(
    chat_messages: List[Dict[str, Any]],
    user_id: str,
    source_id: str,
    topic_name: str,
    process_strategy: Optional[Dict[str, Any]] = None,
    target_type: str = "personal_memory",
    task_id: Optional[str] = None
) -> None:
    """
    Trigger background processing for chat batch after upload.
    
    Args:
        chat_messages: List of chat messages to process
        user_id: User identifier
        source_id: SourceData ID for existing data
        topic_name: Topic name for memory categorization
        process_strategy: Optional processing configuration
        target_type: Target processing type
        task_id: Optional task ID for tracking
    """
    task_id = task_id or str(uuid.uuid4())
    SessionLocal = db_manager.get_session_factory()
    
    try:
        # Prepare metadata with source_id for tools processing
        processing_metadata = {
            "user_id": user_id,
            "source_id": source_id,
            "topic_name": topic_name
        }
        
        # Process the existing source data
        result = tools_wrapper.process_json_request(
            input_data=chat_messages,
            metadata=processing_metadata,
            process_strategy=process_strategy,
            target_type=target_type,
        )
        
        # Update task status with success
        with SessionLocal() as db:
            task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
            if task:
                task.status = "completed"
                task.result = result.to_dict() if hasattr(result, 'to_dict') else result
                db.commit()
        
        logger.info(f"Background processing completed: {result.to_dict()}")
        
    except Exception as e:
        # Update task status with error
        with SessionLocal() as db:
            task = db.query(BackgroundTask).filter(BackgroundTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = str(e)
                db.commit()
        logger.error(f"Background processing failed: {e}")



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
            memory_types=request.memory_types or ["conversation", "insights"],
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
