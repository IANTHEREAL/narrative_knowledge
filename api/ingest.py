import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Awaitable, List
import tempfile
import shutil
import os
import uuid

from fastapi import APIRouter, Request, HTTPException, status, UploadFile, Form
from fastapi.responses import JSONResponse

from api.models import APIResponse, DocumentMetadata, ProcessedDocument
from api.memory import _get_memory_system
from knowledge_graph.models import GraphBuild
from setting.db import SessionLocal, db_manager
from tools.api_integration import PipelineAPIIntegration

# Functions imported from api.knowledge for reuse
from api.knowledge import (
    _validate_file,
    _save_uploaded_file_with_metadata,
    _create_processing_task,
    _get_file_type,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])


# --- Tool-based Processing Functions ---

api_integration = PipelineAPIIntegration()


async def _process_file_with_pipeline(
    file: UploadFile, metadata: Dict[str, Any], process_strategy: Dict[str, Any]) -> JSONResponse:
    """Process an uploaded file using the new tool-based pipeline."""
    _validate_file(file)

    topic_name = metadata.get("topic_name")
    link = metadata.get("link")
    database_uri = metadata.get("database_uri")
    custom_metadata = {
        k: v
        for k, v in metadata.items()
        if k not in ["topic_name", "link", "database_uri"]
    }

    if not topic_name or not link:
        raise HTTPException(
            status_code=400,
            detail="`topic_name` and `link` are required in metadata for target_type 'knowledge_graph'.",
        )

    # Create temporary storage for the file
    temp_dir = tempfile.mkdtemp(prefix="knowledge_ingest_")
    try:
        file_path = Path(temp_dir) / (file.filename or "uploaded_file")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Prepare request data for pipeline
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {
                "topic_name": topic_name,
                "link": link,
                "database_uri": database_uri,
                **custom_metadata
            },
            "process_strategy": process_strategy or {}
        }

        # Prepare file information
        files = [{
            "path": str(file_path),
            "filename": file.filename or "unknown",
            "metadata": custom_metadata
        }]

        # Execute pipeline
        result = api_integration.process_request(request_data, files)

        # Create response
        processed_doc = ProcessedDocument(
            id=result.execution_id or str(uuid.uuid4()),
            name=file.filename or "unknown",
            file_path=str(file_path),
            doc_link=link,
            file_type=_get_file_type(Path(file.filename or "unknown")),
            status="processing" if result.success else "failed",
        )

        response = APIResponse(
            status="success" if result.success else "error",
            message=result.message or "Processing completed",
            data=processed_doc.model_dump(),
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK if result.success else status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump()
        )

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")




async def _process_file_for_knowledge_graph(
    file: UploadFile, metadata: Dict[str, Any], process_strategy: Dict[str, Any]
) -> JSONResponse:
    """Processes an uploaded file to build a knowledge graph."""
    _validate_file(file)

    topic_name = metadata.get("topic_name")
    link = metadata.get("link")
    database_uri = metadata.get("database_uri")
    custom_metadata = {
        k: v
        for k, v in metadata.items()
        if k not in ["topic_name", "link", "database_uri"]
    }

    if not topic_name or not link:
        raise HTTPException(
            status_code=400,
            detail="`topic_name` and `link` are required in metadata for target_type 'knowledge_graph'.",
        )

    file_metadata = DocumentMetadata(
        doc_link=link,
        topic_name=topic_name,
        database_uri=database_uri,
        custom_metadata=custom_metadata,
    )

    storage_directory, build_id = _save_uploaded_file_with_metadata(
        file, file_metadata
    )

    is_existing = False
    external_db_uri = (
        "" if db_manager.is_local_mode(database_uri) else database_uri or ""
    )
    with SessionLocal() as db:
        build_status = (
            db.query(GraphBuild)
            .filter(
                GraphBuild.build_id == build_id,
                GraphBuild.topic_name == topic_name,
                GraphBuild.external_database_uri == external_db_uri,
            )
            .first()
        )
        if build_status:
            is_existing = True

    if is_existing:
        status_msg = "already_exists"
    else:
        _create_processing_task(storage_directory, file_metadata, build_id)
        status_msg = "uploaded"

    processed_doc = ProcessedDocument(
        id=build_id,
        name=file.filename or "unknown",
        file_path=str(storage_directory),
        doc_link=link,
        file_type=_get_file_type(Path(file.filename or "unknown")),
        status=status_msg,
    )

    response = APIResponse(
        status="success",
        message=f"Successfully processed file for knowledge graph. Status: {status_msg}",
        data=processed_doc.dict(),
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


async def _process_json_for_personal_memory(
    input_data: Any, metadata: Dict[str, Any], process_strategy: Dict[str, Any]) -> JSONResponse:
    """Processes JSON input (chat history) to build personal memory."""
    user_id = metadata.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="`user_id` is required in metadata for target_type 'personal_memory'.",
        )

    if not isinstance(input_data, list):
        raise HTTPException(
            status_code=400,
            detail="Input data for personal memory must be a list of chat messages.",
        )

    # Use tool-based pipeline for memory processing instead of standalone system
    from tools.api_integration import PipelineAPIIntegration
    api_integration = PipelineAPIIntegration()
    
    # Prepare request for memory pipeline - unified flow
    request_data = {
        "target_type": "personal_memory",
        "metadata": {
            "user_id": user_id
        },
        "input": input_data
        # Let orchestrator select appropriate pipeline automatically
    }
    
    result = api_integration.process_request(request_data, [])

    response = APIResponse(
        status="success",
        message=f"Successfully processed {len(input_data)} messages for personal memory",
        data=result,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


async def _process_json_for_knowledge_graph(
    input_data: Any, metadata: Dict[str, Any], process_strategy: Dict[str, Any]) -> JSONResponse:
    """Process JSON input for knowledge graph using tool-based pipeline."""
    topic_name = metadata.get("topic_name")
    if not topic_name:
        raise HTTPException(
            status_code=400,
            detail="`topic_name` is required in metadata for target_type 'knowledge_graph'.",
        )

    # Handle different input formats
    if isinstance(input_data, str):
        # Text content
        content = input_data
    elif isinstance(input_data, dict):
        # JSON content
        content = json.dumps(input_data)
    else:
        content = str(input_data)

    # Create temporary file for processing
    temp_dir = tempfile.mkdtemp(prefix="knowledge_json_")
    try:
        file_path = Path(temp_dir) / "input.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Prepare request data for pipeline
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": metadata,
            "process_strategy": process_strategy or {}
        }

        # Prepare file information
        files = [{
            "path": str(file_path),
            "filename": "input.json",
            "metadata": {}
        }]

        # Execute pipeline
        result = api_integration.process_request(request_data, files)

        # Create response
        processed_doc = ProcessedDocument(
            id=result.execution_id or str(uuid.uuid4()),
            name="input.json",
            file_path=str(file_path),
            doc_link=metadata.get("link", "inline_input"),
            file_type="json",
            status="processing" if result.success else "failed",
        )

        response = APIResponse(
            status="success" if result.success else "error",
            message=result.message or "Processing completed",
            data=processed_doc.model_dump(),
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK if result.success else status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump()
        )

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


# --- Request Handlers ---


async def _handle_form_data(
    file: UploadFile,
    metadata_str: str,
    target_type: str,
    process_strategy: Optional[dict] = None,
) -> JSONResponse:
    """Handles multipart/form-data for file uploads."""
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in metadata.",
        )

    # Dispatch based on target_type for form-data
    if target_type == "knowledge_graph":
        # Use new tool-based pipeline system
        return await _process_file_with_pipeline(file, metadata, process_strategy)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target_type '{target_type}' for file uploads.",
        )


async def _handle_json_data(request: Request) -> JSONResponse:
    """Handles application/json for raw data, like chat history."""
    try:
        data = await request.json()
        input_data = data.get("input")
        metadata: Dict[str, Any] = data.get("metadata", {})
        target_type = data.get("target_type")
        process_strategy: Dict[str, Any] = data.get("process_strategy", {})

        if not isinstance(process_strategy, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="process_strategy must be a JSON object.",
            )

        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Input data is missing."
            )
        if not target_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`target_type` is a required field for JSON input.",
            )

        # Dispatch based on target_type for JSON
        if target_type == "personal_memory":
            return await _process_json_for_personal_memory(
                input_data, metadata, process_strategy
            )
        elif target_type == "knowledge_graph":
            return await _process_json_for_knowledge_graph(
                input_data, metadata, process_strategy
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target_type '{target_type}' for JSON input.",
            )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload.",
        )
    except Exception as e:
        logger.error(f"Failed to process json data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process JSON input: {str(e)}",
        )


@router.post("/save", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def save_data(
    request: Request,
    file: Optional[UploadFile] = Form(None),
    metadata: Optional[str] = Form(None),
    target_type: Optional[str] = Form(None),
    process_strategy: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Unified endpoint to save data from file uploads or raw JSON input.

    This endpoint supports both `multipart/form-data` for files and `application/json` for structured data.
    The processing behavior is determined by `target_type`.

    **For File Uploads (`multipart/form-data`):**
    - `file`: The file to be uploaded.
    - `metadata`: A JSON string with fields like `topic_name`, `link`, etc.
    - `target_type`: The desired output (e.g., "knowledge_graph").
    - `process_strategy` (optional): A JSON string representing a dictionary for processing options.

    **For JSON Input (`application/json`):**
    - `input`: The raw data (e.g., list of chat messages).
    - `metadata`: A JSON object with relevant context (e.g., `user_id`).
    - `target_type`: The desired output (e.g., "personal_memory").
    - `input_type`: A hint about the input format (e.g., "chat_history").
    - `process_strategy` (optional): A JSON object for processing options.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        if not file or not metadata or not target_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For multipart/form-data, 'file', 'metadata', and 'target_type' are required.",
            )
        return await _handle_form_data(file, metadata, target_type, process_strategy)
    elif "application/json" in content_type:
        return await _handle_json_data(request)
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported content type. Use application/json or multipart/form-data.",
        )
