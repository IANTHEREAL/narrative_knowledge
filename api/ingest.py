import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

from fastapi import APIRouter, Request, HTTPException, status, UploadFile, Form
from fastapi.responses import JSONResponse

from api.models import APIResponse, DocumentMetadata, ProcessedDocument
from api.memory import _get_memory_system, register_memory_background_task
from api.memory import store_chat_batch, memory_background_processing
from api.knowledge import register_file_background_task, file_background_processing
from knowledge_graph.models import RawDataSource, BackgroundTask
from setting.db import SessionLocal, db_manager

import asyncio
import copy
import uuid
import hashlib

# Functions imported from api.knowledge for reuse
from api.knowledge import (
    _validate_file,
    _save_uploaded_file_with_metadata,
    _create_processing_task,
    _get_file_type,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])


# --- Processing Functions ---


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

    storage_directory, build_id = _save_uploaded_file_with_metadata(file, file_metadata)

    is_existing = False
    with SessionLocal() as db:
        build_status = (
            db.query(RawDataSource)
            .filter(
                RawDataSource.id == build_id,
                RawDataSource.topic_name == topic_name,
            )
            .first()
        )
        if build_status:
            is_existing = True

    if is_existing:
        status_msg = "already_exists"
        logger.info(
            f"File {file.filename} for topic {topic_name} already exists in the database."
        )
    else:
        _create_processing_task(file, storage_directory, file_metadata, build_id)
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
        message=f"Successfully processed file {file.filename} for knowledge graph. Status: {status_msg}",
        data=processed_doc.dict(),
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


async def _store_and_get_build_id(
    file: UploadFile, target_type: str, metadata_str: str, process_strategy: Dict[str, Any]
) -> tuple[Path, str, str, Dict[str, Any]]:
    """Store file and return storage directory and build_id without processing."""
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in metadata.",
        )
    
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

    storage_directory, build_id = _save_uploaded_file_with_metadata(file, file_metadata)

    is_existing = False
    with SessionLocal() as db:
        build_status = (
            db.query(RawDataSource)
            .filter(
                RawDataSource.build_id == build_id,
                RawDataSource.topic_name == topic_name,
            )
            .first()
        )
        if build_status:
            is_existing = True
            build_status.target_type = target_type
            build_status.status = "uploaded"
            build_status.raw_data_source_metadata = file_metadata.dict()
            build_status.process_strategy = process_strategy
            db.commit()

    if is_existing:
        status_msg = f"File {file.filename} already_exists for topic {topic_name}. Apply the latest uploaded information."
        logger.info(
            f"File {file.filename} for topic {topic_name} already exists in the database. Apply the latest uploaded information."
        )
    else:
        logger.info(f"File {file.filename} for topic {topic_name} is being uploaded.")
        _create_processing_task(file, storage_directory, file_metadata, build_id, target_type, process_strategy)
        status_msg = "uploaded"

    processed_doc = ProcessedDocument(
        id=build_id,
        name=file.filename or "unknown",
        file_path=str(storage_directory),
        doc_link=link,
        file_type=_get_file_type(Path(file.filename or "unknown")),
        status=status_msg,
    )
    processed_doc_dict = processed_doc.model_dump()
    return storage_directory, build_id, topic_name, processed_doc_dict


async def _process_json_for_personal_memory(
    input_data: Any, metadata: Dict[str, Any], process_strategy: Dict[str, Any]
) -> JSONResponse:
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

    memory_system = _get_memory_system()
    result = memory_system.process_chat_batch(chat_messages=input_data, user_id=user_id)

    response = APIResponse(
        status="success",
        message=f"Successfully processed {len(input_data)} messages for personal memory",
        data=result,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


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
        return await _process_file_for_knowledge_graph(
            file, metadata, process_strategy or {}
        )
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


@router.post("/save_old", response_model=APIResponse, status_code=status.HTTP_200_OK)
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
        # Parse process_strategy from string to dict if provided
        parsed_process_strategy = None
        if process_strategy:
            try:
                parsed_process_strategy = json.loads(process_strategy)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON in process_strategy.",
                )
        return await _handle_form_data(
            file, metadata, target_type, parsed_process_strategy
        )
    elif "application/json" in content_type:
        return await _handle_json_data(request)
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported content type. Use application/json or multipart/form-data.",
        )


from tools.route_wrapper import ToolsRouteWrapper

# Create wrapper instance at module level
tools_wrapper = ToolsRouteWrapper()


@router.post("/save", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def save_data_pipeline(
    request: Request,
    files: Optional[List[UploadFile]] = Form(None),
    links: Optional[str] = Form(None),
    target_type: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    process_strategy: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Enhanced save endpoint using tools pipeline system.
    """
    content_type = request.headers.get("content-type", "")
    base_url = str(request.base_url)
    if "multipart/form-data" in content_type:
        # Validate required parameters
        if not files or len(files) == 0 or not target_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For multipart/form-data, 'files' and 'target_type' are required.",
            )
        
        # Parse metadata if provided
        try:
            parsed_metadata = json.loads(metadata) if metadata is not None else {}
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'metadata' format. Must be a valid JSON object. Error: {str(e)}",
            )
        
        # Validate links format - either from 'links' or metadata.links, or default
        meta_links = parsed_metadata.get("links")
        all_links = links or meta_links
        if all_links:
            try:
                if isinstance(all_links, str):
                    links_list = json.loads(all_links)
                else:
                    links_list = all_links
                if not isinstance(links_list, list):
                    raise ValueError("'links' is not a JSON list.")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid 'links' format. Must be a JSON list of strings. Error: {str(e)}",
                )

            if len(links_list) != len(files):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"'links' and 'files' count mismatch: {len(links_list)} links vs {len(files)} files.",
                )
        else:
            links_list = [f"file://links/{file.filename}" for file in files]
        
        
        # Store files and get build IDs using existing functions
        build_ids = []
        storage_dirs = []
        processed_docs = []

        # Register background task for entire batch
        topic_name = parsed_metadata.get("topic_name", "")

        
        for file, link in zip(files, links_list):
            single_metadata = copy.deepcopy(parsed_metadata)
            single_metadata["link"] = link
            single_metadata["content_type"] = file.content_type
            
            storage_dir, build_id, topic_name, processed_doc = await _store_and_get_build_id(
                file=file,
                target_type=target_type,
                metadata_str=json.dumps(single_metadata),
                process_strategy=json.loads(process_strategy) if process_strategy else {}
            )
            
            build_ids.append(build_id)
            storage_dirs.append(str(storage_dir))
            processed_docs.append(processed_doc)
            
        # register_file_background_task(
        #     task_id, 
        #     task_id,
        #     topic_name, 
        #     len(files)
        # )

        # Generate task ID for background processing tracking
        task_id = hashlib.sha256(topic_name.encode("utf-8")).hexdigest()

        source_id = build_ids[0][:36] if build_ids else task_id
        
        # Start background processing without waiting
        # asyncio.create_task(
        #     file_background_processing(
        #         files_data=files,
        #         metadata=parsed_metadata,
        #         links_list=links_list,
        #         process_strategy=process_strategy,
        #         target_type=target_type,
        #         task_id=task_id,
        #         source_id=source_id,
        #     )
        # )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Files uploaded successfully. Background processing has started for {len(files)} files.",
                "data": {
                    "files_uploaded": len(files),
                    "task_id": task_id,
                    "build_ids": build_ids,
                    "topic_name": topic_name,
                    "processed_docs": processed_docs
                },
                "retrieval": f"{base_url}api/v1/tasks/{task_id}",
            },
        )

    elif "application/json" in content_type:
        # JSON data processing
        body = await request.json()
        target_type = body.get("target_type", "personal_memory")
        
        if target_type == "personal_memory":
            # Store chat batch and return "uploaded" feedback
            chat_messages = body.get("input", [])
            metadata = body.get("metadata", {})
            user_id = metadata.get("user_id", "")
            process_strategy = body.get("process_strategy")
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="`user_id` is required in metadata for target_type 'personal_memory'.",
                )
            
            if not isinstance(chat_messages, list):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input data for personal memory must be a list of chat messages.",
                )
            
            # Store chat batch
            store_result = await store_chat_batch(chat_messages, user_id, process_strategy, metadata)
            source_id = store_result["source_id"]
            topic_name = store_result["topic_name"]
            # Generate task ID for background processing tracking
            task_id = hashlib.sha256(topic_name.encode("utf-8")).hexdigest()
            
            # Register the task
            # register_memory_background_task(task_id, source_id, user_id, topic_name, len(chat_messages))
            
            # Start background processing without waiting
            # asyncio.create_task(
            #     memory_background_processing(
            #         chat_messages=chat_messages,
            #         user_id=user_id,
            #         source_id=source_id,
            #         topic_name=topic_name,
            #         process_strategy=process_strategy,
            #         target_type=target_type,
            #         task_id=task_id,
            #     )
            # )
            
            # Return confirmation with task ID
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Chat batch uploaded successfully. Background processing has started.",
                    "data": {
                        "status": "uploaded",
                        "source_id": source_id,
                        "user_id": user_id,
                        "message_count": len(chat_messages),
                        "topic_name": store_result["topic_name"],
                        "phase": "stored",
                        "task_id": task_id
                    },
                    "retrieval": f"{base_url}api/v1/tasks/{task_id}",
                },
            )
        
        else:
            # For non-personal_memory target types, use original processing
            result = tools_wrapper.process_json_request(
                input_data=body.get("input"),
                metadata=body.get("metadata", {}),
                process_strategy=body.get("process_strategy", {}),
                target_type=target_type or "personal_memory",
            )

            # Convert ToolResult to dict for JSON serialization
            result_dict = result.to_dict()

            if result.success:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Processing completed successfully",
                        "data": result_dict["data"],
                        "execution_id": result_dict["execution_id"],
                    },
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": result_dict["error_message"],
                        "execution_id": result_dict["execution_id"],
                    },
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported content type. Use application/json or multipart/form-data.",
        )


@router.get("/tasks/{task_id}")
async def get_background_task_status(task_id: str) -> JSONResponse:
    """
    Get status of background processing task.
    
    Args:
        task_id: Background task ID
        
    Returns:
        JSON response with task status and results
    """
    SessionLocal = db_manager.get_session_factory()
    
    with SessionLocal() as db:
        task = db.query(BackgroundTask).filter(BackgroundTask.task_id == task_id).order_by(BackgroundTask.created_at.desc()).first()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Return all tasks with the same task_id from database
        return JSONResponse(
            status_code=status.HTTP_200_OK, 
            content={
                "status": "success",
                "message": f"Found task with ID {task_id}",
                "data": task.to_dict() 
            }
        )