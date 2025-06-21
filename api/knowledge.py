"""
Knowledge document upload API endpoints.
"""

import logging
import json
import uuid
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from setting.db import SessionLocal, db_manager
from sqlalchemy import or_, and_, func, case
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse


from api.models import (
    DocumentMetadata,
    ProcessedDocument,
    APIResponse,
    TopicSummary,
)
from knowledge_graph.models import (
    SourceData,
    GraphBuildStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def log_sql_query(query, query_name="SQL Query"):
    """
    Log the SQL statement for a SQLAlchemy query.

    Args:
        query: SQLAlchemy query object
        query_name: Optional name for the query for logging purposes
    """
    try:
        # Try to compile with literal binds to show actual values
        compiled_query = query.statement.compile(compile_kwargs={"literal_binds": True})
        logger.info(f"{query_name}: {compiled_query}")
    except Exception as e:
        # Fallback to basic string representation if compile fails
        logger.info(f"{query_name} (basic): {str(query)}")
        logger.debug(f"SQL compile error: {e}")


def _generate_temp_token_id(doc_link: str, external_database_uri: str = "") -> str:
    """
    Generate a deterministic temp_token_id based on doc_link and external_database_uri.

    Args:
        doc_link: The document link
        external_database_uri: The external database URI (empty string for local)

    Returns:
        SHA256 hash of the combined string
    """
    # Combine doc_link and external_database_uri for hash generation
    combined_string = f"{doc_link}||{external_database_uri}"

    # Generate SHA256 hash
    hash_object = hashlib.sha256(combined_string.encode("utf-8"))
    return hash_object.hexdigest()


# Configuration
UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".sql"}
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB


def _ensure_upload_dir() -> None:
    """Ensure upload directory exists."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file including filename and type.

    Args:
        file: The uploaded file to validate

    Raises:
        HTTPException: If file validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a filename"
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


def _validate_batch_file_size(files: List[UploadFile]) -> None:
    """
    Validate that the total size of all uploaded files does not exceed the limit.

    Args:
        files: List of uploaded files to validate

    Raises:
        HTTPException: If total file size exceeds the limit
    """
    total_size = 0

    for file in files:
        # Get file size
        file.file.seek(0, 2)  # Seek to end of file
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        total_size += file_size

    if total_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Total file size ({total_size // (1024*1024)}MB) exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
        )


def _save_file_and_metadata(
    file: UploadFile,
    metadata: DocumentMetadata,
    base_dir: Path,
    temp_token_id: str = None,
) -> None:
    """
    Helper function to save file and metadata to directory.

    Args:
        file: The uploaded file to save
        metadata: Document metadata
        base_dir: Directory to save files to
        temp_token_id: Optional temp token ID to save in metadata
    """
    # Create directory
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file
    file_path = base_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)

    # Save metadata as JSON
    metadata_file = base_dir / "document_metadata.json"
    metadata_dict = metadata.dict()
    metadata_dict["file_name"] = file.filename
    if temp_token_id:
        metadata_dict["temp_token_id"] = temp_token_id

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def _get_versioned_directory(base_dir: Path) -> Path:
    """
    Get a versioned directory path that doesn't exist yet.

    Args:
        base_dir: Base directory path

    Returns:
        Path to versioned directory
    """
    base_name = base_dir.name
    parent_dir = base_dir.parent
    counter = 1
    while True:
        versioned_dir = parent_dir / f"{base_name}_v{counter}"
        if not versioned_dir.exists():
            return versioned_dir
        counter += 1


def _save_uploaded_file_with_metadata(
    file: UploadFile, metadata: DocumentMetadata
) -> Tuple[Path, str]:
    try:
        _ensure_upload_dir()

        filename = file.filename
        base_name = Path(filename).stem
        base_dir = UPLOAD_DIR / metadata.topic_name / base_name

        # Generate temp_token_id based on doc_link and database_uri
        external_db_uri = (
            ""
            if db_manager.is_local_mode(metadata.database_uri)
            else metadata.database_uri
        )
        temp_token_id = _generate_temp_token_id(metadata.doc_link, external_db_uri)

        # Check if GraphBuildStatus already exists for this temp_token_id
        # This is sufficient since temp_token_id uniquely identifies doc_link + database_uri
        with SessionLocal() as db:
            existing_build_status = (
                db.query(GraphBuildStatus)
                .filter(
                    GraphBuildStatus.temp_token_id == temp_token_id,
                    GraphBuildStatus.topic_name == metadata.topic_name,
                    GraphBuildStatus.external_database_uri == external_db_uri,
                )
                .first()
            )

            if existing_build_status:
                # Check if directory and metadata already exist
                if not base_dir.exists():
                    # Create versioned directory if base doesn't exist
                    base_dir = _get_versioned_directory(base_dir)
                    _save_file_and_metadata(file, metadata, base_dir, temp_token_id)

                logger.info(
                    f"Found existing document with temp_token_id: {temp_token_id}, "
                    f"storage_directory: {existing_build_status.storage_directory}"
                )
                return base_dir, temp_token_id
            # If no existing source in database, check file system for existing metadata
            # Check all possible versioned directories sequentially
            base_name = base_dir.name
            parent_dir = base_dir.parent

            # Check base directory first, then versioned directories
            directories_to_check = [base_dir]
            counter = 1
            while True:
                versioned_dir = parent_dir / f"{base_name}_v{counter}"
                if versioned_dir.exists():
                    directories_to_check.append(versioned_dir)
                    counter += 1
                else:
                    break

            # Check each directory for matching metadata
            for check_dir in directories_to_check:
                metadata_file = check_dir / "document_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            existing_metadata = json.load(f)
                        if (
                            existing_metadata
                            and existing_metadata.get("doc_link") == metadata.doc_link
                            and existing_metadata.get("topic_name")
                            == metadata.topic_name
                            and existing_metadata.get("database_uri")
                            == metadata.database_uri
                        ):
                            # Found matching metadata in file system, check for existing temp_token_id
                            existing_temp_token = existing_metadata.get("temp_token_id")
                            if existing_temp_token:
                                logger.info(
                                    f"Found existing metadata file with matching temp_token_id: {existing_temp_token} in directory: {check_dir}"
                                )
                                return check_dir, existing_temp_token
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid metadata file found at {metadata_file}, skipping"
                        )

            # New source - create directory and save
            base_dir = _get_versioned_directory(base_dir)
            _save_file_and_metadata(file, metadata, base_dir, temp_token_id)
            logger.info(
                f"File and metadata saved successfully: {base_dir} with hash-based temp_token_id: {temp_token_id}"
            )
            return base_dir, temp_token_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )


def _create_processing_task(
    storage_directory: Path, metadata: DocumentMetadata, temp_token_id: str
) -> None:
    """
    Create a background processing task for uploaded document.

    This function creates a GraphBuildStatus record that will be picked up
    by the GraphBuildDaemon for asynchronous processing.

    Args:
        storage_directory: Path to directory containing file and metadata
        metadata: Document metadata
        temp_token_id: Pre-generated unique identifier for the document

    Raises:
        HTTPException: If task creation fails
    """
    try:
        # Create task record in local database only
        # All task scheduling is centralized in local database
        external_db_uri = (
            ""
            if db_manager.is_local_mode(metadata.database_uri)
            else metadata.database_uri
        )

        with SessionLocal() as db:
            build_status = GraphBuildStatus(
                topic_name=metadata.topic_name,
                temp_token_id=temp_token_id,
                external_database_uri=external_db_uri,
                storage_directory=str(storage_directory),
                status="uploaded",
            )
            db.add(build_status)
            db.commit()

        if db_manager.is_local_mode(metadata.database_uri):
            logger.info(
                f"Created local knowledge graph task: {temp_token_id} in {storage_directory}"
            )
        else:
            logger.info(
                f"Created external-db knowledge graph task: {temp_token_id} in {storage_directory}"
            )

        return

    except Exception as e:
        logger.error(f"Failed to create processing task for {storage_directory}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create processing task: {str(e)}",
        )


def _get_file_type(file_path: Path) -> str:
    """
    Determine file type from extension.

    Args:
        file_path: Path to the file

    Returns:
        File type string
    """
    extension = file_path.suffix.lower()
    type_mapping = {".pdf": "pdf", ".md": "markdown", ".txt": "document", ".sql": "sql"}
    return type_mapping.get(extension, "unknown")


@router.post("/upload", response_model=APIResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload"),
    links: List[str] = Form(
        ...,
        description="List of links to original documents. "
        "Recommended to use accessible links; if not available, "
        "you can use custom unique addresses. Must ensure uniqueness.",
    ),
    topic_name: str = Form(..., description="Topic name for knowledge graph building"),
    database_uri: Optional[str] = Form(
        None, description="Database connection string for storing the data"
    ),
) -> JSONResponse:
    """
    Upload documents for asynchronous knowledge graph building.

    This endpoint accepts multiple files with corresponding links and saves them to storage
    with metadata for background processing. Files are validated for type and total size,
    then saved to individual directories with metadata. Processing tasks are created for
    background execution by the GraphBuildDaemon. The total size of all uploaded files
    must not exceed 30MB.

    Duplicate Detection:
    - If a file with identical metadata already exists, returns 'already_exists' status
    - If a file exists with different metadata, creates a versioned directory
    - Use the /topics API to check processing status

    Args:
        files: List of files to upload (supports pdf, md, txt, sql)
        links: List of links to original documents (must match number of files)
        topic_name: Topic name for knowledge graph building
        database_uri: Database connection string (optional, uses local if not provided)

    Returns:
        JSON response with upload results and task creation status.
        Documents will have status 'uploaded' for new uploads or 'already_exists' for duplicates.
        Use trigger-processing API to change status from 'uploaded' to 'pending' when ready to process.

    Raises:
        HTTPException: If validation fails (including total file size exceeds limit) or upload errors occur
    """

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided"
        )

    if not links:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No links provided"
        )

    # Validate that files and links count match
    if len(files) != len(links):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Number of files ({len(files)}) must match number of links ({len(links)})",
        )

    # Validate link uniqueness
    if len(links) != len(set(links)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="All links must be unique"
        )

    # Validate total file size
    _validate_batch_file_size(files)

    processed_documents: List[ProcessedDocument] = []
    failed_uploads = []

    # Validate database connection if provided
    if database_uri and not db_manager.is_local_mode(database_uri):
        try:
            if not db_manager.validate_database_connection(database_uri):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid database connection string or database is not accessible",
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Database connection failed: {str(e)}",
            )

    # Process each file with its corresponding link individually
    for file, link in zip(files, links):
        try:
            # Validate file format and name
            _validate_file(file)

            # Create metadata for this specific file with its corresponding link
            file_metadata = DocumentMetadata(
                doc_link=link, topic_name=topic_name, database_uri=database_uri
            )

            # Save file with metadata and check for duplicates
            storage_directory, temp_token_id = _save_uploaded_file_with_metadata(
                file, file_metadata
            )

            # check whether GraphBuildStatus exists for this temp_token_id, topic_name, database_uri
            is_existing = False  # Initialize the variable
            external_db_uri = (
                "" if db_manager.is_local_mode(database_uri) else database_uri or ""
            )
            with SessionLocal() as db:
                build_status = (
                    db.query(GraphBuildStatus)
                    .filter(
                        GraphBuildStatus.temp_token_id == temp_token_id,
                        GraphBuildStatus.topic_name == topic_name,
                        GraphBuildStatus.external_database_uri == external_db_uri,
                    )
                    .first()
                )
                if build_status:
                    is_existing = True

            if is_existing:
                # File with same metadata already exists
                processed_doc = ProcessedDocument(
                    id=temp_token_id,
                    name=file.filename or "unknown",
                    file_path=str(storage_directory),
                    doc_link=link,
                    file_type=_get_file_type(Path(file.filename or "unknown")),
                    status="already_exists",
                )
                logger.info(
                    f"File with identical metadata already exists: {file.filename}"
                )
            else:
                _create_processing_task(storage_directory, file_metadata, temp_token_id)

                processed_doc = ProcessedDocument(
                    id=temp_token_id,
                    name=file.filename or "unknown",
                    file_path=str(storage_directory),
                    doc_link=link,
                    file_type=_get_file_type(Path(file.filename or "unknown")),
                    status="uploaded",
                )
                logger.info(
                    f"Created processing task for document: {file.filename} with ID: {temp_token_id}"
                )

            processed_documents.append(processed_doc)

        except HTTPException:
            # Re-raise HTTP exceptions (validation errors)
            raise
        except Exception as e:
            # Handle processing errors - continue with other files
            error_detail = {
                "file": file.filename or "unknown",
                "link": link,
                "reason": str(e),
            }
            failed_uploads.append(error_detail)
            logger.error(f"Failed to upload file {file.filename} with link {link}: {e}")

    # If all files failed, return error
    if not processed_documents and failed_uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "UPLOAD_FAILED",
                "message": "All files failed to process",
                "details": failed_uploads,
            },
        )

    # Prepare unified response with all results
    response_data = {
        "uploaded_count": len(processed_documents),
        "total_count": len(files),
        "documents": [doc.dict() for doc in processed_documents],
        "failed": failed_uploads,
        "success_rate": len(processed_documents) / len(files) if files else 0,
    }

    # Count different types of results
    uploaded_count = sum(1 for doc in processed_documents if doc.status == "uploaded")
    existing_count = sum(
        1 for doc in processed_documents if doc.status == "already_exists"
    )

    # Create appropriate message based on results
    if uploaded_count > 0 and existing_count > 0:
        message = f"Batch upload completed: {uploaded_count} documents uploaded successfully, {existing_count} already exist. Use trigger-processing API to start processing when ready."
    elif uploaded_count > 0:
        message = f"Batch upload completed: {uploaded_count} documents uploaded successfully. Use trigger-processing API to start processing when ready."
    elif existing_count > 0:
        message = f"Batch upload completed: {existing_count} documents already exist with same metadata. Check /topics API for current status."
    else:
        message = "Batch upload completed."

    response = APIResponse(
        status="success",
        message=message,
        data=response_data,
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


@router.post("/trigger-processing", response_model=APIResponse)
async def trigger_processing(
    topic_name: str = Form(
        ..., description="Name of the topic to trigger processing for"
    ),
    database_uri: Optional[str] = Form(
        None, description="Database URI to filter tasks (optional)"
    ),
) -> JSONResponse:
    """
    Trigger processing for uploaded documents in a topic.

    Changes status from 'uploaded' to 'pending' for all uploaded documents
    in the specified topic, which will be picked up by the GraphBuildDaemon.

    Args:
        topic_name: Name of the topic to trigger processing for
        database_uri: Database URI to filter tasks (optional)

    Returns:
        JSON response with the number of documents queued for processing

    Raises:
        HTTPException: If triggering fails
    """
    try:
        # Determine the external_database_uri for filtering
        external_db_uri = (
            "" if db_manager.is_local_mode(database_uri) else database_uri or ""
        )

        with SessionLocal() as db:
            # Find uploaded documents for this topic
            # Build the query
            query = db.query(GraphBuildStatus).filter(
                and_(
                    GraphBuildStatus.topic_name == topic_name,
                    GraphBuildStatus.status == "uploaded",
                    GraphBuildStatus.external_database_uri == external_db_uri,
                )
            )

            if not query.count():
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content=APIResponse(
                        status="success",
                        message=f"No uploaded documents found for topic '{topic_name}'",
                        data={"triggered_count": 0},
                    ).dict(),
                )

            # Update status to pending
            triggered_count = (
                db.query(GraphBuildStatus)
                .filter(
                    and_(
                        GraphBuildStatus.topic_name == topic_name,
                        GraphBuildStatus.status == "uploaded",
                        GraphBuildStatus.external_database_uri == external_db_uri,
                    )
                )
                .update(
                    {
                        "status": "pending",
                        "scheduled_at": func.current_timestamp(),
                        "updated_at": func.current_timestamp(),
                    },
                    synchronize_session=False,
                )
            )

            db.commit()

            logger.info(
                f"Triggered processing for {triggered_count} documents in topic '{topic_name}'"
            )

            response = APIResponse(
                status="success",
                message=f"Successfully triggered processing for {triggered_count} documents in topic '{topic_name}'. Processing will begin shortly.",
                data={"triggered_count": triggered_count, "topic_name": topic_name},
            )

            return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(f"Failed to trigger processing for topic '{topic_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger processing: {str(e)}",
        )


@router.get("/topics", response_model=APIResponse)
async def list_topics(
    database_uri: Optional[str] = None,
) -> JSONResponse:
    """
    List all topics with their status summary.

    This endpoint returns all topics and their processing status from the local database.
    All task scheduling information is centralized in the local database, including
    tasks from external databases (identified by external_database_uri field).

    Args:
        database_uri: Filter topics by database URI (optional, empty string for local,
                     specific URI for external database tasks)

    Returns:
        JSON response with list of topics and their status summaries

    Raises:
        HTTPException: If query errors occur
    """
    try:
        # Always query from local database since all task scheduling is centralized there
        with SessionLocal() as db:
            # Query topics with their status counts, filtered by database_uri if provided
            query = db.query(
                GraphBuildStatus.topic_name,
                GraphBuildStatus.external_database_uri,
                func.count(GraphBuildStatus.temp_token_id).label("total_documents"),
                func.sum(
                    case((GraphBuildStatus.status == "pending", 1), else_=0)
                ).label("pending_count"),
                func.sum(
                    case((GraphBuildStatus.status == "uploaded", 1), else_=0)
                ).label("uploaded_count"),
                func.sum(
                    case((GraphBuildStatus.status == "processing", 1), else_=0)
                ).label("processing_count"),
                func.sum(
                    case((GraphBuildStatus.status == "completed", 1), else_=0)
                ).label("completed_count"),
                func.sum(case((GraphBuildStatus.status == "failed", 1), else_=0)).label(
                    "failed_count"
                ),
                func.max(GraphBuildStatus.updated_at).label("latest_update"),
            )

            # Filter by database_uri if provided
            if database_uri is not None:
                query = query.filter(
                    GraphBuildStatus.external_database_uri == database_uri
                )

            topic_stats = query.group_by(
                GraphBuildStatus.topic_name, GraphBuildStatus.external_database_uri
            ).all()

            # Build topic summaries
            topic_summaries = []
            for stats in topic_stats:
                topic_summary = TopicSummary(
                    topic_name=stats.topic_name,
                    total_documents=stats.total_documents,
                    pending_count=stats.pending_count or 0,
                    uploaded_count=stats.uploaded_count or 0,
                    processing_count=stats.processing_count or 0,
                    completed_count=stats.completed_count or 0,
                    failed_count=stats.failed_count or 0,
                    latest_update=(
                        stats.latest_update.isoformat() if stats.latest_update else None
                    ),
                    database_uri=(
                        "local"
                        if db_manager.is_local_mode(stats.external_database_uri)
                        else "external"
                    ),
                )
                topic_summaries.append(topic_summary)

            # Sort by database_uri first, then topic name
            topic_summaries.sort(key=lambda x: (x.database_uri, x.topic_name))

            response_data = {
                "topics": [topic.dict() for topic in topic_summaries],
                "total_topics": len(topic_summaries),
                "source": "local_database",  # Always from local database
            }

            response = APIResponse(data=response_data)
            return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve topics",
        )
