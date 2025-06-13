"""
Knowledge document upload API endpoints.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from api.models import (
    DocumentMetadata,
    ProcessedDocument,
    DocumentInfo,
    APIResponse,
    TopicSummary,
)
from knowledge_graph.knowledge import KnowledgeBuilder
from knowledge_graph.models import (
    GraphBuildStatus,
)
from typing import List
from setting.db import SessionLocal, db_manager
from sqlalchemy import or_, and_, func

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

# Configuration
UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".sql"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _ensure_upload_dir() -> None:
    """Ensure upload directory exists."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file including filename, type, and size.

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

    # Check file size
    file.file.seek(0, 2)  # Seek to end of file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
        )


def _save_uploaded_file(file: UploadFile, topic_name: str) -> Path:
    """
    Save uploaded file to disk in its own directory.
    Structure: UPLOAD_DIR/filename/filename

    Args:
        file: The uploaded file to save

    Returns:
        Path to the saved file

    Raises:
        HTTPException: If file saving fails
    """
    try:
        _ensure_upload_dir()

        # Create directory structure: UPLOAD_DIR/filename
        filename = file.filename
        file_dir = UPLOAD_DIR / topic_name / filename
        if file_dir.exists():
            return file_dir / filename

        # Create the file directory
        file_dir.mkdir(parents=True, exist_ok=True)

        # Save file inside its directory
        file_path = file_dir / filename

        # Write file content
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)

        logger.info(f"File saved successfully: {file_path}")
        return file_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )


def _process_document(file_path: Path, metadata: DocumentMetadata) -> ProcessedDocument:
    """
    Process uploaded document using knowledge builder and create build status record.

    Args:
        file_path: Path to the uploaded file
        metadata: Document metadata including doc_link, topic_name, and database_uri

    Returns:
        ProcessedDocument with extraction results

    Raises:
        Exception: If document processing fails
    """
    try:
        # Get appropriate session factory for the target database
        target_session_factory = db_manager.get_session_factory(metadata.database_uri)

        # Prepare attributes for knowledge extraction
        attributes = {"doc_link": metadata.doc_link, "topic_name": metadata.topic_name}

        # Extract knowledge using knowledge builder with target database
        kb_builder_instance = KnowledgeBuilder(session_factory=target_session_factory)
        result = kb_builder_instance.extract_knowledge(str(file_path), attributes)

        if result["status"] != "success":
            raise Exception(
                f"Knowledge extraction failed: {result.get('error', 'Unknown error')}"
            )

        # Handle build status record creation based on mode
        if db_manager.is_local_mode(metadata.database_uri):
            # Local mode: create record in local database only
            kb_builder_instance.create_build_status_record(
                result["source_id"], metadata.topic_name
            )
        else:
            # Multi-database mode: create record in user database and sync to local
            # 1. Create record in user database
            kb_builder_instance.create_build_status_record(
                result["source_id"],
                metadata.topic_name,
            )

            # 2. Create sync record in local database for task scheduling
            # TODO: sync with external database
            local_kb_builder = KnowledgeBuilder(session_factory=SessionLocal)
            local_kb_builder.create_build_status_record(
                result["source_id"],
                metadata.topic_name,
                external_database_uri=metadata.database_uri,
            )

        return ProcessedDocument(
            id=result["source_id"],
            name=result["source_name"],
            file_path=str(file_path),
            doc_link=metadata.doc_link,
            file_type=_get_file_type(file_path),
            status="processed",
        )

    except Exception as e:
        logger.error(f"Failed to process document {file_path}: {e}")
        raise


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
    Upload and process documents for knowledge graph building.

    This endpoint accepts multiple files with corresponding links and processes them
    through the knowledge extraction pipeline. Each file is validated, saved, and processed
    individually to extract knowledge content. A build status record is created for each document.

    Args:
        files: List of files to upload (supports pdf, md, txt, sql)
        links: List of links to original documents (must match number of files)
        topic_name: Topic name for knowledge graph building
        database_uri: Database connection string (optional, uses local if not provided)

    Returns:
        JSON response with batch upload results including all processed document information

    Raises:
        HTTPException: If validation fails or processing errors occur
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

    processed_documents: List[ProcessedDocument] = []
    failed_uploads = []

    # Validate database connection if provided
    if database_uri:
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
            # Validate file
            _validate_file(file)

            # Save file
            file_path = _save_uploaded_file(file, topic_name)

            # Create metadata for this specific file with its corresponding link
            file_metadata = DocumentMetadata(
                doc_link=link, topic_name=topic_name, database_uri=database_uri
            )

            # Process document individually
            processed_doc = _process_document(file_path, file_metadata)
            processed_documents.append(processed_doc)

            logger.info(
                f"Successfully processed document: {file.filename} with link: {link}"
            )

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
            logger.error(
                f"Failed to process file {file.filename} with link {link}: {e}"
            )

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

    response = APIResponse(
        status="success",
        message=f"Batch upload completed: {len(processed_documents)}/{len(files)} documents processed successfully",
        data=response_data,
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


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
                func.count(GraphBuildStatus.source_id).label("total_documents"),
                func.sum(
                    func.case([(GraphBuildStatus.status == "pending", 1)], else_=0)
                ).label("pending_count"),
                func.sum(
                    func.case([(GraphBuildStatus.status == "processing", 1)], else_=0)
                ).label("processing_count"),
                func.sum(
                    func.case([(GraphBuildStatus.status == "completed", 1)], else_=0)
                ).label("completed_count"),
                func.sum(
                    func.case([(GraphBuildStatus.status == "failed", 1)], else_=0)
                ).label("failed_count"),
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
                    processing_count=stats.processing_count or 0,
                    completed_count=stats.completed_count or 0,
                    failed_count=stats.failed_count or 0,
                    latest_update=(
                        stats.latest_update.isoformat() if stats.latest_update else None
                    ),
                    database_uri=stats.external_database_uri,
                )
                topic_summaries.append(topic_summary)

            # Sort by database_uri first, then topic name
            topic_summaries.sort(key=lambda x: (x.database_uri, x.topic_name))

            response_data = {
                "topics": [topic.dict() for topic in topic_summaries],
                "total_topics": len(topic_summaries),
                "filter_database_uri": database_uri,
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
