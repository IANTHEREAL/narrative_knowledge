"""
DocumentETLTool: Processes a single raw document file.

- Purpose: To process a single raw document file
- Input: Raw file (e.g., PDF, TXT), Topic ID
- Output: Structured SourceData (text, initial entities, etc.)
- Maps to: KnowledgeBuilder.build_from_file logic
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from tools.base import BaseTool, ToolResult
from etl.extract import extract_source_data
from knowledge_graph.models import RawDataSource, SourceData, ContentStore
from setting.db import SessionLocal


class DocumentETLTool(BaseTool):
    """
    Processes raw document files into structured SourceData.

    This tool takes raw files and topic information, extracts content,
    and creates structured SourceData for further processing.
    Supports both single file and batch file processing.

    Input Schema (Single file):
        file_path (str): Path to the document file
        topic_name (str): Topic name for grouping documents
        metadata (dict, optional): Custom metadata to attach
        force_reprocess (bool, optional): Force reprocessing even if already processed
        link (str, optional): Document URL/link
        original_filename (str, optional): Original filename if different from file_path

    Input Schema (Batch files):
        files (list): List of file objects with path, metadata, link, filename properties
        topic_name (str): Topic name for grouping documents
        force_reprocess (bool, optional): Force reprocessing even if already processed

    Output Schema:
        source_data_ids (list[str]): IDs of created SourceData records
        results (list[dict]): Individual processing results for each file
        batch_summary (dict): Summary of batch processing
    """

    def __init__(self, session_factory=None):
        super().__init__(session_factory=session_factory)
        self.session_factory = session_factory or SessionLocal

    @property
    def tool_name(self) -> str:
        return "DocumentETLTool"

    @property
    def tool_key(self) -> str:
        return "etl"

    @property
    def tool_description(self) -> str:
        return "Processes a single raw document file into structured SourceData"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "oneOf": [
                {
                    "title": "Single File Processing",
                    "required": ["file_path", "topic_name"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file",
                        },
                        "topic_name": {
                            "type": "string",
                            "description": "Topic name for grouping documents",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Request-level metadata to attach to the document",
                            "default": {},
                        },
                        "force_regenerate": {
                            "type": "boolean",
                            "description": "Whether to force regeneration if SourceData already exists",
                            "default": False,
                        },
                        "link": {
                            "type": "string",
                            "description": "Document URL/link",
                            "default": None,
                        },
                        "original_filename": {
                            "type": "string",
                            "description": "Original filename if different from file_path",
                            "default": None,
                        },
                    },
                },
                {
                    "title": "Batch File Processing",
                    "required": ["files", "topic_name"],
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["path"],
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Path to the document file",
                                    },
                                    "metadata": {
                                        "type": "object",
                                        "description": "File-specific metadata to attach to this document",
                                        "default": {},
                                    },
                                    "link": {
                                        "type": "string",
                                        "description": "Document URL/link",
                                        "default": None,
                                    },
                                    "filename": {
                                        "type": "string",
                                        "description": "Original filename if different from path",
                                    },
                                },
                            },
                            "description": "List of files to process",
                        },
                        "topic_name": {
                            "type": "string",
                            "description": "Topic name for grouping documents",
                        },
                        "request_metadata": {
                            "type": "object",
                            "description": "Request-level metadata to apply to all documents",
                            "default": {},
                        },
                        "force_regenerate": {
                            "type": "boolean",
                            "description": "Whether to force regeneration if SourceData already exists",
                            "default": False,
                        },
                    },
                },
            ],
            "properties": {
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Whether to force regeneration if SourceData already exists",
                    "default": False,
                }
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of created SourceData records",
                },
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_data_id": {
                                "type": "string",
                                "description": "ID of created SourceData record",
                            },
                            "content_hash": {
                                "type": "string",
                                "description": "SHA-256 hash of extracted content",
                            },
                            "content_size": {
                                "type": "integer",
                                "description": "Size of extracted content in bytes",
                            },
                            "source_type": {
                                "type": "string",
                                "description": "Detected content type (pdf, txt, etc.)",
                            },
                            "reused_existing": {
                                "type": "boolean",
                                "description": "Whether existing SourceData was reused",
                            },
                            "status": {
                                "type": "string",
                                "description": "Processing status",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Original file path",
                            },
                        },
                    },
                    "description": "Individual processing results for each file",
                },
                "batch_summary": {
                    "type": "object",
                    "properties": {
                        "total_files": {"type": "integer"},
                        "processed_files": {"type": "integer"},
                        "reused_files": {"type": "integer"},
                        "failed_files": {"type": "integer"},
                    },
                    "description": "Summary of batch processing",
                },
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters with detailed error information."""
        # Validate topic_name
        topic_name = input_data.get("topic_name")
        if not topic_name:
            self.logger.error(
                "Validation error: Missing required parameter: 'topic_name'"
            )
            return False
        if not isinstance(topic_name, str):
            self.logger.error(
                f"Validation error: 'topic_name' must be a string, got {type(topic_name).__name__}"
            )
            return False

        # Single file validation
        if "file_path" in input_data:
            file_path = input_data.get("file_path")
            if not file_path:
                self.logger.error(
                    "Validation error: Missing required parameter: 'file_path'"
                )
                return False
            if not isinstance(file_path, str):
                self.logger.error(
                    f"Validation error: 'file_path' must be a string, got {type(file_path).__name__}"
                )
                return False
            if not Path(file_path).exists():
                self.logger.error(f"Validation error: File not found: {file_path}")
                return False

            # Validate metadata is dict for single file
            metadata = input_data.get("metadata", {})
            if not isinstance(metadata, dict):
                self.logger.error(
                    f"Validation error: 'metadata' must be a dict, got {type(metadata).__name__}"
                )
                return False

        # Batch file validation
        if "files" in input_data:
            files = input_data.get("files", [])
            if not files:
                self.logger.error(
                    "Validation error: Missing or empty parameter: 'files'"
                )
                return False
            if not isinstance(files, list):
                self.logger.error(
                    f"Validation error: 'files' must be a list, got {type(files).__name__}"
                )
                return False

            # Validate request_metadata is dict for batch processing
            request_metadata = input_data.get("request_metadata", {})
            if not isinstance(request_metadata, dict):
                self.logger.error(
                    f"Validation error: 'request_metadata' must be a dict, got {type(request_metadata).__name__}"
                )
                return False

            for i, file_info in enumerate(files):
                if not isinstance(file_info, dict):
                    self.logger.error(
                        f"Validation error: Files[{i}] must be a dict, got {type(file_info).__name__}"
                    )
                    return False
                # Validate file metadata is dict
                file_metadata = file_info.get("metadata", {})
                if not isinstance(file_metadata, dict):
                    self.logger.error(
                        f"Validation error: Files[{i}]['metadata'] must be a dict, got {type(file_metadata).__name__}"
                    )
                    return False

        return True

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute ETL processing for documents (single or batch).

        Args:
            input_data: Dictionary containing either:
                - Single file: file_path + topic_name + optional metadata
                - Batch files: files array + topic_name + optional metadata
                - force_regenerate: Whether to force regeneration
                - request_metadata: Global metadata from request (for batch processing)

        Returns:
            ToolResult with batch processing results
        """
        try:
            topic_name = input_data["topic_name"]
            force_regenerate_str = input_data.get("force_regenerate", False)

            if force_regenerate_str == "True" or force_regenerate_str == "true":
                force_regenerate = True
            else:
                force_regenerate = False

            self.logger.info(f"Force regenerate? : {force_regenerate}")

            # Determine processing mode
            if "files" in input_data:
                # Single/Batch processing
                files = input_data["files"]
                request_metadata = input_data.get("request_metadata", {})
                return self._process_batch_files(
                    files, topic_name, force_regenerate, request_metadata
                )
            else:
                # Single file processing (backward compatibility)
                file_path = Path(input_data["file_path"])
                request_metadata = input_data.get("metadata", {})
                link = input_data.get("link")
                original_filename = input_data.get("original_filename", file_path.name)

                file_info = {
                    "path": str(file_path),
                    "file_metadata": {},  # Empty for single file
                    "metadata": request_metadata,  # For backward compatibility and RawDataSource
                    "link": link,
                    "filename": original_filename,
                }

                result = self._process_single_file(
                    file_info, topic_name, force_regenerate
                )

                # Convert single file result to new format for backward compatibility
                if result.success:
                    return ToolResult(
                        success=True,
                        data={
                            "source_data_ids": [result.data["source_data_id"]],
                            "results": [result.data],
                            "batch_summary": {
                                "total_files": 1,
                                "processed_files": 1,
                                "reused_files": (
                                    1 if result.data.get("reused_existing") else 0
                                ),
                                "failed_files": 0,
                            },
                        },
                        metadata=result.metadata,
                    )
                else:
                    return result

        except Exception as e:
            self.logger.error(f"ETL processing failed: {e}")
            return ToolResult(success=False, error_message=str(e))

    def _process_batch_files(
        self,
        files: List[Dict[str, Any]],
        topic_name: str,
        force_regenerate: bool = False,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Process multiple files in batch."""
        results = []
        source_data_ids = []

        # Use empty dict if no request metadata provided
        request_metadata = request_metadata or {}

        total_files = len(files)
        processed_files = 0
        reused_files = 0
        failed_files = 0

        self.logger.info(f"Starting batch ETL processing for {total_files} files")

        for file_info in files:
            try:
                # Separate file metadata from request metadata
                file_metadata = file_info.get("metadata", {})

                # Create updated file_info with both metadata types
                updated_file_info = {
                    **file_info,
                    "request_metadata": request_metadata,
                    "file_metadata": file_metadata,
                    "metadata": request_metadata,  # For backward compatibility and RawDataSource
                }

                file_result = self._process_single_file(
                    updated_file_info, topic_name, force_regenerate
                )

                if file_result.success:
                    source_data_id = file_result.data.get("source_data_id")
                    if source_data_id:
                        source_data_ids.append(source_data_id)

                    results.append(
                        {
                            **file_result.data,
                            "file_path": file_info.get("path"),
                            "status": "success",
                        }
                    )

                    if file_result.data.get("reused_existing"):
                        reused_files += 1
                    else:
                        processed_files += 1

                else:
                    failed_files += 1
                    results.append(
                        {
                            "file_path": file_info.get("path"),
                            "status": "failed",
                            "error": file_result.error_message,
                        }
                    )

            except Exception as e:
                failed_files += 1
                results.append(
                    {
                        "file_path": file_info.get("path"),
                        "status": "failed",
                        "error": str(e),
                    }
                )

        self.logger.info(
            f"Batch ETL completed: {processed_files} processed, {reused_files} reused, {failed_files} failed"
        )

        return ToolResult(
            success=True,
            data={
                "source_data_ids": source_data_ids,
                "results": results,
                "batch_summary": {
                    "total_files": total_files,
                    "processed_files": processed_files,
                    "reused_files": reused_files,
                    "failed_files": failed_files,
                },
            },
            metadata={"topic_name": topic_name, "total_files": total_files},
        )

    def _process_single_file(
        self, file_info: Dict[str, Any], topic_name: str, force_regenerate: bool = False
    ) -> ToolResult:
        """Process a single file."""
        try:
            # Link is already resolved in route_wrapper.py
            link = file_info.get("link", None)

            filename = file_info.get("filename", None)

            self.logger.info(f"Starting ETL processing for file: {filename}")

            with self.session_factory() as db:
                raw_data_source = (
                    db.query(RawDataSource)
                    .filter_by(
                        original_filename=filename,
                        status="uploaded"
                    )
                    .first()
                )
                if not raw_data_source:
                    self.logger.info(f"RawDataSource not found for {file_info['filename']} in topic {topic_name}")
                    return ToolResult(
                        success=False,
                        error_message=f"RawDataSource not found for {file_info['filename']} in topic {topic_name}",
                    )
                file_path = raw_data_source.file_path
                self.logger.info(
                    "Successfully found RawDataSource for file: %s", file_path
                )
                # calculate file hash
                with open(file_path, "rb") as f:
                    file_content = f.read()
                    file_hash = hashlib.sha256(file_content).hexdigest()
                # Check if we already have this content
                content_store = (
                    db.query(ContentStore).filter_by(content_hash=file_hash).first()
                )
                self.logger.info(
                    f"ContentStore lookup for {file_path}: {content_store}"
                )
                if not force_regenerate:
                    self.logger.info(f"Force regenerate? : {force_regenerate}")    

                    existing_source_data = (
                        db.query(SourceData)
                        .filter(SourceData.raw_data_source_id == raw_data_source.id,
                                SourceData.topic_name == topic_name,)
                        .first()
                    )

                    if existing_source_data:
                        raw_data_source.status = "etl_completed"
                        db.commit()
                        self.logger.info(
                            f"SourceData already exists for file: {file_path} in topic {existing_source_data.topic_name}"
                        )
                        return ToolResult(
                            success=True,
                            data={
                                "source_data_id": existing_source_data.id,
                                "content_hash": existing_source_data.content_hash,
                                "content_size": len(
                                    existing_source_data.effective_content or ""
                                ),
                                "source_type": existing_source_data.source_type,
                                "reused_existing": True,
                                "status": "already_processed",
                            },
                            metadata={
                                "file_path": str(file_path),
                                "topic_name": topic_name,
                                "file_size": len(file_content),
                            },
                        )

                # Update status to processing
                raw_data_source.status = "etl_processing"  # type: ignore
                db.commit()

                # Extract content from file
                try:
                    extraction_result = extract_source_data(str(file_path))
                    # Handle both string and dict return types safely
                    if isinstance(extraction_result, dict):
                        content = extraction_result.get("content", "")
                        source_type = extraction_result.get("file_type", "text/plain")
                    else:
                        content = str(extraction_result)
                        source_type = "text/plain"
                    # Normalize content type to match job standards
                    if source_type == "pdf":
                        source_type = "application/pdf"
                    elif (
                        source_type == "markdown"
                    ):  # align with the outputs from extract_source_data()
                        source_type = "text/markdown"
                    elif source_type == "document":
                        source_type = "text/plain"
                    elif source_type == "sql":
                        source_type = "application/sql"
                    else:
                        source_type = "text/plain"
                except Exception as e:
                    self.logger.error(
                        f"Failed to extract content from {file_path}: {e}"
                    )
                    # Update status to failed
                    raw_data_source.status = "etl_failed"  # type: ignore
                    db.commit()
                    return ToolResult(
                        success=False,
                        error_message=f"Content extraction failed: {str(e)}",
                    )
                self.logger.info(
                    f"Extracted content from {file_path}, size: {len(content)} bytes"
                )
                # Create or update ContentStore

                if not content_store:
                    content_store = ContentStore(
                        content_hash=file_hash,
                        content=content,
                        content_size=len(content),
                        content_type=source_type,
                        name=filename,
                        link=link,
                    )
                    db.add(content_store)
                    db.flush()
                self.logger.info(
                    f"ContentStore created/updated for {file_path}: {content_store.content_hash}"
                )
                # Create SourceData record with separated metadata
                source_data = SourceData(
                    name=filename,
                    topic_name=topic_name,
                    raw_data_source_id=raw_data_source.id,
                    content_hash=file_hash,
                    link=link,
                    source_type=source_type,
                    attributes={
                        "file_path": str(file_path),
                        "original_filename": filename,
                        "file_size": len(file_content),
                        "extraction_method": "DocumentETLTool",
                    },
                    status="created",
                )

                # Update RawDataSource status
                raw_data_source.status = "etl_completed"  # type: ignore

                self.logger.info(f"Creating SourceData for file: {file_path}")

                db.add(source_data)
                db.commit()
                db.refresh(source_data)

                self.logger.info(f"ETL processing completed for file: {file_path}")

                return ToolResult(
                    success=True,
                    data={
                        "source_data_id": source_data.id,
                        "content_hash": file_hash,
                        "content_size": len(content),
                        "source_type": source_type,
                        "reused_existing": False,
                        "status": "created",
                    },
                    metadata={
                        "file_path": str(file_path),
                        "topic_name": topic_name,
                        "file_size": len(file_content),
                        "content_type": source_type,
                    },
                )

        except Exception as e:
            self.logger.error(f"ETL processing failed: {e}")
            return ToolResult(success=False, error_message=str(e))


# Register the tool - will be handled by orchestrator initialization
