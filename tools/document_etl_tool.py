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
from typing import Dict, Any, Optional
import logging

from tools.base import BaseTool, ToolResult
from etl.extract import extract_source_data
from knowledge_graph.models import RawDataSource, SourceData, ContentStore
from setting.db import SessionLocal


class DocumentETLTool(BaseTool):
    """
    Processes a single raw document file into structured SourceData.
    
    This tool takes a raw file and topic information, extracts content,
    and creates structured SourceData for further processing.
    
    Input Schema:
        file_path (str): Path to the document file
        topic_name (str): Topic name for grouping documents
        metadata (dict, optional): Custom metadata to attach
        force_reprocess (bool, optional): Force reprocessing even if already processed
        link (str, optional): Document URL/link
        original_filename (str, optional): Original filename if different from file_path
    
    Output Schema:
        source_data_id (str): ID of created SourceData record
        content_hash (str): SHA-256 hash of extracted content
        content_size (int): Size of extracted content in bytes
        source_type (str): Detected content type (pdf, txt, etc.)
        reused_existing (bool): Whether existing SourceData was reused
        status (str): Processing status (created, already_processed, etc.)
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
            "required": ["file_path", "topic_name"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file"
                },
                "topic_name": {
                    "type": "string",
                    "description": "Topic name for grouping documents"
                },
                "metadata": {
                    "type": "object",
                    "description": "Custom metadata to attach to the document",
                    "default": {}
                },
                "force_reprocess": {
                    "type": "boolean",
                    "description": "Whether to force reprocessing if SourceData already exists",
                    "default": False
                },
                "link": {
                    "type": "string",
                    "description": "Document URL/link",
                    "default": None
                },
                "original_filename": {
                    "type": "string",
                    "description": "Original filename if different from file_path",
                    "default": None
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_data_id": {
                    "type": "string",
                    "description": "ID of created SourceData record"
                },
                "content_hash": {
                    "type": "string",
                    "description": "SHA-256 hash of extracted content"
                },
                "content_size": {
                    "type": "integer",
                    "description": "Size of extracted content in bytes"
                },
                "source_type": {
                    "type": "string",
                    "description": "Detected content type (pdf, txt, etc.)"
                },
                "reused_existing": {
                    "type": "boolean",
                    "description": "Whether existing SourceData was reused"
                },
                "status": {
                    "type": "string",
                    "description": "Processing status"
                }
            }
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        file_path = input_data.get("file_path")
        if not file_path or not Path(file_path).exists():
            return False
        
        topic_name = input_data.get("topic_name")
        if not topic_name or not isinstance(topic_name, str):
            return False
            
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute ETL processing for a single document.
        
        Args:
            input_data: Dictionary containing:
                - file_path: Path to the document file
                - topic_name: Topic name for grouping
                - metadata: Optional custom metadata
                - force_reprocess: Whether to force reprocessing
                - link: Optional document link/URL
                - original_filename: Optional original filename
                
        Returns:
            ToolResult with SourceData processing results
        """
        try:
            file_path = Path(input_data["file_path"])
            topic_name = input_data["topic_name"]
            metadata = input_data.get("metadata", {})
            force_reprocess = input_data.get("force_reprocess", False)
            link = input_data.get("link", f"file://{file_path}")
            original_filename = input_data.get("original_filename", file_path.name)
            
            self.logger.info(f"Starting ETL processing for file: {file_path}")
            
            # Check if file exists
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Calculate file hash for deduplication
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()
            
            with self.session_factory() as db:
                # Check if we already have this content
                content_store = db.query(ContentStore).filter_by(
                    content_hash=file_hash
                ).first()
                
                # Create or get RawDataSource record
                raw_data_source = db.query(RawDataSource).filter_by(
                    file_path=str(file_path),
                    topic_name=topic_name
                ).first()
                
                if not raw_data_source:
                    raw_data_source = RawDataSource(
                        file_path=str(file_path),
                        topic_name=topic_name,
                        original_filename=original_filename,
                        metadata=metadata,
                        status="pending"
                    )
                    db.add(raw_data_source)
                    db.flush()
                
                # Check if already processed and not forcing reprocess
                if not force_reprocess:
                    existing_source_data = db.query(SourceData).filter(
                        SourceData.raw_data_source_id == raw_data_source.id
                    ).first()
                    
                    if existing_source_data:
                        self.logger.info(f"SourceData already exists for file: {file_path}")
                        return ToolResult(
                            success=True,
                            data={
                                "source_data_id": existing_source_data.id,
                                "content_hash": existing_source_data.content_hash,
                                "reused_existing": True,
                                "status": "already_processed"
                            },
                            metadata={
                                "file_path": str(file_path),
                                "topic_name": topic_name,
                                "file_size": len(file_content)
                            }
                        )
                
                # Update status to processing
                raw_data_source.status = "etl_processing"
                db.commit()
                
                # Extract content from file
                try:
                    extraction_result = extract_source_data(str(file_path))
                    content = extraction_result.get("content", "")
                    source_type = extraction_result.get("file_type", "unknown")
                    # Normalize content type to match job standards
                    if source_type == "pdf":
                        source_type = "application/pdf"
                    elif source_type == "md":
                        source_type = "text/markdown"
                    elif source_type == "txt":
                        source_type = "text/plain"
                    elif source_type == "sql":
                        source_type = "application/sql"
                    else:
                        source_type = "text/plain"
                except Exception as e:
                    self.logger.error(f"Failed to extract content from {file_path}: {e}")
                    # Update status to failed
                    raw_data_source.status = "etl_failed"
                    db.commit()
                    return ToolResult(
                        success=False,
                        error_message=f"Content extraction failed: {str(e)}"
                    )
                
                # Create or update ContentStore
                if not content_store:
                    content_store = ContentStore(
                        content_hash=file_hash,
                        content=content,
                        content_size=len(content),
                        content_type=source_type,
                        name=file_path.stem,
                        link=link
                    )
                    db.add(content_store)
                    db.flush()
                
                # Create SourceData record
                source_data = SourceData(
                    name=original_filename,
                    topic_name=topic_name,
                    raw_data_source_id=raw_data_source.id,
                    content_hash=file_hash,
                    link=link,
                    source_type=source_type,
                    attributes={
                        **metadata,
                        "file_path": str(file_path),
                        "original_filename": original_filename,
                        "file_size": len(file_content),
                        "extraction_method": "DocumentETLTool"
                    },
                    status="created"
                )
                
                # Update RawDataSource status
                raw_data_source.status = "etl_completed"
                
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
                        "status": "created"
                    },
                    metadata={
                        "file_path": str(file_path),
                        "topic_name": topic_name,
                        "file_size": len(file_content),
                        "content_type": source_type
                    }
                )
                
        except Exception as e:
            self.logger.error(f"ETL processing failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )


# Register the tool
from tools.base import TOOL_REGISTRY
TOOL_REGISTRY.register(DocumentETLTool())