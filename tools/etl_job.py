"""
ETL Job - Extract, Transform, Load raw data files into structured SourceData
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

from knowledge_graph.models import RawDataSource, SourceData, ContentStore
from etl.extract import extract_source_data
from .base import BaseJob, JobContext, JobResult


def _get_content_type_from_path(path: str) -> str:
    """Determine content type from file extension"""
    suffix = Path(path).suffix.lower()
    if suffix in [".pdf"]:
        return "application/pdf"
    elif suffix in [".md", ".markdown"]:
        return "text/markdown"
    elif suffix in [".txt"]:
        return "text/plain"
    elif suffix in [".sql"]:
        return "application/sql"
    else:
        return "text/plain"


class ETLJob(BaseJob):
    """
    ETL Job - Process raw data files and create SourceData records
    
    Input: raw_data_source_id
    Output: source_data_id, extracted content summary
    """
    
    @property
    def job_type(self) -> str:
        return "etl"
    
    @property
    def job_name(self) -> str:
        return "ETL Processing Job"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["raw_data_source_id"],
            "properties": {
                "raw_data_source_id": {
                    "type": "string",
                    "description": "ID of the RawDataSource to process"
                },
                "force_reprocess": {
                    "type": "boolean",
                    "description": "Whether to force reprocessing if SourceData already exists",
                    "default": False
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
                    "description": "ID of the created SourceData record"
                },
                "content_hash": {
                    "type": "string",
                    "description": "SHA-256 hash of the extracted content"
                },
                "content_size": {
                    "type": "integer",
                    "description": "Size of the extracted content in bytes"
                },
                "source_type": {
                    "type": "string",
                    "description": "Detected content type"
                },
                "extraction_summary": {
                    "type": "object",
                    "description": "Summary of the extraction process"
                }
            }
        }
    
    def execute_job_logic(self, context: JobContext) -> JobResult:
        """
        Execute ETL processing for a raw data source
        
        Args:
            context: Job execution context containing raw_data_source_id
            
        Returns:
            JobResult with SourceData creation results
        """
        raw_data_source_id = context.input_data["raw_data_source_id"]
        force_reprocess = context.input_data.get("force_reprocess", False)
        
        self.logger.info(f"Starting ETL processing for raw data source: {raw_data_source_id}")
        
        # Get raw data source
        with self.session_factory() as db:
            raw_data_source = db.query(RawDataSource).filter(
                RawDataSource.id == raw_data_source_id
            ).first()
            
            if not raw_data_source:
                return JobResult(
                    success=False,
                    error_message=f"RawDataSource not found: {raw_data_source_id}"
                )
            
            # Check if already processed (unless forcing reprocess)
            if not force_reprocess:
                existing_source_data = db.query(SourceData).filter(
                    SourceData.raw_data_source_id == raw_data_source_id
                ).first()
                
                if existing_source_data:
                    self.logger.info(f"SourceData already exists for raw data source: {raw_data_source_id}")
                    return JobResult(
                        success=True,
                        data={
                            "source_data_id": existing_source_data.id,
                            "content_hash": existing_source_data.content_hash,
                            "source_type": existing_source_data.source_type,
                            "extraction_summary": {
                                "status": "already_processed",
                                "reused_existing": True
                            }
                        }
                    )
            
            # Update status to processing
            raw_data_source.status = "etl_processing"
            db.commit()
        
        try:
            # Process the file
            result = self._process_raw_file(raw_data_source)
            
            # Update status to completed
            with self.session_factory() as db:
                raw_data_source = db.query(RawDataSource).filter(
                    RawDataSource.id == raw_data_source_id
                ).first()
                raw_data_source.status = "etl_completed"
                db.commit()
            
            self.logger.info(f"ETL processing completed successfully for: {raw_data_source_id}")
            return result
            
        except Exception as e:
            # Update status to failed
            with self.session_factory() as db:
                raw_data_source = db.query(RawDataSource).filter(
                    RawDataSource.id == raw_data_source_id
                ).first()
                raw_data_source.status = "etl_failed"
                raw_data_source.error_message = str(e)
                db.commit()
            
            raise e
    
    def _process_raw_file(self, raw_data_source: RawDataSource) -> JobResult:
        """
        Process the raw file and create SourceData record
        
        Args:
            raw_data_source: RawDataSource to process
            
        Returns:
            JobResult with processing results
        """
        # Check if file exists
        if not os.path.exists(raw_data_source.file_path):
            return JobResult(
                success=False,
                error_message=f"File not found: {raw_data_source.file_path}"
            )
        
        # Read raw file content for hash calculation
        with open(raw_data_source.file_path, "rb") as f:
            raw_content = f.read()
            content_hash = hashlib.sha256(raw_content).hexdigest()
        
        # Determine content type
        content_type = _get_content_type_from_path(raw_data_source.file_path)
        
        with self.session_factory() as db:
            # Check if content already exists in ContentStore
            content_store = db.query(ContentStore).filter_by(
                content_hash=content_hash
            ).first()
            
            extracted_content = None
            
            if not content_store:
                # New content - need to extract
                try:
                    source_info = extract_source_data(raw_data_source.file_path)
                    extracted_content = source_info.get("content", "")
                except Exception as e:
                    self.logger.error(f"Failed to extract content from {raw_data_source.file_path}: {e}")
                    return JobResult(
                        success=False,
                        error_message=f"Failed to extract content: {e}"
                    )
                
                # Create ContentStore record
                content_store = ContentStore(
                    content_hash=content_hash,
                    content=extracted_content,
                    content_size=len(raw_content),
                    content_type=content_type,
                    name=Path(raw_data_source.file_path).stem,
                    link=raw_data_source.file_path,
                )
                db.add(content_store)
                db.flush()  # Get the content_store committed
                
                self.logger.info(f"Created new ContentStore entry: {content_hash[:8]}...")
            else:
                # Content already exists, reuse it
                extracted_content = content_store.content
                self.logger.info(f"Reusing existing ContentStore entry: {content_hash[:8]}...")
            
            # Create or update SourceData record
            source_data = db.query(SourceData).filter(
                SourceData.raw_data_source_id == raw_data_source.id
            ).first()
            
            if source_data:
                # Update existing SourceData
                source_data.content_hash = content_hash
                source_data.source_type = content_type
                source_data.status = "updated"
                source_data.content_version = source_data.generate_content_version()
            else:
                # Create new SourceData
                # Build document link
                doc_link = f"file://{raw_data_source.file_path}"
                
                # Prepare attributes
                attributes = {
                    "topic_name": raw_data_source.topic_name,
                    "doc_link": doc_link,
                    "original_filename": raw_data_source.original_filename,
                }
                
                # Add custom metadata if present
                if raw_data_source.metadata:
                    attributes.update(raw_data_source.metadata)
                
                source_data = SourceData(
                    name=raw_data_source.original_filename,
                    topic_name=raw_data_source.topic_name,
                    raw_data_source_id=raw_data_source.id,
                    content_hash=content_hash,
                    link=doc_link,
                    source_type=content_type,
                    attributes=attributes,
                    status="created",
                    content_version="",  # Will be generated below
                )
                
                # Generate content version
                source_data.content_version = source_data.generate_content_version()
                
                db.add(source_data)
            
            db.commit()
            db.refresh(source_data)
            
            # Prepare extraction summary
            extraction_summary = {
                "status": "completed",
                "content_size_bytes": len(extracted_content),
                "content_type": content_type,
                "content_hash": content_hash,
                "reused_existing_content": content_store.created_at != db.query(ContentStore).filter_by(
                    content_hash=content_hash
                ).first().created_at if content_store else False,
                "source_data_action": "updated" if source_data.status == "updated" else "created"
            }
            
            return JobResult(
                success=True,
                data={
                    "source_data_id": source_data.id,
                    "content_hash": content_hash,
                    "content_size": len(extracted_content),
                    "source_type": content_type,
                    "extraction_summary": extraction_summary
                }
            ) 