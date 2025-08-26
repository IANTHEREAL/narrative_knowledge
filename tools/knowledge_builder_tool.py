"""
KnowledgeBuilderTool - A tool for building knowledge graphs from documents
using the KnowledgeBuilder class as a standalone tool outside the pipeline.
"""

import logging
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path

from tools.base import BaseTool, ToolResult
from knowledge_graph.knowledge import KnowledgeBuilder
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding
from knowledge_graph.models import SourceData
from setting.db import SessionLocal
from setting.base import LLM_MODEL

logger = logging.getLogger(__name__)


class KnowledgeBuilderTool(BaseTool):
    """
    A standalone tool for building knowledge graphs from documents using KnowledgeBuilder.
    This tool operates independently from the standard pipeline execution.
    """

    def __init__(self, session_factory=None, llm_client=None, embedding_func=None):
        """
        Initialize the KnowledgeBuilderTool.

        Args:
            session_factory: Database session factory
            llm_client: LLM interface for processing
            embedding_func: Embedding function for vector generation
        """
        super().__init__("KnowledgeBuilderTool", session_factory=session_factory)
        self.session_factory = session_factory or SessionLocal
        self.llm_client = LLMInterface("ollama", model=LLM_MODEL)
        self.embedding_func = embedding_func or get_text_embedding

    @property
    def tool_name(self) -> str:
        """Human-readable name of the tool."""
        return "KnowledgeBuilderTool"

    @property
    def tool_description(self) -> str:
        """Description of what the tool does."""
        return "Build knowledge graphs from documents using standalone KnowledgeBuilder class"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """JSON Schema for input validation."""
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to the document file to process (can be full path or filename)"
                },
                "files": {
                    "type": "array",
                    "description": "List of files to process in batch mode",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_path": {"type": "string"},
                            "filename": {"type": "string"},
                            "link": {"type": "string"},
                            "content_type": {"type": "string"},
                            "metadata": {"type": "object"}
                        }
                    }
                },
                "attributes": {
                    "type": "object",
                    "description": "Additional attributes for the document including metadata from daemon",
                    "properties": {
                        "doc_link": {"type": "string", "description": "Original document link/URL"},
                        "filename": {"type": "string", "description": "Original filename"},
                        "content_type": {"type": "string", "description": "MIME type of the document"},
                        "topic_name": {"type": "string", "description": "Topic name from context"},
                        "metadata": {"type": "object"}
                    }
                }
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data for the KnowledgeBuilderTool.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        # Support both single file and batch modes
        has_source_path = "source_path" in input_data
        has_files = "files" in input_data and isinstance(input_data["files"], list)
        
        if not has_source_path and not has_files:
            logger.error("Either source_path or files array is required")
            return False
            
        # For single file mode, validate source_path
        if has_source_path and not input_data.get("source_path"):
            logger.error("Source path cannot be empty")
            return False

        return True

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute the KnowledgeBuilderTool to build knowledge from documents.

        Args:
            input_data: Dictionary containing either:
                - source_path: Path to a single document file
                - files: List of file dictionaries (for batch processing)
                - attributes: Optional dict with document attributes
                - skip_extraction: Optional bool to skip knowledge block extraction
                - force_regenerate: Optional bool to force regeneration

        Returns:
            ToolResult: Result of the knowledge building process, with batch support
        """
        try:
            
            # Check for batch processing mode
            files = input_data.get("files", [])
            global_attributes = input_data.get("attributes", {})

            logger.info(f"Starting KnowledgeBuilder batch execution for {len(files)} files")

            # Initialize KnowledgeBuilder
            builder = KnowledgeBuilder(
                llm_client=self.llm_client,
                embedding_func=self.embedding_func,
                session_factory=self.session_factory
            )

            # Process each file
            batch_results = []
            total_knowledge_blocks = 0
            all_source_ids = []

            for idx, file_info in enumerate(files):
                try:
                    # Prepare inputs for KnowledgeBuilder
                    source_path = file_info.get("link") or file_info.get("source_path") or file_info.get("filename")
                    file_attributes = {**global_attributes, **file_info.get("metadata", {})}
                    
                    # Add standard file info
                    if file_info.get("link"):
                        file_attributes["doc_link"] = file_info["link"]
                    if file_info.get("filename"):
                        file_attributes["filename"] = file_info["filename"]
                    if file_info.get("content_type"):
                        file_attributes["content_type"] = file_info["content_type"]

                    if not source_path:
                        logger.warning(f"Skipping file {idx}: no source path found")
                        continue

                    logger.info(f"Processing file {idx+1}/{len(files)}: {source_path}")

                    # Step 1: Extract knowledge from source
                    extraction_result = builder.extract_knowledge(source_path, file_attributes)
                    
                    if extraction_result["status"] != "success":
                        batch_results.append({
                            "source_path": source_path,
                            "success": False,
                            "error": extraction_result.get("error", "Unknown error")
                        })
                        continue

                    source_id = extraction_result["source_id"]
                    all_source_ids.append(source_id)
                    
                    # Step 2: Split into knowledge blocks
                    knowledge_blocks = []
                    knowledge_blocks = builder.split_knowledge_blocks(source_id)
                    total_knowledge_blocks += len(knowledge_blocks)
                    logger.info(f"Extracted {len(knowledge_blocks)} knowledge blocks from {source_path}")

                    # Truncate content for preview display
                    truncated_blocks = []
                    for block in knowledge_blocks:
                        truncated_block = dict(block)
                        if "content" in truncated_block:
                            max_content_length = 200
                            content = truncated_block["content"]
                            if len(content) > max_content_length:
                                truncated_block["content"] = content[:max_content_length] + "..."
                        truncated_blocks.append(truncated_block)

                    # Individual file result
                    file_result = {
                        "source_path": source_path,
                        "source_id": source_id,
                        "source_name": extraction_result.get("source_name"),
                        "source_type": extraction_result.get("source_type"),
                        "source_link": extraction_result.get("source_link"),
                        "knowledge_blocks_count": len(truncated_blocks),
                        "knowledge_blocks": truncated_blocks,
                        "success": True
                    }
                    batch_results.append(file_result)

                except Exception as e:
                    current_source_path = str(file_info) if isinstance(file_info, (str, Path)) else str(file_info.get("link") or file_info.get("source_path") or file_info.get("filename", f"file_{idx}"))
                    logger.error(f"Error processing file {current_source_path}: {e}", exc_info=True)
                    with SessionLocal() as db:
                        source_data = (
                            db.query(SourceData).filter(SourceData.id == source_id).first()
                        )
                        source_data.status = "blocks_failed"
                        db.commit()
                    batch_results.append({
                        "source_path": current_source_path,
                        "success": False,
                        "error": str(e)
                    })

            # Prepare batch response
            successful_files = [r for r in batch_results if r.get("success", False)]
            failed_files = [r for r in batch_results if not r.get("success", False)]
            
            response_data = {
                "total_files": len(files),
                "successful_files": len(successful_files),
                "failed_files": len(failed_files),
                "total_knowledge_blocks": total_knowledge_blocks,
                "source_ids": all_source_ids,
                "batch_results (Showing first one)": batch_results[0],
            }

            success = len(failed_files) == 0
            error_message = None if success else f"Failed to process {len(failed_files)} out of {len(files)} files."

            return ToolResult(
                success=success,
                data=response_data,
                error_message=error_message,
                metadata={
                    "total_files": len(files),
                    "successful_files": len(successful_files),
                    "failed_files": len(failed_files),
                    "total_knowledge_blocks": total_knowledge_blocks,
                    "source_ids": all_source_ids
                }
            )

        except Exception as e:
            logger.error(f"Error in KnowledgeBuilderTool execution: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error_message=str(e),
                data={"error": str(e)}
            )