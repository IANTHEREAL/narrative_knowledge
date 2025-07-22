"""
API Integration for flexible pipeline orchestration.

This module provides the integration layer for the /api/v1/save endpoint
to use the tool-based pipeline system as described in the design document.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import os
import uuid

from tools.orchestrator import PipelineOrchestrator
from tools.base import ToolResult

logger = logging.getLogger(__name__)


class PipelineAPIIntegration:
    """
    API integration layer for pipeline-based processing.
    
    Transforms the /api/v1/save endpoint into a flexible pipeline orchestrator
    that supports both explicit pipeline configuration and intelligent defaults.
    """
    
    def __init__(self, session_factory=None):
        self.orchestrator = PipelineOrchestrator(session_factory)
        self.logger = logging.getLogger(__name__)
    
    def process_request(self, request_data: Dict[str, Any], files: List[Dict[str, Any]], 
                       execution_id: Optional[str] = None) -> ToolResult:
        """
        Process a knowledge ingestion request using pipeline orchestration.
        
        Args:
            request_data: Standard API request data
            files: List of file information (each with path, metadata, etc.)
            execution_id: Optional execution ID for tracking
            
        Returns:
            ToolResult with processing results
        """
        execution_id = execution_id or str(uuid.uuid4())
        
        # Prepare context for pipeline execution
        context = self._prepare_context(request_data, files)
        
        # Execute pipeline based on process strategy
        return self.orchestrator.execute_with_process_strategy(context, execution_id)
    
    def _prepare_context(self, request_data: Dict[str, Any], files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare context for pipeline execution from API request.
        
        Args:
            request_data: API request data
            files: List of file information
            
        Returns:
            Context dictionary for pipeline execution
        """
        metadata = request_data.get("metadata", {})
        
        # Map SmartSave API parameters to tool system parameters
        context = {
            "target_type": request_data.get("target_type", "knowledge_graph"),
            "process_strategy": request_data.get("process_strategy", {}),
            "metadata": metadata,
            "files": files,
            "llm_client": request_data.get("llm_client"),
            "embedding_func": request_data.get("embedding_func"),
            "force_regenerate": request_data.get("force_regenerate", False),
            "topic_name": metadata.get("topic_name"),
            "link": metadata.get("link"),
            "database_uri": metadata.get("database_uri")
        }
        
        # Handle file-specific parameters if files are provided
        if files and len(files) == 1:
            file_info = files[0]
            context["file_path"] = file_info.get("path")
            context["original_filename"] = file_info.get("filename")
            
        return context
    
    def process_single_file(self, file_path: str, topic_name: str, metadata: Dict[str, Any] = None,
                          llm_client=None, embedding_func=None) -> ToolResult:
        """
        Process a single file using the appropriate pipeline.
        
        Args:
            file_path: Path to the file to process
            topic_name: Topic name for grouping
            metadata: Optional metadata for the file
            llm_client: LLM client instance
            embedding_func: Embedding function
            
        Returns:
            ToolResult with processing results
        """
        context = {
            "file_path": file_path,
            "topic_name": topic_name,
            "metadata": metadata or {},
            "llm_client": llm_client,
            "embedding_func": embedding_func
        }
        
        # Determine if this is a new topic or existing
        from setting.db import SessionLocal
        from knowledge_graph.models import SourceData
        
        with SessionLocal() as db:
            existing_count = db.query(SourceData).filter(
                SourceData.topic_name == topic_name
            ).count()
            
            is_new_topic = existing_count == 0
            context["metadata"]["is_new_topic"] = is_new_topic
        
        return self.orchestrator.execute_with_process_strategy(context)
    
    def process_batch_files(self, file_paths: List[str], topic_name: str, metadata: Dict[str, Any] = None,
                          llm_client=None, embedding_func=None) -> ToolResult:
        """
        Process multiple files using batch pipeline.
        
        Args:
            file_paths: List of file paths to process
            topic_name: Topic name for grouping
            metadata: Optional metadata for the files
            llm_client: LLM client instance
            embedding_func: Embedding function
            
        Returns:
            ToolResult with processing results
        """
        files = [{"path": fp, "metadata": {}} for fp in file_paths]
        
        context = {
            "files": files,
            "topic_name": topic_name,
            "metadata": metadata or {},
            "llm_client": llm_client,
            "embedding_func": embedding_func
        }
        
        return self.orchestrator.execute_with_process_strategy(context)


# Example usage
if __name__ == "__main__":
    # Example of direct pipeline usage
    integration = PipelineAPIIntegration()
    
    # Example 1: Explicit pipeline configuration
    explicit_request = {
        "target_type": "knowledge_graph",
        "metadata": {"topic_name": "New Topic"},
        "process_strategy": {
            "pipeline": ["etl", "blueprint_gen", "graph_build"]
        },
        "file_path": "/path/to/document.pdf"
    }
    
    # Example 2: Default pipeline selection
    default_request = {
        "target_type": "knowledge_graph",
        "metadata": {"topic_name": "Existing Topic", "is_new_topic": False},
        "file_path": "/path/to/document.pdf"
    }