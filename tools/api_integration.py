"""
API Integration for flexible pipeline orchestration.

This module provides the integration layer for the /api/v1/save endpoint
to use the tool-based pipeline system as described in the design document.
"""

import logging
from typing import Dict, Any, List, Optional

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

    def process_request(
        self,
        request_data: Dict[str, Any],
        files: List[Dict[str, Any]],
        execution_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Process a knowledge ingestion request using pipeline orchestration.

        Args:
            request_data: Standard API request data
            files: List of file information (each with path, metadata, etc.)
            execution_id: Optional execution ID for tracking

        Returns:
            ToolResult with processing results
        """
        # TODO: what is the function of execution_id?
        execution_id = execution_id or str(uuid.uuid4())

        # Prepare context for pipeline execution
        context = self._prepare_context(request_data, files)

        # Execute pipeline based on process strategy
        return self.orchestrator.execute_with_process_strategy(context, execution_id)

    def _determine_is_new_topic(
        self, metadata: Dict[str, Any], target_type: str
    ) -> bool:
        """
        Determine if this is a new topic based on metadata and target type.

        Args:
            metadata: Request metadata
            target_type: Target type (knowledge_graph or personal_memory)

        Returns:
            True if new topic, False if existing, or default for memory processing
        """
        # First check explicit metadata flag
        explicit_is_new_topic = metadata.get("is_new_topic")
        if explicit_is_new_topic is not None:
            return bool(explicit_is_new_topic)

        # Determine from database if topic exists
        from setting.db import SessionLocal
        from knowledge_graph.models import SourceData

        topic_name = metadata.get("topic_name")
        if topic_name:
            with SessionLocal() as db:
                existing_count = (
                    db.query(SourceData)
                    .filter(SourceData.topic_name == topic_name)
                    .count()
                )
                return existing_count == 0
        else:
            # No topic name provided, assume new topic for documents
            return True

    def _prepare_context(
        self, request_data: Dict[str, Any], files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare context for pipeline execution from API request.

        Args:
            request_data: API request data
            files: List of file information

        Returns:
            Context dictionary for pipeline execution
        """
        metadata = request_data.get("metadata", {})
        target_type = request_data.get("target_type", "knowledge_graph")

        # Personal memory specific parameters
        if target_type == "personal_memory":
            # Extract user_id from metadata or request_data
            context = {
                "target_type": target_type,
                "process_strategy": request_data.get("process_strategy", {}),
                "metadata": metadata,
                "user_id": metadata.get("user_id") or request_data.get("user_id"),
                "source_id": metadata.get("source_id"),  # For processing existing data
                "chat_messages": request_data.get("chat_messages", []),
                "topic_name": request_data.get("topic_name"),
                "force_regenerate": request_data.get("force_regenerate", False),
            }

            # Ensure we have required personal memory fields
            if not context["user_id"]:
                # Try to extract from files metadata if available
                if files and files[0].get("metadata"):
                    context["user_id"] = files[0]["metadata"].get("user_id")

        else:
            # Determine if this is a new topic or existing (only for knowledge_graph processing)
            is_new_topic = self._determine_is_new_topic(metadata, target_type)
            self.logger.info(f"Using new topic? (True/False): {is_new_topic}")

            # Base context for both target types
            context = {
                "target_type": target_type,
                "process_strategy": request_data.get("process_strategy", {}),
                "metadata": metadata,
                "files": files,
                "force_regenerate": request_data.get("force_regenerate", False),
                "topic_name": metadata.get("topic_name"),
                "database_uri": metadata.get("database_uri"),
                "is_new_topic": (
                    is_new_topic if target_type == "knowledge_graph" else None
                ),
            }
        return context

    def process_single_file(
        self,
        file_path: str,
        topic_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        llm_client=None,
        embedding_func=None,
    ) -> ToolResult:
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
            "embedding_func": embedding_func,
        }

        # Determine if this is a new topic or existing
        from setting.db import SessionLocal
        from knowledge_graph.models import SourceData

        with SessionLocal() as db:
            existing_count = (
                db.query(SourceData).filter(SourceData.topic_name == topic_name).count()
            )

            is_new_topic = existing_count == 0
            context["metadata"]["is_new_topic"] = is_new_topic

        return self.orchestrator.execute_with_process_strategy(context)

    def process_batch_files(
        self,
        file_paths: List[str],
        topic_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        llm_client=None,
        embedding_func=None,
    ) -> ToolResult:
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
            "embedding_func": embedding_func,
        }

        return self.orchestrator.execute_with_process_strategy(context)
