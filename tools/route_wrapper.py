import tempfile
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from fastapi import UploadFile
import json

from tools.api_integration import PipelineAPIIntegration
from tools.base import ToolResult
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding


class ToolsRouteWrapper:
    """
    Route wrapper that converts FastAPI's UploadFile to a format processable by the tools system
    """

    def __init__(self, session_factory=None):
        self.api_integration = PipelineAPIIntegration(session_factory)
        self.temp_dir = Path(tempfile.mkdtemp())

    def process_upload_request(
        self,
        files: Union[UploadFile, List[UploadFile]],
        metadata: Union[str, Dict[str, Any]],
        process_strategy: Optional[Union[str, Dict[str, Any]]] = None,
        target_type: str = "knowledge_graph",
        links: List[str] = [],
        
    ) -> ToolResult:
        """
        Route wrapper function: Process file upload requests

        Args:
            files: FastAPI UploadFile object or list
            metadata: Metadata (can be JSON string or dict)
            process_strategy: Processing strategy (can be JSON string or dict)
            target_type: Target type
            links: List of URLs/links corresponding to files (can be JSON string or list)
            llm_client: LLM client (optional, will auto-create)
            embedding_func: Embedding function (optional, will auto-create)

        Returns:
            ToolResult: Processing result
        """
        prepared_files = []
        try:
            # 1. Standardized input parameters
            files_list = files if isinstance(files, list) else [files]
            parsed_metadata = self._parse_metadata(metadata)
            parsed_strategy = self._parse_process_strategy(process_strategy)

            # 3. Save files to temporary directory and prepare information
            prepared_files = self._prepare_files(files_list, parsed_metadata, links)

            # 4. Construct request_data
            request_data = self._build_request_data(
                target_type,
                parsed_metadata,
                parsed_strategy,
            )

            # 5. Call tools system
            result = self.api_integration.process_request(
                request_data=request_data, files=prepared_files
            )

            return result

        except Exception as e:
            # Return tool results in standard format
            return ToolResult(
                success=False, error_message=f"Route wrapper error: {str(e)}", data={}
            )

    def process_json_request(
        self,
        input_data: Any,
        metadata: Union[str, Dict[str, Any]],
        process_strategy: Optional[Union[str, Dict[str, Any]]] = None,
        target_type: str = "personal_memory",
        llm_client=None,
        embedding_func=None,
    ) -> ToolResult:
        """
        Route wrapper function: Process JSON data requests (e.g., chat logs)

        Args:
            input_data: Input data (chat logs, text, etc.)
            metadata: Metadata
            process_strategy: Processing strategy
            target_type: Target type
            llm_client: LLM client
            embedding_func: Embedding function

        Returns:
            ToolResult: Processing result
        """
        try:
            # 1. Analyze parameters
            parsed_metadata = self._parse_metadata(metadata)
            parsed_strategy = self._parse_process_strategy(process_strategy)

            # 2. Create default client
            if llm_client is None:
                llm_client = LLMInterface("openai", model="gpt-4o")
            if embedding_func is None:
                embedding_func = get_text_embedding

            # 3. Construct request_data
            request_data = self._build_request_data(
                target_type,
                parsed_metadata,
                parsed_strategy,
            )

            # 4. Add input data
            if target_type == "personal_memory":
                request_data["chat_messages"] = input_data
                request_data["source_id"] = parsed_metadata.get("source_id")
                request_data["user_id"] = parsed_metadata.get("user_id")
                request_data["topic_name"] = parsed_metadata.get("topic_name")
            else:
                request_data["input"] = input_data

            # 5. Call tools system（without files)
            result = self.api_integration.process_request(
                request_data=request_data, files=[]
            )

            return result

        except Exception as e:
            return ToolResult(
                success=False, error_message=f"JSON request error: {str(e)}", data={}
            )

    def _parse_metadata(self, metadata: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse metadata"""
        if isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in metadata: {e}")
        return metadata or {}

    def _parse_process_strategy(
        self, process_strategy: Optional[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Parse processing strategy"""
        if process_strategy is None:
            return {}
        if isinstance(process_strategy, str):
            try:
                return json.loads(process_strategy)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in process_strategy: {e}")
        return process_strategy

    def _prepare_files(
        self,
        files: List[UploadFile],
        metadata: Dict[str, Any],
        links: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare file information"""
        prepared_files = []
        links = links or []

        for i, file in enumerate(files):
            # Resolve link with steps: links param > metadata.links > metadata.link > default
            link = None

            # First: use provided links parameter
            if i < len(links):
                link = links[i]
            # construct file info
            file_info = {
                "filename": file.filename,
                "metadata": metadata,  # file-specific metadata
                "link": link,
                "content_type": file.content_type,
            }
            prepared_files.append(file_info)
        return prepared_files

    def _build_request_data(
        self,
        target_type: str,
        metadata: Dict[str, Any],
        process_strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build request data"""
        return {
            "target_type": target_type,
            "metadata": metadata,
            "process_strategy": process_strategy,
            "force_regenerate": metadata.get("force_regenerate", False),
        }

    def __del__(self):
        """Cleanup temporary directory"""
        try:
            if hasattr(self, "temp_dir") and self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
