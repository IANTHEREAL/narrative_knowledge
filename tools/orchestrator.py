"""
Pipeline Orchestrator for dynamic tool sequencing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import uuid

from tools.base import ToolResult
from tools.base import TOOL_REGISTRY
from setting.db import SessionLocal


class PipelineOrchestrator:
    """
    Orchestrates tool execution into dynamic pipelines.
    
    Supports main scenarios:
    1. Adding single document to existing topic
    2. Adding batch documents to existing topic  
    3. Creating new topic with batch documents
    """
    
    def __init__(self, session_factory=None):
        self.session_factory = session_factory or SessionLocal
        self.logger = logging.getLogger(__name__)
        
        # Define standard pipelines using tool keys
        self.standard_pipelines = {
            # Knowledge graph pipelines
            "single_doc_existing_topic": ["etl", "graph_build"],
            "batch_doc_existing_topic": ["etl", "blueprint_gen", "graph_build"],
            "new_topic_batch": ["etl", "blueprint_gen", "graph_build"],
            "text_to_graph": ["graph_build"],
            
            # Memory pipelines
            "memory_direct_graph": ["memory_graph_build"],  # Direct memory processing with graph building
            "memory_single": ["memory_graph_build"]  # Single memory processing
        }
        
        # Tool key to name mapping
        self.tool_key_mapping = {
            "etl": "DocumentETLTool",
            "blueprint_gen": "BlueprintGenerationTool", 
            "graph_build": "GraphBuildTool",
            "memory_graph_build": "MemoryGraphBuildTool"
        }
    
    def execute_pipeline(self, pipeline_name: str, context: Dict[str, Any], execution_id: Optional[str] = None) -> ToolResult:
        """
        Execute a predefined pipeline.
        
        Args:
            pipeline_name: Name of the predefined pipeline
            context: Context data for pipeline execution
            execution_id: Optional execution ID for tracking
            
        Returns:
            ToolResult with pipeline execution results
        """
        execution_id = execution_id or str(uuid.uuid4())
        
        if pipeline_name not in self.standard_pipelines:
            return ToolResult(
                success=False,
                error_message=f"Pipeline '{pipeline_name}' not found"
            )
        
        tool_keys = self.standard_pipelines[pipeline_name]
        tools = [self.tool_key_mapping[key] for key in tool_keys]
        return self.execute_custom_pipeline(tools, context, execution_id)
    
    def execute_custom_pipeline(self, tools: List[str], context: Dict[str, Any], execution_id: Optional[str] = None) -> ToolResult:
        """
        Execute a custom pipeline with specific tool sequence.
        
        Args:
            tools: List of tool names to execute in sequence
            context: Context data for pipeline execution
            execution_id: Optional execution ID for tracking
            
        Returns:
            ToolResult with pipeline execution results
        """
        execution_id = execution_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting pipeline execution: {execution_id} - {tools}")
        
        results = {}
        pipeline_context = context.copy()
        
        try:
            for tool_name in tools:
                tool = TOOL_REGISTRY.get_tool(tool_name)
                if not tool:
                    return ToolResult(
                        success=False,
                        error_message=f"Tool '{tool_name}' not found"
                    )
                
                # Prepare input for this tool
                tool_input = self._prepare_tool_input(tool_name, pipeline_context, results)
                
                # Execute tool
                self.logger.info(f"Executing tool: {tool_name}")
                result = tool.execute_with_tracking(tool_input, f"{execution_id}_{tool_name}")
                
                if not result.success:
                    return ToolResult(
                        success=False,
                        error_message=f"Tool '{tool_name}' failed: {result.error_message}",
                        execution_id=execution_id,
                        data={"failed_tool": tool_name, "previous_results": results}
                    )
                
                results[tool_name] = result
                pipeline_context = self._update_context(tool_name, pipeline_context, result)
                
                self.logger.info(f"Tool completed: {tool_name}")
            
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Pipeline execution completed: {execution_id} in {duration:.2f}s")
            
            return ToolResult(
                success=True,
                data={
                    "results": results,
                    "pipeline": tools,
                    "duration_seconds": duration
                },
                execution_id=execution_id,
                duration_seconds=duration
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Pipeline execution failed: {execution_id} - {e}")
            
            return ToolResult(
                success=False,
                error_message=str(e),
                execution_id=execution_id,
                duration_seconds=duration
            )
    
    def select_default_pipeline(self, target_type: str, topic_name: str, file_count: int, is_new_topic: bool, 
                              input_type: str = "document", file_extension: str = None) -> str:
        """
        Select appropriate default pipeline based on context.
        
        Args:
            target_type: Target type (knowledge_graph, personal_memory)
            topic_name: Topic name
            file_count: Number of files to process
            is_new_topic: Whether this is a new topic
            input_type: Type of input (document, dialogue, text)
            file_extension: File extension for document processing
            
        Returns:
            Name of the default pipeline to use
        """
        
        # Memory pipeline for dialogue history/chat
        if target_type == "personal_memory" or input_type == "dialogue":
            if input_type == "dialogue":
                return "memory_direct_graph"  # Direct graph extraction from dialogue
            else:
                return "memory_single"  # Single memory processing
        
        # Knowledge graph pipeline for documents
        if target_type == "knowledge_graph":
            # Document processing pipeline
            if input_type == "document":
                if is_new_topic:
                    return "new_topic_batch"
                elif file_count == 1:
                    return "single_doc_existing_topic"
                else:
                    return "batch_doc_existing_topic"
            elif input_type == "text":
                return "text_to_graph"  # Direct text processing
        
        # Default fallback
        return "single_doc_existing_topic"
    
    def _prepare_tool_input(self, tool_name: str, context: Dict[str, Any], 
                           previous_results: Dict[str, ToolResult]) -> Dict[str, Any]:
        """Prepare input for a specific tool based on context and previous results."""
        
        # Map tool names to keys for consistent handling
        tool_key = None
        for key, name in self.tool_key_mapping.items():
            if name == tool_name:
                tool_key = key
                break
        
        if tool_key == "etl":
            return {
                "file_path": context.get("file_path"),
                "topic_name": context.get("topic_name"),
                "metadata": context.get("metadata", {}),
                "force_regenerate": context.get("force_regenerate", False),
                "link": context.get("link"),
                "original_filename": context.get("original_filename")
            }
        
        elif tool_key == "blueprint_gen":
            # Collect source_data_ids from context (accumulated from ETL results)
            source_data_ids = context.get("source_data_ids", [])
            
            # If no accumulated IDs, check for single ID
            if not source_data_ids and context.get("source_data_id"):
                source_data_ids = [context.get("source_data_id")]
            
            # Handle topic-based processing when no specific IDs provided
            if not source_data_ids and context.get("topic_name"):
                return {
                    "topic_name": context.get("topic_name"),
                    "source_data_ids": source_data_ids or None,
                    "force_regenerate": context.get("force_regenerate", False),
                    "llm_client": context.get("llm_client"),
                    "embedding_func": context.get("embedding_func")
                }
            
            return {
                "topic_name": context.get("topic_name"),
                "source_data_ids": source_data_ids,
                "force_regenerate": context.get("force_regenerate", False),
                "llm_client": context.get("llm_client"),
                "embedding_func": context.get("embedding_func")
            }
        
        elif tool_key == "graph_build":
            topic_name = context.get("topic_name")
            
            # Check if we have ETL and/or Blueprint from pipeline
            has_etl = any(self.tool_key_mapping.get("etl") in tools for tools in [previous_results.keys()])
            has_blueprint = any(self.tool_key_mapping.get("blueprint_gen") in tools for tools in [previous_results.keys()])
            
            # Case 1: We have ETL/Blueprint tools in pipeline (with results)
            if has_etl or has_blueprint:
                source_data_ids = context.get("source_data_ids", [])
                blueprint_id = context.get("blueprint_id")
                
                # Fill missing IDs from context
                if not source_data_ids and context.get("source_data_id"):
                    source_data_ids = [context.get("source_data_id")]
                
                if source_data_ids and blueprint_id:
                    return {
                        "source_data_ids": source_data_ids,
                        "blueprint_id": blueprint_id,
                        "force_regenerate": context.get("force_regenerate", False),
                        "llm_client": context.get("llm_client"),
                        "embedding_func": context.get("embedding_func")
                    }
                elif source_data_ids and len(source_data_ids) == 1:
                    return {
                        "source_data_id": source_data_ids[0],
                        "force_regenerate": context.get("force_regenerate", False),
                        "llm_client": context.get("llm_client"),
                        "embedding_func": context.get("embedding_func")
                    }
                else:
                    return {
                        "topic_name": topic_name,
                        "force_regenerate": context.get("force_regenerate", False),
                        "llm_client": context.get("llm_client"),
                        "embedding_func": context.get("embedding_func")
                    }
            
            # Case 2: No ETL/Blueprint (graph_build-only pipelines)
            else:
                # Default to topic-based processing
                return {
                    "topic_name": topic_name,
                    "force_regenerate": context.get("force_regenerate", False),
                    "llm_client": context.get("llm_client"),
                    "embedding_func": context.get("embedding_func")
                }
        
        return context.copy()
    
    def _update_context(self, tool_name: str, context: Dict[str, Any], result: ToolResult) -> Dict[str, Any]:
        """Update context with results from a tool."""
        updated_context = context.copy()
        
        # Map tool name to key
        tool_key = None
        for key, name in self.tool_key_mapping.items():
            if name == tool_name:
                tool_key = key
                break
        
        if tool_key == "etl" and result.success:
            source_data_id = result.data.get("source_data_id")
            if source_data_id:
                # Maintain both singular and plural forms for consistency
                updated_context["source_data_id"] = source_data_id
                if "source_data_ids" not in updated_context:
                    updated_context["source_data_ids"] = []
                updated_context["source_data_ids"].append(source_data_id)
            
            updated_context["topic_name"] = result.metadata.get("topic_name")
        
        elif tool_key == "blueprint_gen" and result.success:
            updated_context["blueprint_id"] = result.data.get("blueprint_id")
            updated_context["topic_name"] = result.metadata.get("topic_name")
        
        return updated_context
    
    def execute_scenario(self, scenario: str, context: Dict[str, Any], execution_id: Optional[str] = None) -> ToolResult:
        """
        Execute a specific scenario with appropriate pipeline.
        
        Args:
            scenario: One of 'single_doc_existing', 'batch_doc_existing', 'new_topic'
            context: Context data for the scenario
            execution_id: Optional execution ID
            
        Returns:
            ToolResult with scenario execution results
        """
        scenario_to_pipeline = {
            "single_doc_existing": "single_doc_existing_topic",
            "batch_doc_existing": "batch_doc_existing_topic", 
            "new_topic": "new_topic_batch"
        }
        
        if scenario not in scenario_to_pipeline:
            return ToolResult(
                success=False,
                error_message=f"Scenario '{scenario}' not supported"
            )
        
        return self.execute_pipeline(
            scenario_to_pipeline[scenario], 
            context, 
            execution_id
        )
    
    def execute_with_process_strategy(self, request_data: Dict[str, Any], execution_id: Optional[str] = None) -> ToolResult:
        """
        Execute pipeline based on process strategy parameter from API request.
        
        Args:
            request_data: API request data containing process_strategy
            execution_id: Optional execution ID
            
        Returns:
            ToolResult with execution results
        """
        execution_id = execution_id or str(uuid.uuid4())
        
        process_strategy = request_data.get("process_strategy", {})
        target_type = request_data.get("target_type", {})
        metadata = request_data.get("metadata", {})
        
        # Explicit pipeline execution
        if "pipeline" in process_strategy:
            pipeline = process_strategy["pipeline"]
            tools = [self.tool_key_mapping.get(tool_key, tool_key) for tool_key in pipeline]
            return self.execute_custom_pipeline(tools, request_data, execution_id)
        
        # Default pipeline selection
        topic_name = metadata.get("topic_name")
        file_count = len(request_data.get("files", []))
        is_new_topic = metadata.get("is_new_topic", False)
        
        # Determine input type and context
        input_type = "dialogue" if target_type == "personal_memory" else "document"
        if isinstance(request_data.get("input"), str) and not request_data.get("files"):
            input_type = "text"
        
        pipeline_name = self.select_default_pipeline(
            target_type, topic_name, file_count, is_new_topic, 
            input_type=input_type
        )
        return self.execute_pipeline(pipeline_name, request_data, execution_id)