"""
Base tool interface and shared utilities for the tool-based architecture.

Tools are stateless, independent functions that perform single,
well-defined tasks in the pipeline design.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json
from datetime import datetime, timezone
import uuid
from enum import Enum

from setting.db import SessionLocal


class ExecutionStatus(Enum):
    """Tool execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolResult:
    """Result object returned by tools."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ):
        self.success = success
        self.data = data or {}
        self.error_message = error_message
        self.metadata = metadata or {}
        self.execution_id = execution_id
        self.duration_seconds = duration_seconds
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "execution_id": self.execution_id,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp
        }


class BaseTool(ABC):
    """
    Base class for all tools in the knowledge graph construction pipeline.
    
    Tools are stateless, independent functions that perform a single, well-defined task.
    Enhanced with job-like execution tracking and state management.
    
    1. DocumentETLTool - Processes raw documents into structured SourceData
    2. BlueprintGenerationTool - Creates analysis blueprints from topic documents
    3. GraphBuildTool - Extracts knowledge and builds the knowledge graph
    """
    
    def __init__(self, logger_name: Optional[str] = None, session_factory=None):
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)
        self.session_factory = session_factory
    
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Human-readable name of the tool."""
        pass
        
    @property
    def tool_key(self) -> str:
        """Short key identifier for pipeline configuration."""
        return self.tool_name.lower().replace("tool", "").replace(" ", "_")
    
    @property
    @abstractmethod
    def tool_description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        JSON Schema for input validation.
        Override in subclasses to provide specific validation schema.
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        JSON Schema for output validation.
        Override in subclasses to provide specific output schema.
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given input data.
        
        Args:
            input_data: Dictionary containing tool-specific parameters
            
        Returns:
            ToolResult with execution results
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the tool's input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            import jsonschema
            jsonschema.validate(input_data, self.input_schema)
            return True
        except ImportError:
            # Fallback to basic validation if jsonschema not available
            required = self.input_schema.get("required", [])
            for field in required:
                if field not in input_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            return True
        except Exception:
            return False
    
    def get_required_inputs(self) -> List[str]:
        """
        Return list of required input parameters.
        
        Returns:
            List of required parameter names
        """
        return self.input_schema.get("required", [])
    
    def get_optional_inputs(self) -> List[str]:
        """
        Return list of optional input parameters.
        
        Returns:
            List of optional parameter names
        """
        properties = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))
        return [k for k in properties.keys() if k not in required]
    
    def execute_with_tracking(self, input_data: Dict[str, Any], 
                            execution_id: Optional[str] = None) -> ToolResult:
        """
        Execute the tool with execution tracking (job-like).
        
        Args:
            input_data: Dictionary containing tool-specific parameters
            execution_id: Optional execution ID for tracking
            
        Returns:
            ToolResult with execution results and tracking info
        """
        
        execution_id = execution_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc).isoformat()
        
        self.logger.info(f"Starting tool execution: {self.tool_name} ({execution_id})")
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                return ToolResult(
                    success=False,
                    error_message="Input validation failed",
                    execution_id=execution_id
                )
            
            # Execute tool
            result = self.execute(input_data)
            
            # Add tracking info
            end_time = datetime.now(timezone.utc).isoformat()
            result.execution_id = execution_id
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Tool execution completed: {self.tool_name} ({execution_id}) in {result.duration_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = datetime.now(timezone.utc).isoformat()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Tool execution failed: {self.tool_name} ({execution_id}) - {e}")
            
            return ToolResult(
                success=False,
                error_message=str(e),
                execution_id=execution_id,
                duration_seconds=duration
            )


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools = {}
    
    def register(self, tool: BaseTool):
        """Register a tool with the registry."""
        self._tools[tool.tool_name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def execute_tool(self, name: str, input_data: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given input."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error_message=f"Tool '{name}' not found"
            )
        
        # Validate required inputs
        required = tool.get_required_inputs()
        missing = [req for req in required if req not in input_data]
        if missing:
            return ToolResult(
                success=False,
                error_message=f"Missing required inputs: {', '.join(missing)}"
            )
        
        return tool.execute(input_data)


# Global tool registry
TOOL_REGISTRY = ToolRegistry()