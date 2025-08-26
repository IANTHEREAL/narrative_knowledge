"""
MemoryGraphBuildTool - Integrates PersonalMemorySystem with the tool-based pipeline architecture.

This tool provides direct integration between memory processing and graph building,
replacing the standalone PersonalMemorySystem.process_chat_batch() with pipeline-compatible processing.
"""

import logging
from typing import Dict, Any, List, Optional

from base import ToolResult
from graph_build_tool import GraphBuildTool

logger = logging.getLogger(__name__)


class MemoryGraphBuildTool(GraphBuildTool):
    """
    Extended GraphBuildTool for memory-specific processing.
    
    This tool integrates PersonalMemorySystem functionality directly into the 
    pipeline architecture, replacing standalone memory processing with 
    tool-based processing.
    """
    
    def __init__(self, session_factory=None, llm_client=None, embedding_func=None):
        super().__init__(session_factory, llm_client, embedding_func)
        self.memory_system = None
        self.llm_client = llm_client
        self.embedding_func = embedding_func
    
    @property
    def tool_name(self) -> str:
        return "MemoryGraphBuildTool"
        
    @property
    def tool_key(self) -> str:
        return "memory_graph_build"

    @property
    def tool_description(self) -> str:
        return "Processes chat messages and builds personal knowledge graph using memory-specific extraction"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["chat_messages", "user_id"],
            "properties": {
                "chat_messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message_content": {"type": "string", "description": "The message content"},
                            "session_id": {"type": "string", "description": "Conversation session ID"},
                            "conversation_title": {"type": "string", "description": "Title of the conversation"},
                            "date": {"type": "string", "description": "ISO format timestamp"},
                            "role": {"type": "string", "enum": ["user", "assistant"], "description": "Message role"}
                        },
                        "required": ["message_content", "role"]
                    },
                    "description": "List of chat messages to process"
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for memory categorization"
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if already processed",
                    "default": False
                },
                "llm_client": {
                    "type": "object",
                    "description": "LLM client instance for processing"
                },
                "embedding_func": {
                    "type": "object",
                    "description": "Embedding function for vector operations"
                }
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User identifier"},
                "topic_name": {"type": "string", "description": "Generated topic name for this user"},
                "source_data_id": {"type": "string", "description": "ID of created SourceData"},
                "knowledge_block_id": {"type": "string", "description": "ID of created KnowledgeBlock"},
                "entities_created": {"type": "integer", "description": "Number of entities created"},
                "relationships_created": {"type": "integer", "description": "Number of relationships created"},
                "triplets_extracted": {"type": "integer", "description": "Number of triplets extracted"},
                "status": {"type": "string", "description": "Processing status"}
            }
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the tool's input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check required fields exist
            required = ["chat_messages", "user_id"]
            for field in required:
                if field not in input_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Validate chat_messages structure
            chat_messages = input_data.get("chat_messages", [])
            if not isinstance(chat_messages, list):
                print("chat_messages must be a list")
                return False
            
            for msg in chat_messages:
                if not isinstance(msg, dict):
                    print("Each chat message must be a dict")
                    return False
                if "message_content" not in msg or "role" not in msg:
                    print("Each message must have message_content and role")
                    return False
                if msg.get("role") not in ["user", "assistant"]:
                    print("Role must be 'user' or 'assistant'")
                    return False
            
            return True
        except Exception as e:
            print(f"MemoryGraphBuildTool validation error: {e}")
            return False

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Process chat messages through memory pipeline and build knowledge graph.
        
        Args:
            input_data: Dictionary containing:
                - chat_messages: List of chat message dicts
                - user_id: User identifier
                - force_reprocess: Whether to force reprocessing
                - llm_client: LLM client instance
                - embedding_func: Embedding function
                
        Returns:
            ToolResult with memory processing results
        """
        try:
            chat_messages = input_data.get("chat_messages", [])
            user_id = input_data.get("user_id")
            force_regenerate = input_data.get("force_regenerate", False)
            
            if not chat_messages:
                return ToolResult(
                    success=False,
                    error_message="No chat messages provided"
                )
            
            if not user_id:
                return ToolResult(
                    success=False,
                    error_message="user_id is required"
                )
            
            # Initialize memory system with provided clients
            llm_client = input_data.get("llm_client", self.llm_client)
            embedding_func = input_data.get("embedding_func", self.embedding_func)
            
            if not llm_client:
                return ToolResult(
                    success=False,
                    error_message="LLM client is required for memory processing"
                )
            
            # Initialize components
            self.llm_client = llm_client
            self.embedding_func = embedding_func
            # self._initialize_components()
            
            # # Initialize memory system
            # memory_system = PersonalMemorySystem(
            #     llm_client=llm_client,
            #     embedding_func=embedding_func,
            #     session_factory=self.session_factory
            # )
            
            # # Generate topic name for this user
            # topic_name = generate_topic_name_for_personal_memory(user_id)
            
            # Check if this exact batch was already processed
            if not force_regenerate:
                existing_result = self._check_existing_processing(chat_messages, user_id)
                if existing_result:
                    return existing_result
            
            # # Process through memory system (creates SourceData, Blueprint, KnowledgeBlock)
            # memory_result = memory_system.process_chat_batch(chat_messages, user_id)
            
            # if not memory_result or "source_id" not in memory_result:
            #     return ToolResult(
            #         success=False,
            #         error_message="Memory system failed to process chat messages"
            #     )
            
            # # Now process the created SourceData through graph building
            # source_data_id = memory_result["source_id"]
            
            # # Get the personal blueprint created by memory system
            # with self.session_factory() as db:
            #     blueprint = db.query(AnalysisBlueprint).filter(
            #         AnalysisBlueprint.topic_name == topic_name
            #     ).order_by(AnalysisBlueprint.created_at.desc()).first()
                
            #     if not blueprint:
            #         return ToolResult(
            #             success=False,
            #             error_message=f"No personal blueprint found for topic: {topic_name}"
            #         )
                
            #     blueprint_id = blueprint.id
            
            # # Process through graph building using the parent class
            # graph_result = self._process_single_document(
            #     blueprint_id, 
            #     source_data_id, 
            #     force_regenerate
            # )
            
            # if not graph_result.success:
            #     return graph_result
            
            # Combine results
            # final_result = {
            #     "user_id": user_id,
            #     "topic_name": topic_name,
            #     "source_data_id": source_data_id,
            #     "knowledge_block_id": memory_result.get("knowledge_block_id"),
            #     "entities_created": graph_result.data.get("entities_created", 0),
            #     "relationships_created": graph_result.data.get("relationships_created", 0),
            #     "triplets_extracted": graph_result.data.get("triplets_extracted", 0),
            #     "status": "completed"
            # }
            print("modeling memory_graph_build_tool!\n")
            return ToolResult(
                success=True,
                data=None,
                metadata={
                    "user_id": user_id,
                    "message_count": len(chat_messages),
                    "topic_name": f"topic_name for {user_id}"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Memory processing failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )

    def _check_existing_processing(self, chat_messages: List[Dict], user_id: str) -> Optional[ToolResult]:
        """
        Check if this exact chat batch was already processed.
        
        Args:
            chat_messages: List of chat messages
            user_id: User identifier
            
        Returns:
            ToolResult if already processed, None otherwise
        """
        try:
            import hashlib
            import json
            
            # # Generate content hash for deduplication
            # content_json = json.dumps(chat_messages, sort_keys=True, ensure_ascii=False)
            # content_hash = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
            
            # topic_name = generate_topic_name_for_personal_memory(user_id)
            
            # with self.session_factory() as db:
            #     # Check for existing SourceData with this content hash
            #     source_data = db.query(SourceData).filter(
            #         SourceData.topic_name == topic_name,
            #         SourceData.attributes.contains({"content_hash": content_hash})
            #     ).first()
                
            #     if source_data and source_data.status == "graph_completed":
            #         return ToolResult(
            #             success=True,
            #             data={
            #                 "user_id": user_id,
            #                 "topic_name": topic_name,
            #                 "source_data_id": source_data.id,
            #                 "entities_created": 0,
            #                 "relationships_created": 0,
            #                 "triplets_extracted": 0,
            #                 "status": "already_processed",
            #                 "reused_existing": True
            #             },
            #             metadata={"user_id": user_id, "message_count": len(chat_messages)}
            #         )
            print("check_existing_processing!\n")
            
            return None
            
        except Exception as e:
            print(f"Error checking existing processing: {e}")
            return None


# Register the tool - will be handled by orchestrator initialization