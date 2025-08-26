"""
BlueprintGenerationTool - Creates analysis blueprints for topics:

- Purpose: To create or update the shared context for a topic
- Input: All SourceData associated with a Topic
- Output: A new AnalysisBlueprint for the Topic
- Maps to: New logic that analyzes multiple documents to find shared themes and entities
"""

import hashlib
from typing import Dict, Any, List, Optional, Callable

from tools.base import BaseTool, ToolResult
from knowledge_graph.models import SourceData, AnalysisBlueprint
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from setting.db import SessionLocal
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding


class BlueprintGenerationTool(BaseTool):
    """
    Creates analysis blueprints by analyzing all documents in a topic.

    This tool takes all SourceData for a topic, generates cognitive maps,
    and creates a comprehensive AnalysisBlueprint that serves as shared
    context for processing individual documents within that topic.

    Input Schema:
        topic_name (str): Name of the topic to generate blueprint
        source_data_ids (list[str], optional): Specific source data IDs
        force_regenerate (bool, optional)
        llm_client: (required)
        embedding_func: (optional)

    Output Schema:
        blueprint_id (str): ID of created/updated AnalysisBlueprint
        source_data_version_hash (str): Hash of contributing source data versions
        contributing_source_data_count (int): Number of source data records used
        blueprint_summary (dict): Summary of blueprint content
        reused_existing (bool): Whether existing blueprint was reused
    """

    def __init__(
        self,
        session_factory=None,
        llm_client=LLMInterface("openai", "gpt-4o"),
        embedding_func: Optional[Callable] = None,
    ):
        super().__init__(session_factory=session_factory)
        self.session_factory = session_factory or SessionLocal
        self.llm_client = llm_client
        self.embedding_func = get_text_embedding if embedding_func is None else embedding_func

        # Initialize components
        self.cm_generator: Optional[DocumentCognitiveMapGenerator] = None
        self.graph_builder: Optional[NarrativeKnowledgeGraphBuilder] = None

    def _initialize_components(self):
        """Initialize cognitive map generator and graph builder."""
        if not self.llm_client:
            raise ValueError("LLM client is required for blueprint generation")

        if not self.cm_generator:
            self.cm_generator = DocumentCognitiveMapGenerator(
                self.llm_client, self.session_factory, worker_count=3
            )
            self.graph_builder = NarrativeKnowledgeGraphBuilder(
                self.llm_client, self.embedding_func, self.session_factory
            )

    @property
    def tool_name(self) -> str:
        return "BlueprintGenerationTool"

    @property
    def tool_key(self) -> str:
        return "blueprint_gen"

    @property
    def tool_description(self) -> str:
        return "Creates analysis blueprints by analyzing all documents in a topic"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["topic_name"],
            "properties": {
                "topic_name": {
                    "type": "string",
                    "description": "Name of the topic to generate blueprint for",
                },
                "source_data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific source data IDs to include",
                    "default": None,
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if up-to-date",
                    "default": False,
                },
                "llm_client": {
                    "type": "object",
                    "description": "LLM client instance for blueprint generation",
                },
                "embedding_func": {
                    "type": "object",
                    "description": "Embedding function for vector operations",
                },
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "blueprint_id": {
                    "type": "string",
                    "description": "ID of created/updated AnalysisBlueprint record",
                },
                "source_data_version_hash": {
                    "type": "string",
                    "description": "Hash of contributing source data versions",
                },
                "contributing_source_data_count": {
                    "type": "integer",
                    "description": "Number of source data record that contributed to the blueprint",
                },
                "blueprint_summary": {
                    "type": "object",
                    "description": "Summary of blueprint content",
                    "properties": {
                        "canonical_entities_count": {"type": "integer"},
                        "key_patterns_count": {"type": "integer"},
                        "global_timeline_events": {"type": "integer"},
                        "processing_instructions_length": {"type": "integer"},
                        "cognitive_maps_used": {"type": "integer"},
                    },
                },
                "reused_existing": {
                    "type": "boolean",
                    "description": "Whether existing blueprint was reused",
                },
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters with detailed error information."""
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

        # Validate source_data_ids if provided
        source_data_ids = input_data.get("source_data_ids")
        if source_data_ids is not None:
            if not isinstance(source_data_ids, list):
                self.logger.error(
                    f"Validation error: 'source_data_ids' must be a list, got {type(source_data_ids).__name__}"
                )
                return False
            for i, source_id in enumerate(source_data_ids):
                if not isinstance(source_id, str):
                    self.logger.error(
                        f"Validation error: source_data_ids[{i}] must be a string, got {type(source_id).__name__}"
                    )
                    return False

        # Validate required LLM client

        # Validate embedding function

        return True

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Generate an analysis blueprint for a topic.

        Args:
            input_data: Dictionary containing:
                - topic_name: Name of the topic to generate blueprint for
                - source_data_ids: Optional list of specific source data IDs
                - force_regenerate: Whether to force regeneration
                - llm_client: LLM client instance (required)
                - embedding_func: Embedding function (optional)

        Returns:
            ToolResult with blueprint generation results
        """
        blueprint_id = None  # Ensure blueprint_id is always defined
        try:
            topic_name = input_data["topic_name"]
            source_data_ids = input_data.get("source_data_ids")
            self.logger.info("source_data_ids: %s", source_data_ids)
            force_regenerate_str = input_data.get("force_regenerate", False)

            if force_regenerate_str == "True" or force_regenerate_str == "true":
                force_regenerate = True
            else:
                force_regenerate = False
                    
            self.logger.info(f"Force regenerate? : {force_regenerate}")

            # Initialize components with provided clients
            self._initialize_components()

            self.logger.info(f"Starting blueprint generation for topic: {topic_name}")

            with self.session_factory() as db:
                # Get source data for the topic
                all_source_data = (
                    db.query(SourceData)
                    .filter(SourceData.topic_name == topic_name)
                    .order_by(SourceData.created_at)
                    .all()
                )
                self.logger.info(
                    f"Retrieved {len(all_source_data)} source data records for topic: {topic_name}"
                )
                if source_data_ids:
                    new_source_data_list = (
                        db.query(SourceData)
                        .filter(
                            SourceData.id.in_(source_data_ids),
                        )
                        .order_by(SourceData.created_at)
                        .all()
                    )
                else:
                    new_source_data_list = all_source_data
                existing_ids = {item.id for item in all_source_data}
                for item in new_source_data_list:
                    if item.id not in existing_ids:
                        all_source_data.append(item)
                        self.logger.info(
                            f"Added new source data: {item.id} - {item.name} to all_source_data"
                        )
                self.logger.info(
                    f"Retrieved new_source_data_list: {new_source_data_list}"
                )

                if not new_source_data_list:
                    return ToolResult(
                        success=False,
                        error_message=f"No source data found for topic: {topic_name}",
                    )
                rate: float = len(new_source_data_list) / len(all_source_data)

                # Calculate version hash from all source data versions
                version_input = "|".join(
                    sorted([sd.content_hash for sd in all_source_data])
                )
                version_hash = hashlib.sha256(version_input.encode("utf-8")).hexdigest()

                self.logger.info(
                    f"Calculated version hash for source_data_list: {version_hash}"
                )
                # Ensure components are properly initialized
                if not self.cm_generator or not self.graph_builder:
                    raise ValueError("Required components not initialized")

                if force_regenerate:
                    documents = self._convert_source_data_to_documents(
                        new_source_data_list
                    )

                    self.logger.info(
                        f"successfully converted {len(documents)} new source data records to documents for cognitive map generation"
                    )
                else:
                    documents = self._convert_source_data_to_documents(all_source_data)
                    # Generate cognitive maps for all documents
                    self.logger.info(
                        f"successfully converted {len(documents)} all source data records to documents for cognitive map generation"
                    )
            try:
                  # Ensure cognitive_maps is always defined
                if rate > 0.15:
                    self.logger.info(
                        f"Rate {rate} is greater than 0.15, will generate cognitive maps for all documents"
                    )
                    cognitive_maps = self.cm_generator.batch_generate_cognitive_maps(
                        topic_name, documents, force_regenerate=force_regenerate
                    )
                    self.logger.info(
                        f"successfully generated cognitive maps {cognitive_maps}"
                    )
                else:
                    self.logger.info("Rate is lower than 0.15, skipping cognitive map generation")
                    cognitive_maps: List[Dict] = []
                # Generate analysis blueprint
                blueprint_result = self.graph_builder.generate_analysis_blueprint(
                    topic_name, cognitive_maps, force_regenerate=force_regenerate, rate_limit=rate
                )
                self.logger.info(f"Generated blueprint result: {blueprint_result}")

                # Update blueprint with results
                with self.session_factory() as db:
                    blueprint = (
                        db.query(AnalysisBlueprint)
                        .filter(AnalysisBlueprint.topic_name == topic_name)
                        .first()
                    )

                    blueprint.status = "ready"
                    blueprint.source_data_version_hash = version_hash
                    blueprint.contributing_source_data_ids = [
                        doc["source_id"] for doc in documents
                    ]
                    blueprint.error_message = None
                    blueprint_id = blueprint.id
                    processing_items = blueprint.processing_items
                    processing_instructions = blueprint.processing_instructions
                    db.commit()

                # Prepare summary

                summary = {
                    "canonical_entities_count": len(
                        processing_items.get("canonical_entities", {})
                    ),
                    "key_patterns_count": len(processing_items.get("key_patterns", {})),
                    "global_timeline_events": len(
                        processing_items.get("global_timeline", [])
                    ),
                    "processing_instructions_length": len(str(processing_instructions)),
                    "cognitive_maps_used": len(cognitive_maps),
                }

                self.logger.info(
                    f"Blueprint generation completed for topic: {topic_name}"
                )

                return ToolResult(
                    success=True,
                    data={
                        "blueprint_id": blueprint_id,
                        "reused_existing": False,
                        "contributing_source_data_count": len(new_source_data_list),
                        "source_data_version_hash": version_hash,
                        "blueprint_summary": summary,
                    },
                    metadata={
                        "topic_name": topic_name,
                        "source_data_count": len(all_source_data),
                        "cognitive_maps_count": len(cognitive_maps),
                    },
                )
            except Exception as e:
                # Update blueprint status to failed only if blueprint_id is set
                if blueprint_id is not None:
                    with self.session_factory() as db:
                        blueprint = (
                            db.query(AnalysisBlueprint)
                            .filter(AnalysisBlueprint.id == blueprint_id)
                            .first()
                        )
                        if blueprint:
                            blueprint.status = "failed"
                            blueprint.error_message = str(e)
                            db.commit()

                raise e
                raise e

        except Exception as e:
            self.logger.error(f"Blueprint generation failed: {e}")
            return ToolResult(success=False, error_message=str(e))

    def _convert_source_data_to_documents(
        self, source_data_list: List[SourceData]
    ) -> List[Dict[str, Any]]:
        """
        Convert SourceData records to document format for cognitive map generation.

        Args:
            source_data_list: List of SourceData records

        Returns:
            List of document dictionaries
        """
        documents = []

        for source_data in source_data_list:
            document = {
                "source_id": source_data.id,
                "source_name": source_data.name,
                "source_content": source_data.effective_content or "",
                "source_attributes": source_data.attributes or {},
                "source_link": source_data.link,
                "topic_name": source_data.topic_name,
            }
            documents.append(document)

        return documents


# Register the tool - will be handled by orchestrator initialization
