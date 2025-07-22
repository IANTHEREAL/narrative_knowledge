"""
BlueprintGenerationTool - Creates analysis blueprints for topics:

- Purpose: To create or update the shared context for a topic
- Input: All SourceData associated with a Topic
- Output: A new AnalysisBlueprint for the Topic
- Maps to: New logic that analyzes multiple documents to find shared themes and entities
"""

import hashlib
from typing import Dict, Any, List, Optional
import logging

from tools.base import BaseTool, ToolResult
from knowledge_graph.models import SourceData, AnalysisBlueprint
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from setting.db import SessionLocal


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
    
    def __init__(self, session_factory=None, llm_client=None, embedding_func=None):
        super().__init__(session_factory=session_factory)
        self.session_factory = session_factory or SessionLocal
        self.llm_client = llm_client
        self.embedding_func = embedding_func
        
        # Initialize components
        self.cm_generator = None
        self.graph_builder = None
    
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
                    "description": "Name of the topic to generate blueprint for"
                },
                "source_data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific source data IDs to include",
                    "default": None
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if up-to-date",
                    "default": False
                },
                "llm_client": {
                    "type": "object",
                    "description": "LLM client instance for blueprint generation"
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
                "blueprint_id": {
                    "type": "string",
                    "description": "ID of created/updated AnalysisBlueprint record"
                },
                "source_data_version_hash": {
                    "type": "string",
                    "description": "Hash of contributing source data versions"
                },
                "contributing_source_data_count": {
                    "type": "integer",
                    "description": "Number of source data record that contributed to the blueprint"
                },
                "blueprint_summary": {
                    "type": "object",
                    "description": "Summary of blueprint content",
                    "properties": {
                        "canonical_entities_count": {"type": "integer"},
                        "key_patterns_count": {"type": "integer"},
                        "global_timeline_events": {"type": "integer"},
                        "processing_instructions_length": {"type": "integer"},
                        "cognitive_maps_used": {"type": "integer"}
                    }
                },
                "reused_existing": {
                    "type": "boolean",
                    "description": "Whether existing blueprint was reused"
                }
            }
        }

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
        try:
            topic_name = input_data["topic_name"]
            source_data_ids = input_data.get("source_data_ids")
            force_regenerate = input_data.get("force_regenerate", False)
            
            # Get LLM client from input or use provided one
            llm_client = input_data.get("llm_client", self.llm_client)
            embedding_func = input_data.get("embedding_func", self.embedding_func)
            
            if not llm_client:
                return ToolResult(
                    success=False,
                    error_message="LLM client is required for blueprint generation"
                )
            
            # Initialize components with provided clients
            self.llm_client = llm_client
            self.embedding_func = embedding_func
            self._initialize_components()
            
            self.logger.info(f"Starting blueprint generation for topic: {topic_name}")
            
            with self.session_factory() as db:
                # Get source data for the topic
                query = db.query(SourceData).filter(SourceData.topic_name == topic_name)
                
                if source_data_ids:
                    query = query.filter(SourceData.id.in_(source_data_ids))
                
                source_data_list = query.order_by(SourceData.created_at).all()
                
                if not source_data_list:
                    return ToolResult(
                        success=False,
                        error_message=f"No source data found for topic: {topic_name}"
                    )
                
                # Calculate version hash from all source data versions
                version_input = "|".join(sorted([sd.content_version for sd in source_data_list]))
                version_hash = hashlib.sha256(version_input.encode("utf-8")).hexdigest()
                
                # Check if blueprint is up-to-date (unless forcing regeneration)
                if not force_regenerate:
                    existing_blueprint = db.query(AnalysisBlueprint).filter(
                        AnalysisBlueprint.topic_name == topic_name,
                        AnalysisBlueprint.status == "ready",
                        AnalysisBlueprint.source_data_version_hash == version_hash
                    ).first()
                    
                    if existing_blueprint:
                        self.logger.info(f"Blueprint already up-to-date for topic: {topic_name}")
                        return ToolResult(
                            success=True,
                            data={
                                "blueprint_id": existing_blueprint.id,
                                "reused_existing": True,
                                "contributing_source_data_count": len(source_data_list),
                                "source_data_version_hash": version_hash
                            },
                            metadata={
                                "topic_name": topic_name,
                                "source_data_count": len(source_data_list)
                            }
                        )
                
                # Create or update blueprint record
                blueprint = db.query(AnalysisBlueprint).filter(
                    AnalysisBlueprint.topic_name == topic_name
                ).first()
                
                if not blueprint:
                    blueprint = AnalysisBlueprint(
                        topic_name=topic_name,
                        status="generating",
                        source_data_version_hash="",
                        contributing_source_data_ids=[]
                    )
                    db.add(blueprint)
                    db.flush()
                else:
                    blueprint.status = "generating"
                    blueprint.error_message = None
                
                db.commit()
                blueprint_id = blueprint.id
            
            try:
                # Convert source data to document format
                documents = self._convert_source_data_to_documents(source_data_list)
                
                # Generate cognitive maps for all documents
                self.logger.info(f"Generating cognitive maps for {len(documents)} documents")
                cognitive_maps = self.cm_generator.batch_generate_cognitive_maps(
                    topic_name, documents, force_regenerate=force_regenerate
                )
                
                if not cognitive_maps:
                    raise ValueError(f"Failed to generate cognitive maps for topic: {topic_name}")
                
                # Generate analysis blueprint
                self.logger.info(f"Generating analysis blueprint from {len(cognitive_maps)} cognitive maps")
                blueprint_result = self.graph_builder.generate_analysis_blueprint(
                    topic_name, cognitive_maps, force_regenerate=True
                )
                
                # Update blueprint with results
                with self.session_factory() as db:
                    blueprint = db.query(AnalysisBlueprint).filter(
                        AnalysisBlueprint.id == blueprint_id
                    ).first()
                    
                    blueprint.status = "ready"
                    blueprint.processing_items = blueprint_result.processing_items
                    blueprint.processing_instructions = blueprint_result.processing_instructions
                    blueprint.source_data_version_hash = version_hash
                    blueprint.contributing_source_data_ids = [sd.id for sd in source_data_list]
                    blueprint.error_message = None
                    
                    db.commit()
                
                # Prepare summary
                processing_items = blueprint_result.processing_items or {}
                summary = {
                    "canonical_entities_count": len(processing_items.get("canonical_entities", {})),
                    "key_patterns_count": len(processing_items.get("key_patterns", {})),
                    "global_timeline_events": len(processing_items.get("global_timeline", [])),
                    "processing_instructions_length": len(blueprint_result.processing_instructions or ""),
                    "cognitive_maps_used": len(cognitive_maps)
                }
                
                self.logger.info(f"Blueprint generation completed for topic: {topic_name}")
                
                return ToolResult(
                    success=True,
                    data={
                        "blueprint_id": blueprint_id,
                        "reused_existing": False,
                        "contributing_source_data_count": len(source_data_list),
                        "source_data_version_hash": version_hash,
                        "blueprint_summary": summary
                    },
                    metadata={
                        "topic_name": topic_name,
                        "source_data_count": len(source_data_list),
                        "cognitive_maps_count": len(cognitive_maps)
                    }
                )
                
            except Exception as e:
                # Update blueprint status to failed
                with self.session_factory() as db:
                    blueprint = db.query(AnalysisBlueprint).filter(
                        AnalysisBlueprint.id == blueprint_id
                    ).first()
                    if blueprint:
                        blueprint.status = "failed"
                        blueprint.error_message = str(e)
                        db.commit()
                
                raise e
                
        except Exception as e:
            self.logger.error(f"Blueprint generation failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def _convert_source_data_to_documents(self, source_data_list: List[SourceData]) -> List[Dict[str, Any]]:
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
                "topic_name": source_data.topic_name
            }
            documents.append(document)
        
        return documents


# Register the tool
from tools.base import TOOL_REGISTRY
TOOL_REGISTRY.register(BlueprintGenerationTool())