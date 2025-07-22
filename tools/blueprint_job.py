"""
Blueprint Generation Job - Generate analysis blueprints from multiple source documents for a topic
"""

import hashlib
import json
from typing import Any, Dict, List

from knowledge_graph.models import SourceData, AnalysisBlueprint
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from .base import BaseJob, JobContext, JobResult


class BlueprintGenerationJob(BaseJob):
    """
    Blueprint Generation Job - Create analysis blueprints for topics
    
    Input: topic_name, optional force_regenerate
    Output: blueprint_id, blueprint content summary
    """
    
    def __init__(self, session_factory, llm_client=None, embedding_func=None, worker_count: int = 3):
        """
        Initialize blueprint generation job
        
        Args:
            session_factory: Database session factory
            llm_client: LLM client for blueprint generation
            embedding_func: Embedding function (optional)
            worker_count: Number of workers for parallel processing
        """
        super().__init__(session_factory, llm_client, embedding_func)
        self.worker_count = worker_count
        
        if not llm_client:
            raise ValueError("LLM client is required for blueprint generation")
        
        # Initialize cognitive map generator
        self.cm_generator = DocumentCognitiveMapGenerator(
            llm_client, session_factory, worker_count
        )
        
        # Initialize graph builder for blueprint generation
        self.graph_builder = NarrativeKnowledgeGraphBuilder(
            llm_client, embedding_func, session_factory
        )
    
    @property
    def job_type(self) -> str:
        return "blueprint_generation"
    
    @property
    def job_name(self) -> str:
        return "Analysis Blueprint Generation Job"
    
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
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Whether to force regeneration of blueprint even if up-to-date",
                    "default": False
                },
                "source_data_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific source data IDs to include (if not provided, uses all for topic)"
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
                    "description": "ID of the created/updated AnalysisBlueprint record"
                },
                "source_data_version_hash": {
                    "type": "string",
                    "description": "Hash of all source data versions that contributed to this blueprint"
                },
                "contributing_source_data_count": {
                    "type": "integer",
                    "description": "Number of source data records that contributed to the blueprint"
                },
                "cognitive_maps_generated": {
                    "type": "integer",
                    "description": "Number of cognitive maps generated"
                },
                "blueprint_summary": {
                    "type": "object",
                    "description": "Summary of the generated blueprint content"
                }
            }
        }
    
    def execute_job_logic(self, context: JobContext) -> JobResult:
        """
        Execute blueprint generation for a topic
        
        Args:
            context: Job execution context containing topic_name
            
        Returns:
            JobResult with blueprint generation results
        """
        topic_name = context.input_data["topic_name"]
        force_regenerate = context.input_data.get("force_regenerate", False)
        specific_source_data_ids = context.input_data.get("source_data_ids", None)
        
        self.logger.info(f"Starting blueprint generation for topic: {topic_name}")
        
        # Get source data for the topic
        source_data_list, source_data_version_hash = self._get_source_data_for_topic(
            topic_name, specific_source_data_ids
        )
        
        if not source_data_list:
            return JobResult(
                success=False,
                error_message=f"No source data found for topic: {topic_name}"
            )
        
        # Check if blueprint is up-to-date (unless forcing regeneration)
        if not force_regenerate:
            up_to_date_blueprint = self._check_blueprint_up_to_date(
                topic_name, source_data_version_hash
            )
            if up_to_date_blueprint:
                self.logger.info(f"Blueprint is already up-to-date for topic: {topic_name}")
                return JobResult(
                    success=True,
                    data={
                        "blueprint_id": up_to_date_blueprint.id,
                        "source_data_version_hash": up_to_date_blueprint.source_data_version_hash,
                        "contributing_source_data_count": len(up_to_date_blueprint.contributing_source_data_ids),
                        "cognitive_maps_generated": 0,  # No new maps generated
                        "blueprint_summary": {
                            "status": "already_up_to_date",
                            "reused_existing": True
                        }
                    }
                )
        
        # Update blueprint status to generating
        with self.session_factory() as db:
            blueprint = db.query(AnalysisBlueprint).filter(
                AnalysisBlueprint.topic_name == topic_name
            ).first()
            
            if blueprint:
                blueprint.status = "generating"
                blueprint.error_message = None
            else:
                blueprint = AnalysisBlueprint(
                    topic_name=topic_name,
                    status="generating",
                    source_data_version_hash="",
                    contributing_source_data_ids=[]
                )
                db.add(blueprint)
            
            db.commit()
            db.refresh(blueprint)
            blueprint_id = blueprint.id
        
        try:
            # Convert source data to document format for processing
            documents = self._convert_source_data_to_documents(source_data_list)
            
            # Generate cognitive maps
            self.logger.info(f"Generating cognitive maps for {len(documents)} documents")
            cognitive_maps = self.cm_generator.batch_generate_cognitive_maps(
                topic_name, documents, force_regenerate=force_regenerate
            )
            
            if not cognitive_maps:
                raise ValueError(f"Failed to generate cognitive maps for topic: {topic_name}")
            
            # Generate analysis blueprint
            self.logger.info(f"Generating analysis blueprint from {len(cognitive_maps)} cognitive maps")
            blueprint_result = self.graph_builder.generate_analysis_blueprint(
                topic_name, cognitive_maps, force_regenerate=True  # Always regenerate when called from job
            )
            
            # Update blueprint record with results
            with self.session_factory() as db:
                blueprint = db.query(AnalysisBlueprint).filter(
                    AnalysisBlueprint.id == blueprint_id
                ).first()
                
                blueprint.status = "ready"
                blueprint.processing_items = blueprint_result.processing_items
                blueprint.processing_instructions = blueprint_result.processing_instructions
                blueprint.source_data_version_hash = source_data_version_hash
                blueprint.contributing_source_data_ids = [sd.id for sd in source_data_list]
                blueprint.error_message = None
                
                db.commit()
                db.refresh(blueprint)
            
            # Prepare blueprint summary
            blueprint_summary = {
                "status": "completed",
                "canonical_entities_count": len(blueprint_result.processing_items.get("canonical_entities", {}) if blueprint_result.processing_items else {}),
                "key_patterns_count": len(blueprint_result.processing_items.get("key_patterns", {}) if blueprint_result.processing_items else {}),
                "global_timeline_events": len(blueprint_result.processing_items.get("global_timeline", []) if blueprint_result.processing_items else []),
                "processing_instructions_length": len(blueprint_result.processing_instructions or ""),
                "cognitive_maps_used": len(cognitive_maps)
            }
            
            self.logger.info(f"Blueprint generation completed successfully for topic: {topic_name}")
            
            return JobResult(
                success=True,
                data={
                    "blueprint_id": blueprint.id,
                    "source_data_version_hash": source_data_version_hash,
                    "contributing_source_data_count": len(source_data_list),
                    "cognitive_maps_generated": len(cognitive_maps),
                    "blueprint_summary": blueprint_summary
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
    
    def _get_source_data_for_topic(self, topic_name: str, specific_source_data_ids: List[str] = None) -> tuple[List[SourceData], str]:
        """
        Get all source data for a topic and calculate version hash
        
        Args:
            topic_name: Topic name to get source data for
            specific_source_data_ids: Optional list of specific source data IDs
            
        Returns:
            Tuple of (source_data_list, version_hash)
        """
        with self.session_factory() as db:
            query = db.query(SourceData).filter(SourceData.topic_name == topic_name)
            
            if specific_source_data_ids:
                query = query.filter(SourceData.id.in_(specific_source_data_ids))
            
            source_data_list = query.order_by(SourceData.created_at).all()
            
            # Calculate version hash from all content versions
            if source_data_list:
                version_input = "|".join(sorted([sd.content_version for sd in source_data_list]))
                version_hash = hashlib.sha256(version_input.encode("utf-8")).hexdigest()
            else:
                version_hash = ""
            
            return source_data_list, version_hash
    
    def _check_blueprint_up_to_date(self, topic_name: str, source_data_version_hash: str) -> AnalysisBlueprint:
        """
        Check if existing blueprint is up-to-date with current source data
        
        Args:
            topic_name: Topic name
            source_data_version_hash: Hash of current source data versions
            
        Returns:
            AnalysisBlueprint if up-to-date, None otherwise
        """
        with self.session_factory() as db:
            blueprint = db.query(AnalysisBlueprint).filter(
                AnalysisBlueprint.topic_name == topic_name,
                AnalysisBlueprint.status == "ready",
                AnalysisBlueprint.source_data_version_hash == source_data_version_hash
            ).first()
            
            return blueprint
    
    def _convert_source_data_to_documents(self, source_data_list: List[SourceData]) -> List[Dict]:
        """
        Convert SourceData records to document format expected by cognitive map generator
        
        Args:
            source_data_list: List of SourceData records
            
        Returns:
            List of document dicts
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