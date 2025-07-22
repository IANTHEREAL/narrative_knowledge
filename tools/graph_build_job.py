"""
Graph Build Job - Build knowledge graph from source data using analysis blueprints
"""

from typing import Any, Dict, List, Optional

from knowledge_graph.models import SourceData, AnalysisBlueprint
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from .base import BaseJob, JobContext, JobResult


class GraphBuildJob(BaseJob):
    """
    Graph Build Job - Extract triplets and build knowledge graph
    
    Input: blueprint_id + source_data_id OR topic_name (for batch processing)
    Output: graph building results (entities created, relationships created)
    """
    
    def __init__(self, session_factory, llm_client=None, embedding_func=None, worker_count: int = 3):
        """
        Initialize graph build job
        
        Args:
            session_factory: Database session factory
            llm_client: LLM client for graph building
            embedding_func: Embedding function for vector operations
            worker_count: Number of workers for parallel processing
        """
        super().__init__(session_factory, llm_client, embedding_func)
        self.worker_count = worker_count
        
        if not llm_client:
            raise ValueError("LLM client is required for graph building")
        
        # Initialize graph builder
        self.graph_builder = NarrativeKnowledgeGraphBuilder(
            llm_client, embedding_func, session_factory
        )
        
        # Initialize cognitive map generator for retrieving existing maps
        self.cm_generator = DocumentCognitiveMapGenerator(
            llm_client, session_factory, worker_count
        )
    
    @property
    def job_type(self) -> str:
        return "graph_build"
    
    @property
    def job_name(self) -> str:
        return "Knowledge Graph Build Job"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "oneOf": [
                {
                    "title": "Single Document Processing",
                    "required": ["blueprint_id", "source_data_id"],
                    "properties": {
                        "blueprint_id": {
                            "type": "string",
                            "description": "ID of the AnalysisBlueprint to use"
                        },
                        "source_data_id": {
                            "type": "string",
                            "description": "ID of the SourceData to process"
                        }
                    }
                },
                {
                    "title": "Batch Topic Processing",
                    "required": ["topic_name"],
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": "Name of the topic to process all pending source data"
                        },
                        "source_data_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of specific source data IDs to process"
                        }
                    }
                }
            ],
            "properties": {
                "force_reprocess": {
                    "type": "boolean",
                    "description": "Whether to force reprocessing even if already completed",
                    "default": False
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "processed_source_data_count": {
                    "type": "integer",
                    "description": "Number of source data records processed"
                },
                "total_triplets_extracted": {
                    "type": "integer",
                    "description": "Total number of triplets extracted"
                },
                "total_entities_created": {
                    "type": "integer",
                    "description": "Total number of entities created in the graph"
                },
                "total_relationships_created": {
                    "type": "integer",
                    "description": "Total number of relationships created in the graph"
                },
                "processing_summary": {
                    "type": "object",
                    "description": "Detailed summary of processing results per document"
                },
                "blueprint_info": {
                    "type": "object",
                    "description": "Information about the blueprint used"
                }
            }
        }
    
    def execute_job_logic(self, context: JobContext) -> JobResult:
        """
        Execute graph building for source data
        
        Args:
            context: Job execution context
            
        Returns:
            JobResult with graph building results
        """
        input_data = context.input_data
        force_reprocess = input_data.get("force_reprocess", False)
        
        # Determine processing mode
        if "blueprint_id" in input_data and "source_data_id" in input_data:
            # Single document processing
            return self._process_single_document(
                input_data["blueprint_id"], 
                input_data["source_data_id"], 
                force_reprocess
            )
        elif "topic_name" in input_data:
            # Batch topic processing
            return self._process_topic_batch(
                input_data["topic_name"],
                input_data.get("source_data_ids", None),
                force_reprocess
            )
        else:
            return JobResult(
                success=False,
                error_message="Invalid input: must provide either (blueprint_id + source_data_id) or topic_name"
            )
    
    def _process_single_document(self, blueprint_id: str, source_data_id: str, force_reprocess: bool) -> JobResult:
        """
        Process a single document with a specific blueprint
        
        Args:
            blueprint_id: ID of the blueprint to use
            source_data_id: ID of the source data to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            JobResult with processing results
        """
        self.logger.info(f"Processing single document: {source_data_id} with blueprint: {blueprint_id}")
        
        # Get blueprint and source data
        with self.session_factory() as db:
            blueprint = db.query(AnalysisBlueprint).filter(
                AnalysisBlueprint.id == blueprint_id
            ).first()
            
            if not blueprint:
                return JobResult(
                    success=False,
                    error_message=f"AnalysisBlueprint not found: {blueprint_id}"
                )
            
            if blueprint.status != "ready":
                return JobResult(
                    success=False,
                    error_message=f"Blueprint is not ready (status: {blueprint.status})"
                )
            
            source_data = db.query(SourceData).filter(
                SourceData.id == source_data_id
            ).first()
            
            if not source_data:
                return JobResult(
                    success=False,
                    error_message=f"SourceData not found: {source_data_id}"
                )
            
            # Check if already processed (unless forcing reprocess)
            if not force_reprocess and source_data.status == "graph_completed":
                self.logger.info(f"SourceData already processed: {source_data_id}")
                return JobResult(
                    success=True,
                    data={
                        "processed_source_data_count": 1,
                        "total_triplets_extracted": 0,  # Not re-extracted
                        "total_entities_created": 0,
                        "total_relationships_created": 0,
                        "processing_summary": {
                            source_data_id: {
                                "status": "already_processed",
                                "reused_existing": True
                            }
                        },
                        "blueprint_info": {
                            "blueprint_id": blueprint.id,
                            "topic_name": blueprint.topic_name
                        }
                    }
                )
            
            # Update status to processing
            source_data.status = "graph_processing"
            db.commit()
        
        try:
            # Process the document
            result = self._process_document_with_blueprint(source_data, blueprint)
            
            # Update status to completed
            with self.session_factory() as db:
                source_data = db.query(SourceData).filter(
                    SourceData.id == source_data_id
                ).first()
                source_data.status = "graph_completed"
                db.commit()
            
            return result
            
        except Exception as e:
            # Update status to failed
            with self.session_factory() as db:
                source_data = db.query(SourceData).filter(
                    SourceData.id == source_data_id
                ).first()
                source_data.status = "graph_failed"
                db.commit()
            
            raise e
    
    def _process_topic_batch(self, topic_name: str, specific_source_data_ids: Optional[List[str]], force_reprocess: bool) -> JobResult:
        """
        Process all pending source data for a topic
        
        Args:
            topic_name: Name of the topic to process
            specific_source_data_ids: Optional specific source data IDs to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            JobResult with batch processing results
        """
        self.logger.info(f"Processing topic batch: {topic_name}")
        
        # Get blueprint for the topic
        with self.session_factory() as db:
            blueprint = db.query(AnalysisBlueprint).filter(
                AnalysisBlueprint.topic_name == topic_name,
                AnalysisBlueprint.status == "ready"
            ).first()
            
            if not blueprint:
                return JobResult(
                    success=False,
                    error_message=f"No ready blueprint found for topic: {topic_name}"
                )
            
            # Get source data to process
            query = db.query(SourceData).filter(SourceData.topic_name == topic_name)
            
            if specific_source_data_ids:
                query = query.filter(SourceData.id.in_(specific_source_data_ids))
            elif not force_reprocess:
                # Only process pending or failed documents
                query = query.filter(SourceData.status.in_(["created", "updated", "graph_pending", "graph_failed"]))
            
            source_data_list = query.all()
            
            if not source_data_list:
                self.logger.info(f"No source data to process for topic: {topic_name}")
                return JobResult(
                    success=True,
                    data={
                        "processed_source_data_count": 0,
                        "total_triplets_extracted": 0,
                        "total_entities_created": 0,
                        "total_relationships_created": 0,
                        "processing_summary": {},
                        "blueprint_info": {
                            "blueprint_id": blueprint.id,
                            "topic_name": blueprint.topic_name
                        }
                    }
                )
        
        # Process all documents
        total_triplets = 0
        total_entities = 0
        total_relationships = 0
        processing_summary = {}
        failed_count = 0
        
        for source_data in source_data_list:
            try:
                # Update status to processing
                with self.session_factory() as db:
                    sd = db.query(SourceData).filter(SourceData.id == source_data.id).first()
                    sd.status = "graph_processing"
                    db.commit()
                
                # Process the document
                result = self._process_document_with_blueprint(source_data, blueprint)
                
                if result.success:
                    # Update status to completed
                    with self.session_factory() as db:
                        sd = db.query(SourceData).filter(SourceData.id == source_data.id).first()
                        sd.status = "graph_completed"
                        db.commit()
                    
                    # Accumulate results
                    doc_summary = result.data["processing_summary"][source_data.id]
                    total_triplets += doc_summary.get("triplets_extracted", 0)
                    total_entities += doc_summary.get("entities_created", 0)
                    total_relationships += doc_summary.get("relationships_created", 0)
                    processing_summary[source_data.id] = doc_summary
                    
                    self.logger.info(f"Successfully processed document: {source_data.id}")
                else:
                    failed_count += 1
                    processing_summary[source_data.id] = {
                        "status": "failed",
                        "error": result.error_message
                    }
                    
                    # Update status to failed
                    with self.session_factory() as db:
                        sd = db.query(SourceData).filter(SourceData.id == source_data.id).first()
                        sd.status = "graph_failed"
                        db.commit()
                    
                    self.logger.error(f"Failed to process document: {source_data.id} - {result.error_message}")
                
            except Exception as e:
                failed_count += 1
                error_msg = f"Unexpected error processing {source_data.id}: {e}"
                processing_summary[source_data.id] = {
                    "status": "failed",
                    "error": error_msg
                }
                
                # Update status to failed
                with self.session_factory() as db:
                    sd = db.query(SourceData).filter(SourceData.id == source_data.id).first()
                    sd.status = "graph_failed"
                    db.commit()
                
                self.logger.error(error_msg, exc_info=True)
        
        success_count = len(source_data_list) - failed_count
        
        self.logger.info(f"Batch processing completed for topic: {topic_name} - {success_count}/{len(source_data_list)} successful")
        
        if failed_count > 0 and success_count == 0:
            return JobResult(
                success=False,
                error_message=f"All {failed_count} documents failed to process",
                data={
                    "processed_source_data_count": len(source_data_list),
                    "total_triplets_extracted": total_triplets,
                    "total_entities_created": total_entities,
                    "total_relationships_created": total_relationships,
                    "processing_summary": processing_summary,
                    "blueprint_info": {
                        "blueprint_id": blueprint.id,
                        "topic_name": blueprint.topic_name
                    }
                }
            )
        
        return JobResult(
            success=True,
            data={
                "processed_source_data_count": len(source_data_list),
                "total_triplets_extracted": total_triplets,
                "total_entities_created": total_entities,
                "total_relationships_created": total_relationships,
                "processing_summary": processing_summary,
                "blueprint_info": {
                    "blueprint_id": blueprint.id,
                    "topic_name": blueprint.topic_name
                }
            }
        )
    
    def _process_document_with_blueprint(self, source_data: SourceData, blueprint: AnalysisBlueprint) -> JobResult:
        """
        Process a single document using the given blueprint
        
        Args:
            source_data: SourceData to process
            blueprint: AnalysisBlueprint to use for processing
            
        Returns:
            JobResult with processing results
        """
        # Convert source data to document format
        document = {
            "source_id": source_data.id,
            "source_name": source_data.name,
            "source_content": source_data.effective_content or "",
            "source_attributes": source_data.attributes or {},
            "source_link": source_data.link,
            "topic_name": source_data.topic_name
        }
        
        # Get cognitive map for the document (if exists)
        cognitive_maps = self.cm_generator.get_cognitive_maps_for_topic(blueprint.topic_name)
        document_cognitive_map = None
        
        for cm in cognitive_maps:
            if cm.document_id == source_data.id:
                # Convert DocumentSummary to cognitive map format
                try:
                    business_context = cm.business_context or "{}"
                    business_context_dict = eval(business_context) if isinstance(business_context, str) else business_context
                except:
                    business_context_dict = {}
                
                document_cognitive_map = {
                    "source_id": cm.document_id,
                    "source_name": source_data.name,
                    "summary": cm.summary_content or "",
                    "key_entities": cm.key_entities or [],
                    "theme_keywords": cm.main_themes or [],
                    "important_timeline": business_context_dict.get("important_timeline", []),
                    "structural_patterns": business_context_dict.get("structural_patterns", "unknown"),
                }
                break
        
        # Extract triplets from document
        triplets = self.graph_builder.extract_triplets_from_document(
            blueprint.topic_name, document, blueprint, document_cognitive_map
        )
        
        if not triplets:
            self.logger.warning(f"No triplets extracted from document: {source_data.id}")
            return JobResult(
                success=True,
                data={
                    "processed_source_data_count": 1,
                    "total_triplets_extracted": 0,
                    "total_entities_created": 0,
                    "total_relationships_created": 0,
                    "processing_summary": {
                        source_data.id: {
                            "status": "completed",
                            "triplets_extracted": 0,
                            "entities_created": 0,
                            "relationships_created": 0,
                            "warning": "No triplets extracted"
                        }
                    },
                    "blueprint_info": {
                        "blueprint_id": blueprint.id,
                        "topic_name": blueprint.topic_name
                    }
                }
            )
        
        # Convert triplets to graph and save to database
        entities_created, relationships_created = self.graph_builder.convert_triplets_to_graph(
            triplets, source_data.id
        )
        
        self.logger.info(f"Document {source_data.id}: extracted {len(triplets)} triplets, "
                        f"created {entities_created} entities, {relationships_created} relationships")
        
        return JobResult(
            success=True,
            data={
                "processed_source_data_count": 1,
                "total_triplets_extracted": len(triplets),
                "total_entities_created": entities_created,
                "total_relationships_created": relationships_created,
                "processing_summary": {
                    source_data.id: {
                        "status": "completed",
                        "triplets_extracted": len(triplets),
                        "entities_created": entities_created,
                        "relationships_created": relationships_created
                    }
                },
                "blueprint_info": {
                    "blueprint_id": blueprint.id,
                    "topic_name": blueprint.topic_name
                }
            }
        ) 