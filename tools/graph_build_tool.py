"""
GraphBuildTool - Extracts knowledge from documents and adds to the global graph using analysis blueprints:

- Input: A single SourceData object and its topic's AnalysisBlueprint
- Output: New nodes and relationships added to the global knowledge graph
- Maps to: KnowledgeGraphBuilder.build logic, enhanced to use AnalysisBlueprint as context
"""

from typing import Dict, Any, Optional, List
import logging

from tools.base import BaseTool, ToolResult
from knowledge_graph.models import SourceData, AnalysisBlueprint
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from setting.db import SessionLocal


class GraphBuildTool(BaseTool):
    """
    Extracts knowledge from a document and builds the knowledge graph.
    
    Input Schema:
        source_data_id (str): ID of the SourceData to process
        blueprint_id (str): ID of the AnalysisBlueprint to use as context
        force_reprocess (bool, optional): Force reprocessing even if already processed
        llm_client: LLM client instance (required)
        embedding_func: Embedding function (optional)
    
    Output Schema:
        source_data_id (str): ID of processed SourceData
        blueprint_id (str): ID of used AnalysisBlueprint
        entities_created (int): Number of entities created in the graph
        relationships_created (int): Number of relationships created in the graph
        triplets_extracted (int): Number of triplets extracted from the document
        reused_existing (bool): Whether existing processing was reused
        status (str): Processing status
    """
    
    def __init__(self, session_factory=None, llm_client=None, embedding_func=None):
        super().__init__(session_factory=session_factory)
        self.session_factory = session_factory or SessionLocal
        self.llm_client = llm_client
        self.embedding_func = embedding_func
        
        # Initialize components
        self.graph_builder = None
        self.cm_generator = None
    
    def _initialize_components(self):
        """Initialize graph builder and cognitive map generator."""
        if not self.llm_client:
            raise ValueError("LLM client is required for graph building")
        if not self.graph_builder:
            self.graph_builder = NarrativeKnowledgeGraphBuilder(
                self.llm_client, self.embedding_func, self.session_factory
            )
            self.cm_generator = DocumentCognitiveMapGenerator(
                self.llm_client, self.session_factory, worker_count=3
            )

    @property
    def tool_name(self) -> str:
        return "GraphBuildTool"
        
    @property
    def tool_key(self) -> str:
        return "graph_build"

    @property
    def tool_description(self) -> str:
        return "Extracts knowledge from a document and adds it to the global knowledge graph"

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
                    "description": "Force reprocessing even if already processed",
                    "default": False
                },
                "llm_client": {
                    "type": "object",
                    "description": "LLM client instance for graph building"
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
                "source_data_id": {
                    "type": "string",
                    "description": "ID of processed SourceData"
                },
                "blueprint_id": {
                    "type": "string",
                    "description": "ID of used AnalysisBlueprint"
                },
                "entities_created": {
                    "type": "integer",
                    "description": "Number of entities created in the graph"
                },
                "relationships_created": {
                    "type": "integer",
                    "description": "Number of relationships created in the graph"
                },
                "triplets_extracted": {
                    "type": "integer",
                    "description": "Number of triplets extracted from the document"
                },
                "reused_existing": {
                    "type": "boolean",
                    "description": "Whether existing processing was reused"
                },
                "status": {
                    "type": "string",
                    "description": "Processing status"
                }
            }
        }

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Build knowledge graph from documents using blueprint context.
        
        Args:
            input_data: Dictionary containing either:
                - Single document: source_data_id + blueprint_id
                - Batch processing: topic_name (+ optional source_data_ids)
                - force_reprocess: Whether to force reprocessing
                - llm_client: LLM client instance (required)
                - embedding_func: Embedding function (optional)
                
        Returns:
            ToolResult with graph building results
        """
        try:
            force_reprocess = input_data.get("force_reprocess", False)
            
            # Get LLM client from input or use provided one
            llm_client = input_data.get("llm_client", self.llm_client)
            embedding_func = input_data.get("embedding_func", self.embedding_func)
            
            if not llm_client:
                return ToolResult(
                    success=False,
                    error_message="LLM client is required for graph building"
                )
            
            # Initialize components with provided clients
            self.llm_client = llm_client
            self.embedding_func = embedding_func
            self._initialize_components()
            
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
                    input_data.get("source_data_ids"),
                    force_reprocess
                )
            else:
                return ToolResult(
                    success=False,
                    error_message="Invalid input: must provide either (blueprint_id + source_data_id) or topic_name"
                )
                
        except Exception as e:
            self.logger.error(f"Graph building failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def _process_single_document(self, blueprint_id: str, source_data_id: str, force_reprocess: bool = False) -> ToolResult:
        """
        Process a single document with a given blueprint.
        
        Args:
            blueprint_id: ID of the AnalysisBlueprint to use
            source_data_id: ID of the SourceData to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            ToolResult with processing results
        """
        try:
            self.logger.info(f"Starting single document processing: {source_data_id}")
            
            with self.session_factory() as db:
                # Get source data and blueprint
                source_data = db.query(SourceData).filter(
                    SourceData.id == source_data_id
                ).first()
                
                if not source_data:
                    return ToolResult(
                        success=False,
                        error_message=f"SourceData not found: {source_data_id}"
                    )
                
                blueprint = db.query(AnalysisBlueprint).filter(
                    AnalysisBlueprint.id == blueprint_id
                ).first()
                
                if not blueprint:
                    return ToolResult(
                        success=False,
                        error_message=f"AnalysisBlueprint not found: {blueprint_id}"
                    )
                
                if blueprint.status != "ready":
                    return ToolResult(
                        success=False,
                        error_message=f"Blueprint is not ready (status: {blueprint.status})"
                    )
                
                # Check if already processed and not forcing reprocess
                if not force_reprocess and source_data.status == "graph_completed":
                    self.logger.info(f"SourceData already processed: {source_data_id}")
                    return ToolResult(
                        success=True,
                        data={
                            "source_data_id": source_data_id,
                            "blueprint_id": blueprint_id,
                            "reused_existing": True,
                            "status": "already_processed",
                            "entities_created": 0,
                            "relationships_created": 0,
                            "triplets_extracted": 0
                        },
                        metadata={
                            "source_data_name": source_data.name,
                            "topic_name": source_data.topic_name
                        }
                    )
                
                # Update status to processing
                source_data.status = "graph_processing"
                db.commit()
            
            try:
                # Process the document
                result = self._process_document_with_blueprint(source_data, blueprint)
                
                if result.success:
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
                
        except Exception as e:
            self.logger.error(f"Single document processing failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def _process_topic_batch(self, topic_name: str, source_data_ids: Optional[List[str]] = None, force_reprocess: bool = False) -> ToolResult:
        """
        Process a batch of documents for a topic.
        
        Args:
            topic_name: Name of the topic to process
            source_data_ids: Optional list of specific source data IDs to process
            force_reprocess: Whether to force reprocessing
            
        Returns:
            ToolResult with batch processing results
        """
        try:
            self.logger.info(f"Starting batch processing for topic: {topic_name}")
            
            with self.session_factory() as db:
                # Get the latest blueprint for this topic
                blueprint = db.query(AnalysisBlueprint).filter(
                    AnalysisBlueprint.topic_name == topic_name,
                    AnalysisBlueprint.status == "ready"
                ).order_by(AnalysisBlueprint.created_at.desc()).first()
                
                if not blueprint:
                    return ToolResult(
                        success=False,
                        error_message=f"No ready blueprint found for topic: {topic_name}"
                    )
                
                # Get source data for the topic
                query = db.query(SourceData).filter(
                    SourceData.topic_name == topic_name,
                    SourceData.status.in_(["created", "etl_completed", "graph_failed"])
                )
                
                if source_data_ids:
                    query = query.filter(SourceData.id.in_(source_data_ids))
                
                source_data_list = query.all()
                
                if not source_data_list:
                    return ToolResult(
                        success=False,
                        error_message=f"No pending source data found for topic: {topic_name}"
                    )
                
                # Process each document
                results = []
                total_entities = 0
                total_relationships = 0
                total_triplets = 0
                processed_count = 0
                failed_count = 0
                
                for source_data in source_data_list:
                    # Skip if already processed and not forcing reprocess
                    if not force_reprocess and source_data.status == "graph_completed":
                        continue
                    
                    self.logger.info(f"Processing document: {source_data.id}")
                    
                    try:
                        result = self._process_single_document(blueprint.id, source_data.id, force_reprocess)
                        
                        if result.success:
                            processed_count += 1
                            total_entities += result.data.get("entities_created", 0)
                            total_relationships += result.data.get("relationships_created", 0)
                            total_triplets += result.data.get("triplets_extracted", 0)
                            results.append({
                                "source_data_id": source_data.id,
                                "status": "success",
                                "entities_created": result.data.get("entities_created", 0),
                                "relationships_created": result.data.get("relationships_created", 0),
                                "triplets_extracted": result.data.get("triplets_extracted", 0)
                            })
                        else:
                            failed_count += 1
                            results.append({
                                "source_data_id": source_data.id,
                                "status": "failed",
                                "error": result.error_message
                            })
                            
                    except Exception as e:
                        failed_count += 1
                        results.append({
                            "source_data_id": source_data.id,
                            "status": "failed",
                            "error": str(e)
                        })
                
                self.logger.info(
                    f"Batch processing completed for topic {topic_name}: "
                    f"{processed_count} succeeded, {failed_count} failed"
                )
                
                return ToolResult(
                    success=True,
                    data={
                        "topic_name": topic_name,
                        "blueprint_id": blueprint.id,
                        "processed_count": processed_count,
                        "failed_count": failed_count,
                        "total_entities_created": total_entities,
                        "total_relationships_created": total_relationships,
                        "total_triplets_extracted": total_triplets,
                        "results": results
                    },
                    metadata={
                        "topic_name": topic_name,
                        "source_data_count": len(source_data_list)
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )

    def _process_document_with_blueprint(self, source_data: SourceData, blueprint: AnalysisBlueprint) -> ToolResult:
        """
        Core document processing logic used by both single and batch processing.
        
        Args:
            source_data: SourceData record to process
            blueprint: AnalysisBlueprint to use as context
            
        Returns:
            ToolResult with processing results
        """
        try:
            # Convert source data to document format
            document = self._convert_source_data_to_document(source_data)
            
            # Get cognitive map for the document (if exists)
            cognitive_maps = self.cm_generator.get_cognitive_maps_for_topic(
                source_data.topic_name
            )
            document_cognitive_map = None
            
            for cm in cognitive_maps:
                if cm.document_id == source_data.id:
                    # Convert DocumentSummary to cognitive map format
                    try:
                        business_context = cm.business_context or "{}"
                        import ast
                        business_context_dict = ast.literal_eval(business_context) if isinstance(business_context, str) else business_context
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
            
            # Extract triplets using blueprint context
            triplets = self.graph_builder.extract_triplets_from_document(
                source_data.topic_name,
                document,
                blueprint,
                document_cognitive_map
            )
            
            if not triplets:
                self.logger.warning(f"No triplets extracted from document: {source_data.id}")
                return ToolResult(
                    success=True,
                    data={
                        "source_data_id": source_data.id,
                        "blueprint_id": blueprint.id,
                        "triplets_extracted": 0,
                        "entities_created": 0,
                        "relationships_created": 0
                    }
                )
            
            # Convert triplets to graph and save to database
            entities_created, relationships_created = self.graph_builder.convert_triplets_to_graph(
                triplets, source_data.id
            )
            
            self.logger.info(
                f"Document {source_data.id}: extracted {len(triplets)} triplets, "
                f"created {entities_created} entities, {relationships_created} relationships"
            )
            
            return ToolResult(
                success=True,
                data={
                    "source_data_id": source_data.id,
                    "blueprint_id": blueprint.id,
                    "triplets_extracted": len(triplets),
                    "entities_created": entities_created,
                    "relationships_created": relationships_created
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document {source_data.id}: {e}")
            return ToolResult(
                success=False,
                error_message=str(e)
            )

    def _convert_source_data_to_document(self, source_data: SourceData) -> Dict[str, Any]:
        """
        Convert SourceData to document format expected by graph builder.
        
        Args:
            source_data: SourceData record
            
        Returns:
            Document dictionary
        """
        return {
            "source_id": source_data.id,
            "source_name": source_data.name,
            "source_content": source_data.effective_content or "",
            "source_attributes": source_data.attributes or {},
            "source_link": source_data.link,
            "topic_name": source_data.topic_name
        }


# Register the tool
from tools.base import TOOL_REGISTRY
TOOL_REGISTRY.register(GraphBuildTool())