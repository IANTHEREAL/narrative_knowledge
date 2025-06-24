import json
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge_graph.models import DocumentSummary, AnalysisBlueprint
from knowledge_graph.congnitive_map import DocumentCognitiveMapGenerator
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from llm.factory import LLMInterface
from setting.db import SessionLocal


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder using document summaries.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        embedding_func,
        session_factory=None,
        worker_count: int = 3,
    ):
        """
        Initialize the iterative builder.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.embedding_func = embedding_func
        self.session_factory = session_factory or SessionLocal
        self.cm_generator = DocumentCognitiveMapGenerator(
            llm_client, session_factory, worker_count
        )
        self.graph_builder = NarrativeKnowledgeGraphBuilder(
            llm_client, embedding_func, session_factory
        )
        self.worker_count = worker_count

    def build_knowledge_graph(
        self,
        topic_name: str,
        documents: List[Dict],
        force_regenerate_cognitive_maps: bool = False,
        force_regenerate_blueprint: bool = False,
    ) -> Dict:
        """
        Build knowledge graph using iterative approach with document summaries.

        Args:
            topic_name: Topic to focus analysis on
            documents: List of document dicts with knowledge blocks
            force_regenerate_cognitive_maps: Whether to regenerate existing cognitive maps
            force_regenerate_blueprint: Whether to regenerate existing blueprint

        Returns:
            Dict with construction results and statistics
        """
        logger.info(
            f"Building narrative knowledge graph for topic: {topic_name}: {len(documents)} documents"
        )

        # Pre-processing Stage 1: Generate document cognitive maps
        logger.info("=== Stage 0: Generating document cognitive maps ===")
        cognitive_maps = self.generate_document_cognitive_maps(
            topic_name, documents, force_regenerate_cognitive_maps
        )

        # Pre-processing Stage 2: Generate analysis blueprint
        blueprint = self.graph_builder.generate_analysis_blueprint(
            topic_name,
            cognitive_maps,
            force_regenerate_blueprint,
        )

        # Extract narrative triplets
        logger.info(
            f"Processing {len(documents)} documents in parallel using {self.worker_count} workers"
        )

        def extract_and_convert_worker(doc_with_index):
            """Worker function to extract triplets and immediately convert to graph for a single document."""
            index, doc = doc_with_index
            try:
                # Get document cognitive map for enhanced extraction
                doc_cognitive_map = None
                for cognitive_map in cognitive_maps:
                    if cognitive_map.get("source_id") == doc["source_id"]:
                        doc_cognitive_map = cognitive_map
                        break

                # Extract triplets with cognitive map if available
                triplets = self.graph_builder.extract_triplets_from_document(
                    topic_name, doc, blueprint, doc_cognitive_map
                )

                logger.info(
                    f"Document {doc['source_name']}: extracted {len(triplets)} triplets"
                )

                # Immediately convert triplets to graph and save to database
                entities_created, relationships_created = (
                    self.graph_builder.convert_triplets_to_graph(
                        triplets, doc["source_id"]
                    )
                )

                logger.info(
                    f"Document {doc['source_name']}: successfully saved {entities_created} entities, {relationships_created} relationships to database"
                )

                return (
                    index,
                    doc,
                    len(triplets),
                    entities_created,
                    relationships_created,
                    None,
                )
            except Exception as e:
                error_msg = f"Failed to process document {doc['source_name']}: {e}"
                logger.error(error_msg, exc_info=True)
                return index, doc, 0, 0, 0, error_msg

        # Process documents in parallel with immediate database saves
        processing_results = [None] * len(documents)  # Pre-allocate to maintain order
        processing_errors = []

        indexed_documents = list(enumerate(documents))

        all_triplets = 0
        entities_created = 0
        relationships_created = 0

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(
                    extract_and_convert_worker, doc_with_index
                ): doc_with_index[1]
                for doc_with_index in indexed_documents
            }

            completed_processing = 0
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                completed_processing += 1
                try:
                    (
                        index,
                        doc_result,
                        triplets_count,
                        doc_entities,
                        doc_relationships,
                        error,
                    ) = future.result()

                    if error:
                        processing_errors.append(error)
                        logger.warning(
                            f"Document processing failed ({completed_processing}/{len(documents)}): {doc['source_name']}"
                        )
                    else:
                        processing_results[index] = doc_result
                        # Accumulate counts
                        all_triplets += triplets_count
                        entities_created += doc_entities
                        relationships_created += doc_relationships

                        logger.info(
                            f"Document processing completed ({completed_processing}/{len(documents)}): {doc['source_name']}"
                        )

                except Exception as e:
                    error_msg = f"Unexpected error processing document {doc['source_name']}: {e}"
                    processing_errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

        # Log processing summary
        successful_documents = len([r for r in processing_results if r is not None])

        if processing_errors:
            logger.warning(
                f"Document processing completed with {len(processing_errors)} errors:"
            )
            for error in processing_errors:
                logger.warning(f"  - {error}")

            logger.info(
                f"Successfully processed {successful_documents}/{len(documents)} documents"
            )
        else:
            logger.info(f"All {successful_documents} documents processed successfully")

        logger.info(f"Total triplets extracted: {all_triplets} (all semantic triplets)")

        # Compile results
        result = {
            "topic_name": topic_name,
            "blueprint_id": blueprint.id,
            "documents_processed": successful_documents,
            "documents_failed": len(processing_errors),
            "cognitive_maps_generated": len(cognitive_maps),
            "triplets_extracted": all_triplets,
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "global_blueprint": {
                "processing_instructions": blueprint.processing_instructions,
                "processing_items": blueprint.processing_items,
            },
        }

        logger.info(
            f"Iterative knowledge graph construction completed! Results: {result}"
        )

        # Check if we should raise an exception for processing errors
        if processing_errors:
            raise RuntimeError(
                f"Document processing failed with {len(processing_errors)} errors. "
                f"Successfully processed {successful_documents}/{len(documents)} documents. "
                f"Results: {result}"
            )

        return result

    def generate_document_cognitive_maps(
        self,
        topic_name: str,
        documents: List[Dict],
        force_regenerate: bool = False,
    ) -> List[Dict]:
        """
        Generate topic-focused cognitive maps for all documents.
        Returns cognitive maps in document-like format for blueprint generation.

        Args:
            topic_name: Topic to focus cognitive maps on
            documents: List of document dicts
            force_regenerate: Whether to regenerate existing cognitive maps

        Returns:
            List of document-like cognitive map objects
        """
        return self.cm_generator.batch_generate_cognitive_maps(
            topic_name, documents, force_regenerate
        )

    def get_topic_cognitive_maps(self, topic_name: str) -> List[DocumentSummary]:
        """
        Get all existing cognitive maps for a topic.

        Args:
            topic_name: Topic name to filter by

        Returns:
            List of DocumentSummary objects with cognitive maps
        """
        return self.cm_generator.get_cognitive_maps_for_topic(topic_name)

    def get_global_blueprint_details(self, topic_name: str) -> Dict:
        """
        Get detailed information about the global blueprint for a topic.

        Args:
            topic_name: Topic name to get blueprint for

        Returns:
            Dict containing detailed blueprint information
        """
        with self.session_factory() as db:
            blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if not blueprint:
                return {"error": f"No blueprint found for topic: {topic_name}"}

            return {
                "topic_name": blueprint.topic_name,
                "blueprint_id": blueprint.id,
                "created_at": (
                    blueprint.created_at.isoformat() if blueprint.created_at else None
                ),
                "processing_instructions": blueprint.processing_instructions,
                "processing_items": blueprint.processing_items,
            }

    def enhance_knowledge_graph(
        self,
        topic_name: str,
        documents: List[Dict],
    ) -> Dict:
        """
        Build knowledge graph with optional reasoning enhancement for discovering implicit knowledge.

        This method first queries existing blueprint and cognitive maps from the database, followed by detective-style reasoning enhancement.

        Args:
            topic_name: Topic to focus analysis on
            documents: List of document dicts with knowledge blocks

        Returns:
            Dict with construction results, statistics, and reasoning insights
        """
        logger.info(
            f"Building enhanced knowledge graph with reasoning for topic: {topic_name}: {len(documents)} documents"
        )

        # Stage 1: Query existing blueprint from database
        logger.info("=== Stage 1: Querying existing blueprint ===")
        blueprint = None
        with self.session_factory() as db:
            blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if blueprint:
                logger.info(
                    f"Found existing blueprint for {topic_name}, created at {blueprint.created_at}"
                )

        # Stage 2: Query existing cognitive maps from database
        logger.info("=== Stage 2: Querying existing cognitive maps ===")
        cognitive_maps = []
        document_ids = [doc["source_id"] for doc in documents]

        if document_ids:
            with self.session_factory() as db:
                # Query all cognitive maps for the topic and documents in one query
                cognitive_map_summaries = (
                    db.query(DocumentSummary)
                    .filter(
                        DocumentSummary.document_id.in_(document_ids),
                        DocumentSummary.topic_name == topic_name,
                    )
                    .all()
                )

                # Convert DocumentSummary objects to cognitive map format
                for summary in cognitive_map_summaries:
                    try:
                        business_context = (
                            json.loads(summary.business_context)
                            if summary.business_context
                            else {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        business_context = {}

                    cognitive_map = {
                        "source_id": summary.document_id,
                        "source_name": (
                            summary.source_data.name
                            if summary.source_data
                            else f"doc_{summary.document_id}"
                        ),
                        "summary": summary.summary_content or "",
                        "key_entities": summary.key_entities or [],
                        "theme_keywords": summary.main_themes or [],
                        "important_timeline": business_context.get(
                            "important_timeline", []
                        ),
                        "structural_patterns": business_context.get(
                            "structural_patterns", "unknown"
                        ),
                    }
                    cognitive_maps.append(cognitive_map)

                logger.info(
                    f"Found {len(cognitive_maps)} existing cognitive maps in database"
                )

        # Stage 3: Reasoning enhancement for each document
        logger.info("=== Stage 3: Reasoning Enhancement ===")

        total_reasoning_entities = 0
        total_reasoning_relationships = 0
        reasoning_summaries = []

        def enhance_document_with_reasoning(doc_with_index):
            """Worker function to perform reasoning enhancement on a single document."""
            index, doc = doc_with_index
            try:
                # Get document cognitive map
                doc_cognitive_map = None
                for cognitive_map in cognitive_maps:
                    if cognitive_map.get("source_id") == doc["source_id"]:
                        doc_cognitive_map = cognitive_map
                        break

                # Perform reasoning enhancement
                reasoning_results = self.graph_builder.enhance_knowledge_graph(
                    topic_name, doc, blueprint, doc_cognitive_map
                )

                # Convert reasoning results to graph database
                entities_enhanced, relationships_enhanced = (
                    self.graph_builder.convert_reasoning_results_to_graph(
                        reasoning_results, doc["source_id"]
                    )
                )

                logger.info(
                    f"Document {doc['source_name']} reasoning enhancement: "
                    f"{entities_enhanced} entities, {relationships_enhanced} relationships"
                )

                return (
                    index,
                    doc,
                    entities_enhanced,
                    relationships_enhanced,
                    None,
                )

            except Exception as e:
                error_msg = f"Failed reasoning enhancement for document {doc['source_name']}: {e}"
                logger.error(error_msg, exc_info=True)
                return index, doc, 0, 0, error_msg

        # Process reasoning enhancement in parallel
        reasoning_results = [None] * len(documents)
        reasoning_errors = []

        indexed_documents = list(enumerate(documents))

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all reasoning enhancement tasks
            future_to_doc = {
                executor.submit(
                    enhance_document_with_reasoning, doc_with_index
                ): doc_with_index[1]
                for doc_with_index in indexed_documents
            }

            completed_reasoning = 0
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                completed_reasoning += 1
                try:
                    (
                        index,
                        doc_result,
                        entities_enhanced,
                        relationships_enhanced,
                        error,
                    ) = future.result()

                    if error:
                        reasoning_errors.append(error)
                        logger.warning(
                            f"Reasoning enhancement failed ({completed_reasoning}/{len(documents)}): {doc['source_name']}"
                        )
                    else:
                        reasoning_results[index] = doc_result
                        total_reasoning_entities += entities_enhanced
                        total_reasoning_relationships += relationships_enhanced

                        logger.info(
                            f"Reasoning enhancement completed ({completed_reasoning}/{len(documents)}): {doc['source_name']}"
                        )

                except Exception as e:
                    error_msg = f"Unexpected error in reasoning enhancement for {doc['source_name']}: {e}"
                    reasoning_errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

        # Compile enhanced results
        successful_reasoning = len([r for r in reasoning_results if r is not None])

        if reasoning_errors:
            logger.warning(
                f"Reasoning enhancement completed with {len(reasoning_errors)} errors:"
            )
            for error in reasoning_errors:
                logger.warning(f"  - {error}")

        enhanced_results = {
            "reasoning_enhancement_enabled": True,
            "reasoning_entities_created": total_reasoning_entities,
            "reasoning_relationships_created": total_reasoning_relationships,
            "documents_reasoning_enhanced": successful_reasoning,
            "documents_reasoning_failed": len(reasoning_errors),
        }

        logger.info(
            f"Enhanced knowledge graph construction completed! "
            f"Reasoning: +{total_reasoning_entities} entities, +{total_reasoning_relationships} relationships. "
        )

        if reasoning_errors:
            logger.warning(
                f"Reasoning enhancement had {len(reasoning_errors)} failures. "
                f"Enhanced {successful_reasoning}/{len(documents)} documents successfully."
            )

        return enhanced_results
