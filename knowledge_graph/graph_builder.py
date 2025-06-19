import json
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge_graph.models import DocumentSummary
from knowledge_graph.summarizer import DocumentSummarizer
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
        self.summarizer = DocumentSummarizer(llm_client, session_factory, worker_count)
        self.graph_builder = NarrativeKnowledgeGraphBuilder(
            llm_client, embedding_func, session_factory
        )
        self.worker_count = worker_count

    def build_knowledge_graph(
        self,
        topic_name: str,
        documents: List[Dict],
        force_regenerate_summaries: bool = False,
        force_regenerate_blueprint: bool = False,
    ) -> Dict:
        """
        Build knowledge graph using iterative approach with document summaries.

        Args:
            topic_name: Topic to focus analysis on
            documents: List of document dicts with knowledge blocks
            force_regenerate_summaries: Whether to regenerate existing summaries
            force_regenerate_blueprint: Whether to regenerate existing blueprint

        Returns:
            Dict with construction results and statistics
        """
        logger.info(
            f"Building narrative knowledge graph for topic: {topic_name}: {len(documents)} documents"
        )

        # Stage 0: Generate document summaries
        logger.info("=== Stage 0: Generating document summaries ===")
        summaries = self.generate_document_summaries(
            topic_name, documents, force_regenerate_summaries
        )

        # Stage 1: Generate skeletal graph from summaries
        logger.info("\n=== Stage 1: Generating skeletal graph ===")
        skeletal_graph = self.graph_builder.generate_skeletal_graph_from_summaries(
            topic_name, summaries, force_regenerate_blueprint
        )

        skeletal_context = ""
        if skeletal_graph and skeletal_graph.get("skeletal_entities"):
            skeletal_context = f"""The skeletal graph for {topic_name}:

**Core Entities:**
{json.dumps(skeletal_graph.get('skeletal_entities', []), indent=2)}

**Core Relationships:**
{json.dumps(skeletal_graph.get('skeletal_relationships', []), indent=2)}"""

        # Stage 2: Generate analysis blueprint using skeletal graph
        logger.info("\n=== Stage 2: Generating analysis blueprint ===")
        blueprint = self.graph_builder.generate_analysis_blueprint(
            topic_name,
            summaries,
            skeletal_context,
            skeletal_graph,
            force_regenerate_blueprint,
        )
        if skeletal_graph is None:
            skeletal_graph = blueprint.attributes.get("skeletal_graph", None)

        # Convert skeletal graph to actual entities and relationships after blueprint generation
        logger.info(
            "\n=== Stage 2.5: Converting skeletal graph to entities and relationships ==="
        )
        skeletal_entities_created, skeletal_relationships_created = 0, 0
        if skeletal_graph:
            skeletal_entities_created, skeletal_relationships_created = (
                self.graph_builder.convert_skeletal_graph_to_entities_relationships(
                    skeletal_graph, topic_name, source_id=None
                )
            )

        # Stage 3: Extract narrative triplets to enrich skeletal graph
        logger.info(
            "\n=== Stage 3: Extracting narrative triplets to enrich skeletal graph ==="
        )

        # Phase 1: Parallel triplet extraction
        logger.info(
            f"Processing {len(documents)} documents in parallel using {self.worker_count} workers"
        )

        def extract_and_convert_worker(doc_with_index):
            """Worker function to extract triplets and immediately convert to graph for a single document."""
            index, doc = doc_with_index
            try:
                # Extract triplets from document
                triplets = self.graph_builder.extract_triplets_from_document(
                    topic_name, doc, blueprint, skeletal_graph
                )

                # Count different types of triplets
                doc_semantic = sum(
                    1 for t in triplets if t.get("category") == "narrative"
                )
                doc_structural = sum(
                    1 for t in triplets if t.get("category") == "skeletal"
                )

                logger.info(
                    f"Document {doc['source_name']}: extracted {doc_semantic} narrative + {doc_structural} skeletal triplets"
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
                    doc_semantic,
                    doc_structural,
                    entities_created,
                    relationships_created,
                    None,
                )
            except Exception as e:
                error_msg = f"Failed to process document {doc['source_name']}: {e}"
                logger.error(error_msg, exc_info=True)
                return index, doc, 0, 0, 0, 0, 0, error_msg

        # Process documents in parallel with immediate database saves
        processing_results = [None] * len(documents)  # Pre-allocate to maintain order
        processing_errors = []

        indexed_documents = list(enumerate(documents))

        all_triplets = 0
        semantic_triplets_count = 0
        structural_triplets_count = 0
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
                        doc_semantic,
                        doc_structural,
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
                        semantic_triplets_count += doc_semantic
                        structural_triplets_count += doc_structural
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

        logger.info(
            f"Total triplets extracted: {all_triplets} ({semantic_triplets_count} semantic + {structural_triplets_count} structural)"
        )

        # Compile results
        result = {
            "topic_name": topic_name,
            "blueprint_id": blueprint.id,
            "documents_processed": successful_documents,
            "documents_failed": len(processing_errors),
            "summaries_generated": len(summaries),
            "triplets_extracted": all_triplets,
            "semantic_triplets": semantic_triplets_count,
            "structural_triplets": structural_triplets_count,
            "entities_created": entities_created + skeletal_entities_created,
            "relationships_created": relationships_created
            + skeletal_relationships_created,
            "skeletal_entities_created": skeletal_entities_created,
            "skeletal_relationships_created": skeletal_relationships_created,
            "narrative_entities_created": entities_created,
            "narrative_relationships_created": relationships_created,
            "skeletal_graph": {
                "entities_count": len(skeletal_graph.get("skeletal_entities", [])),
                "relationships_count": len(
                    skeletal_graph.get("skeletal_relationships", [])
                ),
                "skeletal_entities": skeletal_graph.get("skeletal_entities", []),
                "skeletal_relationships": skeletal_graph.get(
                    "skeletal_relationships", []
                ),
            },
            "analysis_blueprint": {
                "suggested_entity_types": blueprint.suggested_entity_types,
                "key_narrative_themes": blueprint.key_narrative_themes,
                "processing_instructions": blueprint.processing_instructions,
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

    def generate_document_summaries(
        self,
        topic_name: str,
        documents: List[Dict],
        force_regenerate: bool = False,
    ) -> List[Dict]:
        """
        Generate topic-focused summaries for all documents.
        Returns summaries in document-like format for blueprint generation.

        Args:
            topic_name: Topic to focus summaries on
            documents: List of document dicts
            force_regenerate: Whether to regenerate existing summaries

        Returns:
            List of document-like summary objects
        """
        return self.summarizer.batch_summarize_documents(
            topic_name, documents, force_regenerate
        )

    def get_topic_summaries(self, topic_name: str) -> List[DocumentSummary]:
        """
        Get all existing summaries for a topic.

        Args:
            topic_name: Topic name to filter by

        Returns:
            List of DocumentSummary objects
        """
        return self.summarizer.get_summaries_for_topic(topic_name)

    def get_skeletal_graph_for_topic(self, topic_name: str) -> Optional[Dict]:
        """
        Get skeletal graph for a topic.

        Args:
            topic_name: Topic name to get skeletal graph for

        Returns:
            Dict with skeletal graph data or None if not found
        """
        return self.graph_builder.get_skeletal_graph_for_topic(topic_name)
