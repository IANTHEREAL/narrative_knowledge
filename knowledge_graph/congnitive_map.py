import json
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge_graph.models import DocumentSummary
from utils.json_utils import robust_json_parse
from setting.db import SessionLocal
from llm.factory import LLMInterface

logger = logging.getLogger(__name__)


class DocumentCognitiveMapGenerator:
    """
    Generates cognitive maps of documents for efficient knowledge graph construction.
    Results are cached in database to avoid recomputation.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        session_factory=None,
        worker_count: int = 3,
    ):
        """
        Initialize the document summarizer.

        Args:
            llm_client: LLM interface for generating cognitive maps
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.SessionLocal = session_factory or SessionLocal
        self.worker_count = worker_count

    def generate_cognitive_map(
        self, topic_name: str, document: Dict, force_regenerate: bool = False
    ) -> Dict:
        """
        Generate a cognitive map for a single document to identify basic structure.

        This is the foundation for subsequent processing stages. The cognitive map
        quickly identifies the document's basic structure including core entities,
        themes, and temporal markers.

        Args:
            topic_name: Topic to focus the cognitive map on
            document: Document dict with source_id, source_name, source_content, etc.
            force_regenerate: Whether to regenerate existing cognitive map

        Returns:
            Dict containing cognitive map data
        """
        with self.SessionLocal() as db:
            # Check for existing cognitive map (stored as summary with special type)
            logger.info(
                f"Checking for existing cognitive map for document: {document['source_name']}"
            )
            existing_map = (
                db.query(DocumentSummary)
                .filter(
                    DocumentSummary.document_id == document["source_id"],
                    DocumentSummary.topic_name == topic_name,
                    DocumentSummary.document_type == "cognitive_map",
                )
                .first()
            )

            if existing_map and not force_regenerate:
                logger.info(
                    f"Using cached cognitive map for document: {document['source_name']}"
                )
                return self._parse_cognitive_map_from_summary(existing_map)

        try:
            # Generate new cognitive map
            logger.info(
                f"Generating cognitive map for document: {document['source_name']}"
            )
            cognitive_map_data = self._generate_document_cognitive_map(
                topic_name, document
            )
        except Exception as e:
            logger.error(
                f"Error generating cognitive map for document: {document['source_name']}: {e}",
                exc_info=True,
            )
            raise e

        # Save cognitive map to database
        with self.SessionLocal() as db:
            if existing_map:
                # Update existing cognitive map
                current_map = (
                    db.query(DocumentSummary)
                    .filter(
                        DocumentSummary.document_id == document["source_id"],
                        DocumentSummary.topic_name == topic_name,
                        DocumentSummary.document_type == "cognitive_map",
                    )
                    .first()
                )

                if current_map:
                    current_map.summary_content = cognitive_map_data["summary"]
                    current_map.key_entities = cognitive_map_data["key_entities"]
                    current_map.main_themes = cognitive_map_data["theme_keywords"]
                    current_map.business_context = json.dumps(
                        {
                            "important_timeline": cognitive_map_data[
                                "important_timeline"
                            ],
                            "structural_patterns": cognitive_map_data[
                                "structural_patterns"
                            ],
                        }
                    )
                    db.commit()
                    db.refresh(current_map)
                    return cognitive_map_data

            # Create new cognitive map entry
            cognitive_map = DocumentSummary(
                document_id=document["source_id"],
                topic_name=topic_name,
                summary_content=cognitive_map_data["summary"],
                key_entities=cognitive_map_data["key_entities"],
                main_themes=cognitive_map_data["theme_keywords"],
                business_context=json.dumps(
                    {
                        "important_timeline": cognitive_map_data["important_timeline"],
                        "structural_patterns": cognitive_map_data[
                            "structural_patterns"
                        ],
                    }
                ),
                document_type="cognitive_map",
            )
            db.add(cognitive_map)
            db.commit()
            db.refresh(cognitive_map)

            cognitive_map_data["source_id"] = document["source_id"]
            cognitive_map_data["source_name"] = document["source_name"]
            return cognitive_map_data

    def batch_generate_cognitive_maps(
        self, topic_name: str, documents: List[Dict], force_regenerate: bool = False
    ) -> List[Dict]:
        """
        Generate cognitive maps for a batch of documents in parallel.

        Args:
            topic_name: Topic to focus cognitive maps on
            documents: List of document dicts
            force_regenerate: Whether to regenerate existing cognitive maps

        Returns:
            List of cognitive map dicts for successfully processed documents
        """
        if not documents:
            return []

        cognitive_maps = []
        errors = []

        def process_document(doc_with_index):
            """Worker function to process a single document."""
            index, doc = doc_with_index
            try:
                cognitive_map = self.generate_cognitive_map(
                    topic_name, doc, force_regenerate
                )
                return index, cognitive_map, None
            except Exception as e:
                error_msg = (
                    f"Failed to generate cognitive map for {doc['source_name']}: {e}"
                )
                logger.error(error_msg, exc_info=True)
                return index, None, error_msg

        # Create list of documents with their original indices to maintain order
        indexed_documents = list(enumerate(documents))

        logger.info(
            f"Starting parallel cognitive map generation for {len(documents)} documents using {self.worker_count} workers"
        )

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(process_document, doc_with_index): doc_with_index[1]
                for doc_with_index in indexed_documents
            }

            # Collect results as they complete
            results = [None] * len(documents)  # Pre-allocate to maintain order
            completed_count = 0

            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                completed_count += 1
                try:
                    index, cognitive_map, error = future.result()
                    if error:
                        errors.append(error)
                        logger.warning(
                            f"Document processing failed ({completed_count}/{len(documents)}): {doc['source_name']}"
                        )
                    else:
                        results[index] = cognitive_map
                        logger.info(
                            f"Document processed successfully ({completed_count}/{len(documents)}): {doc['source_name']}"
                        )
                except Exception as e:
                    error_msg = f"Unexpected error processing {doc['source_name']}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

        # Filter out None values (failed documents) while maintaining order
        cognitive_maps = [result for result in results if result is not None]

        # Log final results
        success_count = len(cognitive_maps)
        failure_count = len(errors)

        if errors:
            logger.warning(
                f"Batch cognitive map generation completed with {failure_count} failures:"
            )
            for error in errors:
                logger.warning(f"  - {error}")

            raise RuntimeError(
                f"Failed to generate cognitive maps for {failure_count}/{len(documents)} documents"
            )

        logger.info(f"Generated {success_count} cognitive maps for topic: {topic_name}")
        return cognitive_maps

    def _generate_document_cognitive_map(self, topic_name: str, document: Dict) -> Dict:
        """
        Generate a cognitive map for a single document using LLM.

        The cognitive map focuses on quickly identifying basic document structure
        to serve as foundation for subsequent processing stages.

        Args:
            topic_name: Topic to focus the cognitive map on
            document: Document dict with source_content

        Returns:
            Dict with cognitive map data
        """
        # Use document source content directly
        doc_content = (
            f"Document: {document['source_name']}\n\n{document['source_content']}\n\n"
            f"Document attributes: {document['source_attributes']}"
        )

        # Generate cognitive map prompt
        cognitive_map_prompt = f"""Analyze this document to create a cognitive map that identifies its basic structure. This will serve as the foundation for deeper knowledge extraction related to "{topic_name}".

<document>
{doc_content}
</document>

Create a cognitive map in JSON format that captures the document's basic structure (surrounding by ```json and ```):

```json
{{
    "summary": "Brief 2-3 sentence overview of the document's main content and purpose",
    "key_entities": ["entity1", "entity2", "entity3"],
    "theme_keywords": ["keyword1", "keyword2", "keyword3"],
    "important_timeline": ["event1: date1", "event2: date2", "event3: date3"],
    "structural_patterns": "document_organization_pattern"
}}
```

Focus on:
1. **Summary**: Core purpose and main content in 2-3 sentences
2. **Key Entities**: Most important people, organizations, systems, concepts (5-10 items)
3. **Theme Keywords**: Main topics and concepts relevant to {topic_name} (5-8 items)
4. **Important Timeline**: Sequential events with their timeframes (if document contains temporal progression)
5. **Structural Patterns**: How the document is organized (e.g., "chronological", "hierarchical", "process_flow", "problem_solution", "comparison")

Guidelines:
- Extract explicit time references (dates, quarters, versions)
- Identify implicit temporal markers (before/after relationships)
- Focus on entities and themes most relevant to {topic_name}
- Keep it concise but comprehensive for structure identification

Return only the JSON, no other text."""

        try:
            response = self.llm_client.generate(cognitive_map_prompt)

            # Use robust JSON parsing with escape error fixing and LLM fallback
            cognitive_map_data = robust_json_parse(response, self.llm_client, "object")

            # Validate and set defaults for required fields
            cognitive_map_data.setdefault("summary", "Summary generation failed")
            cognitive_map_data.setdefault("key_entities", [])
            cognitive_map_data.setdefault("theme_keywords", [])
            cognitive_map_data.setdefault("important_timeline", [])
            cognitive_map_data.setdefault("structural_patterns", "unknown")

            logger.info(
                f"Generated cognitive map for {document['source_name']}: "
                f"{len(cognitive_map_data['key_entities'])} entities, "
                f"{len(cognitive_map_data['theme_keywords'])} themes, "
                f"{len(cognitive_map_data['important_timeline'])} timeline events"
            )

            return cognitive_map_data

        except Exception as e:
            logger.error(
                f"Error generating cognitive map: {e}",
                exc_info=True,
            )
            raise e

    def _parse_cognitive_map_from_summary(self, summary: DocumentSummary) -> Dict:
        """
        Parse cognitive map data from a DocumentSummary object.

        Args:
            summary: DocumentSummary object containing cognitive map data

        Returns:
            Dict with cognitive map structure
        """
        try:
            business_context = (
                json.loads(summary.business_context) if summary.business_context else {}
            )
        except (json.JSONDecodeError, TypeError):
            business_context = {}

        return {
            "source_id": summary.document_id,
            "source_name": summary.source_data.name,
            "summary": summary.summary_content or "",
            "key_entities": summary.key_entities or [],
            "theme_keywords": summary.main_themes or [],
            "important_timeline": business_context.get("important_timeline", []),
            "structural_patterns": business_context.get(
                "structural_patterns", "unknown"
            ),
        }

    def get_cognitive_maps_for_topic(self, topic_name: str) -> List[DocumentSummary]:
        """
        Retrieve all existing cognitive maps for a given topic.

        Args:
            topic_name: Topic name to filter by

        Returns:
            List of DocumentSummary objects with cognitive maps
        """
        with self.SessionLocal() as db:
            return (
                db.query(DocumentSummary)
                .filter(
                    DocumentSummary.topic_name == topic_name,
                    DocumentSummary.document_type == "cognitive_map",
                )
                .order_by(DocumentSummary.created_at.desc())
                .all()
            )
