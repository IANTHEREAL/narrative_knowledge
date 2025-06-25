import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func

from knowledge_graph.models import (
    Entity,
    Relationship,
    AnalysisBlueprint,
    SourceGraphMapping,
)
from utils.json_utils import robust_json_parse
from setting.db import SessionLocal
from llm.factory import LLMInterface

logger = logging.getLogger(__name__)


class NarrativeKnowledgeGraphBuilder:
    """
    A builder class for constructing narrative knowledge graphs from documents.
    Implements the two-stage extraction pipeline from graph_design.md.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        embedding_func: Callable,
        session_factory=None,
    ):
        """
        Initialize the builder with LLM client and embedding function.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.embedding_func = embedding_func
        self.SessionLocal = session_factory or SessionLocal

    def _parse_llm_json_response(
        self, response: str, expected_format: str = "object"
    ) -> Any:
        """
        Parse LLM JSON response with escape error fixing and LLM fallback.
        Focuses on escape issues with simple fallback strategy.
        """
        return robust_json_parse(response, expected_format, self.llm_client)

    def generate_analysis_blueprint(
        self,
        topic_name: str,
        topic_docs: List[Dict],
        force_regenerate: bool = False,
    ) -> AnalysisBlueprint:
        """
        Stage 1: Generate analysis blueprint for a topic.
        LLM acts as a strategist to create extraction strategy.
        """
        if len(topic_docs) == 0:
            raise ValueError(f"No documents found for topic: {topic_name}")

        with self.SessionLocal() as db:
            # Check if blueprint already exists
            existing_blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if existing_blueprint and not force_regenerate:
                logger.info(f"Using existing analysis blueprint for {topic_name}")
                return existing_blueprint

            # Prepare document summaries for analysis
            doc_summaries = []
            for doc in topic_docs:
                doc_summary = (
                    f"Document: {doc['source_name']}\n\n{doc['source_content']}"
                )
                doc_summaries.append(doc_summary)

            combined_docs_summary = "\n".join(doc_summaries)

            # Stage 2 prompt: Analysis blueprint generation
            blueprint_prompt = f"""You are an expert strategist analyzing a complete "context package" for {topic_name}.
Your task is to generate a strategic analysis blueprint that will guide the detailed extraction of narrative knowledge from documents.

<documents_summary>
{combined_docs_summary}
</documents_summary>

Based on the document summaries, create a JSON analysis blueprint for detailed knowledge extraction:

The blueprint should be in the following structure:

{{
    "suggested_entity_types": [
        "List of entity types that seem most relevant for {topic_name}",
        "Examples: Key Personnel, Requirements, Challenges, Core Projects, Business Goals, Technical Components, etc."
    ],
    "key_narrative_themes": [
        "List of narrative themes to focus on during extraction",
        "Examples: Decision-making processes, Problem-solution relationships, Timeline of events, etc."
    ],
    "processing_instructions": "Additional guidance for the extraction process specific to this topic's context",
}}

Focus on:
1. What types of entities appear most frequently and seem most important
2. What narrative patterns or themes would capture the most valuable insights
3. What business context or domain-specific considerations are relevant
4. create processing_instructions to guide the extraction process to achieve the goal of the blueprint

Now, please generate the analysis blueprint for {topic_name} in valid JSON format.
"""

            try:
                logger.info(f"Generating analysis blueprint for {topic_name}")
                response = self.llm_client.generate(blueprint_prompt, max_tokens=4096)
            except Exception as e:
                logger.error(f"Error generating analysis blueprint: {e}")
                raise RuntimeError(f"Error generating analysis blueprint: {e}")

            try:
                blueprint_data = self._parse_llm_json_response(response, "object")

                # Create and save blueprint
                attributes = {"document_count": len(topic_docs)}

                processing_instructions_data = blueprint_data.get(
                    "processing_instructions", ""
                )
                if isinstance(processing_instructions_data, list):
                    processing_instructions = "\n".join(processing_instructions_data)
                else:
                    processing_instructions = str(processing_instructions_data)

                blueprint = AnalysisBlueprint(
                    topic_name=topic_name,
                    suggested_entity_types=blueprint_data.get(
                        "suggested_entity_types", []
                    ),
                    key_narrative_themes=blueprint_data.get("key_narrative_themes", []),
                    processing_instructions=processing_instructions,
                    attributes=attributes,
                )

                db.add(blueprint)
                db.commit()
                db.refresh(blueprint)

                logger.info(
                    f"Generated analysis blueprint for {topic_name}, entity types: {blueprint.suggested_entity_types}, narrative themes: {blueprint.key_narrative_themes}"
                )
                return blueprint

            except Exception as e:
                logger.error(
                    f"Error generating analysis blueprint: {e}. response: {response}"
                )
                raise RuntimeError(f"Error generating analysis blueprint: {e}")

    def extract_triplets_from_document(
        self,
        topic_name: str,
        document: Dict,
        blueprint: AnalysisBlueprint,
    ) -> List[Dict]:
        """
        Stage 3: Extract enhanced narrative triplets from entire document.
        Returns all triplets with their source document information.
        """
        logger.info(
            f"Processing document to extract triplets: {document['source_name']}"
        )
        # check whether the document is already processed with topic_name in SourceGraphMapping
        with self.SessionLocal() as db:
            existing_document = (
                db.query(SourceGraphMapping)
                .filter(
                    SourceGraphMapping.source_id == document["source_id"],
                    SourceGraphMapping.attributes["topic_name"] == topic_name,
                )
                .first()
            )
            if existing_document:
                logger.info(
                    f"Document already exists in the database: {document['source_name']}"
                )
                return []

        document_content = (
            f"Document: {document['source_name']}\n\n{document['source_content']}\n\n"
            f"Document attributes: {document['source_attributes']}"
        )

        try:
            # 1. Extract semantic triplets from entire document
            semantic_triplets = self.extract_narrative_triplets_from_document_content(
                topic_name, document_content, blueprint
            )

            for triplet in semantic_triplets:
                logger.info(f"semantic triplet: {triplet}")

            logger.info(
                f"Document({document['source_name']}): {len(semantic_triplets)} semantic triplets. Extracted {len(semantic_triplets)} total triplets."
            )

            return semantic_triplets

        except Exception as e:
            logger.error(
                f"Error extracting from document {document['source_name']}: {e}"
            )
            raise RuntimeError(
                f"Error extracting from document {document['source_name']}: {e}"
            )

    def extract_narrative_triplets_from_document_content(
        self,
        topic_name: str,
        document_content: str,
        blueprint: AnalysisBlueprint,
    ) -> List[Dict]:
        """
        Extract enhanced narrative triplets from entire document content.
        Each triplet includes rich entity descriptions and temporal information indicating when facts occurred.
        """

        # Enhanced extraction prompt
        extraction_prompt = f"""You are an expert knowledge extractor working on {topic_name} documents.

Use this analysis blueprint to guide extraction:

**Entity Types to Focus On:**
{json.dumps(blueprint.suggested_entity_types, indent=2)}

**Narrative Themes:**
{json.dumps(blueprint.key_narrative_themes, indent=2)}

**Instructions:**
{blueprint.processing_instructions}

Extract enhanced narrative triplets from this document. Focus on:
1. Finding WHY, HOW, WHEN details for existing relationships
2. Discovering new supporting relationships that add depth

**CRITICAL: TIME EXTRACTION REQUIREMENTS**
For each triplet, you MUST identify when the fact occurred or was true. Use this systematic approach:

**Time Identification Strategy:**
1. **Explicit Time Markers**: Look for direct time references
   - Absolute dates: "2024年", "January 2023", "Q1 2024"
   - Relative times: "last year", "next month", "recently"
   - Versions/iterations: "v2.0", "latest version", "updated system"

2. **Contextual Time Inference**: When no explicit time exists
   - Document publication/creation date as baseline
   - Sequential indicators: "after X", "before Y", "following the meeting"
   - Project phases: "during development", "post-launch", "initial phase"
   - Business cycles: "this quarter", "fiscal year", "annual review"

3. **Time Expression Standards**:
   - Precise dates: "2024-03-15"
   - Year/month: "2024-03" or "March 2024"
   - Quarters: "Q1 2024"
   - Relative: "late 2023", "early 2024"
   - Event-based: "post-project-launch", "pre-system-migration"

Each triplet should include:
- Rich entity descriptions and attributes
- Detailed narrative relationships
- Proper categorization
- **MANDATORY temporal information**

<document_content>
{document_content}
</document_content>

Return a JSON array of enhanced triplets:

```json
[
    {{
        "subject": {{
            "name": "Entity name",
            "description": "Detailed contextual description of the entity",
            "attributes": {{
                "entity_type": "one of the suggested types"
            }}
        }},
        "predicate": "Rich narrative relationship with WHO, WHAT, WHEN, WHERE, WHY context",
        "object": {{
            "name": "Entity name", 
            "description": "Detailed contextual description of the entity",
            "attributes": {{
                "entity_type": "one of the suggested types"
            }}
        }},
        "relationship_attributes": {{
            "fact_time": "when this relationship/fact occurred or was true",
            "time_expression": "original time expression from text if any",
            "sentiment": "positive|negative|neutral"
        }}
    }}
]
```

Focus on extracting meaningful relationships that reveal business insights WITH their temporal context.
Only extract triplets if they contain valuable knowledge.

Now, please generate the narrative triplets for {topic_name} in valid JSON format.
"""

        try:
            response = self.llm_client.generate(extraction_prompt, max_tokens=16384)
        except Exception as e:
            logger.error(f"Error generating narrative triplets: {e}")
            raise RuntimeError(f"Error generating narrative triplets: {e}")

        try:
            triplets = self._parse_llm_json_response(response, "array")
            # Add metadata to each triplet
            for triplet in triplets:
                triplet.update({"topic_name": topic_name, "category": "narrative"})

            return triplets

        except Exception as e:
            logger.error(
                f"Error processing narrative triplets from document content: {e}, response: {response}"
            )
            raise RuntimeError(
                f"Error processing narrative triplets from document content: {e}"
            )

    def _simple_retry(self, operation_func, max_retries=3):
        """Simple retry for database operations with connection timeouts."""
        for attempt in range(max_retries):
            try:
                return operation_func()
            except Exception as e:
                if "Lost connection" in str(e) or "MySQL server has gone away" in str(
                    e
                ):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database connection lost, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(1)
                        continue
                raise e

    def convert_triplets_to_graph(
        self, triplets: List[Dict], source_id: str
    ) -> Tuple[int, int]:
        """
        Convert enhanced narrative triplets to Entity/Relationship objects with SourceGraphMapping.
        Returns (entities_created, relationships_created).
        """
        entities_created = 0
        relationships_created = 0
        entity_id_cache = {}  # Cache entity IDs to avoid cross-session issues

        for triplet in triplets:

            def process_single_triplet():
                nonlocal entities_created, relationships_created

                with self.SessionLocal() as db:
                    try:
                        # Create or get subject entity
                        subject_data = triplet["subject"]
                        subject_name = subject_data["name"]
                        subject_hash = hashlib.md5(subject_name.encode()).hexdigest()

                        if subject_hash not in entity_id_cache:
                            subject_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == subject_name,
                                    Entity.attributes["topic_name"]
                                    == triplet["topic_name"],
                                )
                                .first()
                            )

                            if not subject_entity:
                                subject_entity = Entity(
                                    name=subject_name,
                                    description=subject_data.get("description", ""),
                                    description_vec=self.embedding_func(
                                        subject_data.get("description", subject_name)
                                    ),
                                    attributes={
                                        **subject_data.get("attributes", {}),
                                        "topic_name": triplet["topic_name"],
                                        "category": triplet["category"],
                                    },
                                )
                                db.add(subject_entity)
                                db.flush()
                                entities_created += 1
                            entity_id_cache[subject_hash] = subject_entity.id

                        subject_entity_id = entity_id_cache[subject_hash]
                        self._create_source_mapping(
                            db,
                            source_id,
                            subject_entity_id,
                            "entity",
                            triplet["topic_name"],
                        )

                        # Create or get object entity
                        object_data = triplet["object"]
                        object_name = object_data["name"]
                        object_hash = hashlib.md5(object_name.encode()).hexdigest()

                        if object_hash not in entity_id_cache:
                            object_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == object_name,
                                    Entity.attributes["topic_name"]
                                    == triplet["topic_name"],
                                )
                                .first()
                            )

                            if not object_entity:
                                object_entity = Entity(
                                    name=object_name,
                                    description=object_data.get("description", ""),
                                    description_vec=self.embedding_func(
                                        object_data.get("description", object_name)
                                    ),
                                    attributes={
                                        **object_data.get("attributes", {}),
                                        "topic_name": triplet["topic_name"],
                                        "category": triplet["category"],
                                    },
                                )
                                db.add(object_entity)
                                db.flush()
                                entities_created += 1
                            entity_id_cache[object_hash] = object_entity.id

                        object_entity_id = entity_id_cache[object_hash]
                        self._create_source_mapping(
                            db,
                            source_id,
                            object_entity_id,
                            "entity",
                            triplet["topic_name"],
                        )

                        # Create relationship
                        relationship_desc = triplet["predicate"]

                        # Check if relationship already exists
                        existing_rel = (
                            db.query(Relationship)
                            .filter(
                                Relationship.source_entity_id == subject_entity_id,
                                Relationship.target_entity_id == object_entity_id,
                                Relationship.relationship_desc == relationship_desc,
                            )
                            .first()
                        )

                        if not existing_rel:
                            # Create new relationship
                            rel_attributes = {
                                "topic_name": triplet["topic_name"],
                                "category": triplet["category"],
                                **triplet.get("relationship_attributes", {}),
                            }

                            relationship = Relationship(
                                source_entity_id=subject_entity_id,
                                target_entity_id=object_entity_id,
                                relationship_desc=relationship_desc,
                                relationship_desc_vec=self.embedding_func(
                                    relationship_desc
                                ),
                                attributes=rel_attributes,
                            )
                            db.add(relationship)
                            db.flush()
                            relationships_created += 1

                            self._create_source_mapping(
                                db,
                                source_id,
                                relationship.id,
                                "relationship",
                                triplet["topic_name"],
                            )
                        else:
                            # Relationship exists - just create the source mapping
                            self._create_source_mapping(
                                db,
                                source_id,
                                existing_rel.id,
                                "relationship",
                                triplet["topic_name"],
                            )

                        db.commit()

                    except Exception as e:
                        db.rollback()
                        raise e

            try:
                self._simple_retry(process_single_triplet)
            except Exception as e:
                logger.error(f"Error processing triplet {triplet}: {e}")
                raise RuntimeError(f"Error processing triplet {triplet}: {e}")

        return entities_created, relationships_created

    def _create_source_mapping(
        self,
        db,
        source_id: str,
        graph_element_id: str,
        element_type: str,
        topic_name: str,
    ):
        """Create SourceGraphMapping entry if it doesn't exist."""
        # Skip if source_id is None or empty
        if not source_id:
            return

        existing_mapping = (
            db.query(SourceGraphMapping)
            .filter(
                SourceGraphMapping.source_id == source_id,
                SourceGraphMapping.graph_element_id == graph_element_id,
                SourceGraphMapping.graph_element_type == element_type,
                SourceGraphMapping.attributes["topic_name"] == topic_name,
            )
            .first()
        )

        if not existing_mapping:
            mapping = SourceGraphMapping(
                source_id=source_id,
                graph_element_id=graph_element_id,
                graph_element_type=element_type,
                attributes={"topic_name": topic_name},
            )
            db.add(mapping)
