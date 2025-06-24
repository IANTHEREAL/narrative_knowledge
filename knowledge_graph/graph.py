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
        return robust_json_parse(response, self.llm_client, expected_format)

    def generate_analysis_blueprint(
        self,
        topic_name: str,
        cognitive_maps: List[Dict],
        force_regenerate: bool = False,
    ) -> AnalysisBlueprint:
        """
        Stage 2: Generate Global Blueprint & Instructions for cross-document coordination.

        This creates a comprehensive global blueprint that integrates insights from all
        document cognitive maps to establish a God's-eye view analysis framework.

        Key capabilities:
        - Cross-document entity normalization
        - Global pattern recognition
        - Conflict resolution strategies
        - Unified timeline integration
        """
        if len(cognitive_maps) == 0:
            raise ValueError(f"No cognitive maps found for topic: {topic_name}")

        with self.SessionLocal() as db:
            # Check if blueprint already exists
            existing_blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if existing_blueprint and not force_regenerate:
                logger.info(f"Using existing global blueprint for {topic_name}")
                return existing_blueprint

        # Enhanced Global Blueprint Generation Prompt
        blueprint_prompt = f"""You are a master strategist analyzing cognitive maps from {len(cognitive_maps)} documents for "{topic_name}". 

Your task is to generate a GLOBAL BLUEPRINT that provides cross-document coordination and God's-eye view insights that no single document can provide.

<cognitive_maps_collection>
{json.dumps(cognitive_maps, indent=2, ensure_ascii=False)}
</cognitive_maps_collection>

Generate a comprehensive global blueprint in JSON format with the following structure (surround by ```json and ```):

```json
{{
"canonical_entities": {{
    "normalized_name_1": {{
        "aliases": ["variation1", "variation2", "variation3"],
        "entity_type": "Person|Organization|System|Concept|Event",
        "primary_source": "most_authoritative_document_name",
        "description": "unified description combining insights from all documents"
    }},
    "normalized_name_2": {{
        "aliases": ["Google", "谷歌", "Google Inc."],
        "entity_type": "Organization", 
        "primary_source": "official_press_release.pdf",
        "description": "Global technology company, search engine provider"
    }}
}},
"key_patterns": {{
    "relationship_patterns": [
        "Rich natural language descriptions of meaningful relationship patterns across documents",
        "For example: 'Leadership transitions often trigger organizational restructuring within 3-6 months, affecting both technology adoption and team dynamics'",
        "Another example: 'When companies face external pressure, they tend to accelerate digital transformation while simultaneously tightening internal controls'"
    ],
    "temporal_patterns": [
        "Natural language descriptions of time-based patterns",
        "For example: 'Strategic decisions typically follow a cycle of problem identification, stakeholder consultation, pilot testing, and full implementation spanning 6-12 months'"
    ],
    "narrative_themes": [
        "Cross-document narrative themes that provide rich context",
        "For example: 'The tension between innovation speed and operational stability appears as a recurring challenge across multiple business units'"
    ]
}},
"global_timeline": [
    {{
        "period": "2023-Q1",
        "key_events": ["Event1 from doc_A", "Event2 from doc_B"],
        "cross_document_connections": ["How events relate across documents"]
    }},
    {{
        "period": "2023-Q2", 
        "key_events": ["Major decision point", "System launch"],
        "cross_document_connections": ["Impact chain across multiple documents"]
    }}
],
"processing_instructions": {{
    "conflict_handling": "Guidelines for resolving contradictory information between documents",
    "quality_focus": "What aspects to prioritize for high-quality extraction",
    "extraction_emphasis": "Areas that deserve special attention during detailed analysis",
                "cross_document_insights": "How to leverage the global context for deeper understanding"
    }}
}}
```

**CRITICAL REQUIREMENTS:**

1. **Canonical Entities**: Identify entities mentioned across multiple documents with different names (e.g., "Google" vs "谷歌" vs "Google Inc."). Create normalized names and track all variations.

2. **Rich Relationship Patterns**: Instead of atomic patterns like "A-relation-B", describe meaningful, context-rich relationship patterns in natural language that capture the complexity and nuance of real-world interactions.

3. **Global Timeline**: Integrate timeline events from all documents into a coherent chronological framework, identifying cross-document event sequences.

4. **Flexible Processing Instructions**: Provide guidance on conflict handling, quality focus, extraction emphasis, and cross-document insights without rigid schemas.

5. **Cross-Document Insights**: Focus on patterns, themes, and relationships that only become visible when analyzing all documents together.


**Focus on providing insights that are IMPOSSIBLE to derive from any single document alone.**

Generate the global blueprint for "{topic_name}"."""

        try:
            logger.info(
                f"Generating global blueprint for {topic_name} with {len(cognitive_maps)} cognitive maps"
            )
            response = self.llm_client.generate(blueprint_prompt, max_tokens=8192)
        except Exception as e:
            logger.error(f"Error generating global blueprint: {e}")
            raise RuntimeError(f"Error generating global blueprint: {e}")

        try:
            blueprint_data = self._parse_llm_json_response(response, "object")

            # Extract and format the enhanced blueprint data
            canonical_entities = blueprint_data.get("canonical_entities", {})
            key_patterns = blueprint_data.get("key_patterns", {})
            global_timeline = blueprint_data.get("global_timeline", [])
            processing_instructions_data = blueprint_data.get(
                "processing_instructions", {}
            )

            # Format processing instructions as a comprehensive text
            processing_instructions_parts = []

            if isinstance(processing_instructions_data, dict):
                # Handle flexible processing instructions structure
                for key, value in processing_instructions_data.items():
                    if value:
                        processing_instructions_parts.append(f"{key.upper()}:")
                        processing_instructions_parts.append(value)
                        processing_instructions_parts.append("")

            elif isinstance(processing_instructions_data, str):
                # Handle simple string format
                processing_instructions_parts.append(processing_instructions_data)

            processing_instructions = "\n".join(processing_instructions_parts)

            # All blueprint data in the content JSON field
            blueprint_items = {
                "canonical_entities": canonical_entities,
                "key_patterns": key_patterns,
                "global_timeline": global_timeline,
                "document_count": len(cognitive_maps),
            }

            with self.SessionLocal() as db:
                blueprint = AnalysisBlueprint(
                    topic_name=topic_name,
                    processing_items=blueprint_items,
                    processing_instructions=processing_instructions,
                )

                db.add(blueprint)
                db.commit()
                db.refresh(blueprint)

            logger.info(
                f"Generated global blueprint for {topic_name}:"
                f"\n  - Processing instructions: {processing_instructions}"
                f"\n  - Processing items: {blueprint.processing_items}"
            )

            return blueprint

        except Exception as e:
            logger.error(
                f"Error generating global blueprint: {e}. response: {response}"
            )
            raise RuntimeError(f"Error generating global blueprint: {e}")

    def extract_triplets_from_document(
        self,
        topic_name: str,
        document: Dict,
        blueprint: AnalysisBlueprint,
        document_cognitive_map: Dict = None,
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
                topic_name, document_content, blueprint, document_cognitive_map
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
        document_cognitive_map: Dict = None,
    ) -> List[Dict]:
        """
        Extract enhanced narrative triplets from entire document content.
        Each triplet includes rich entity descriptions and temporal information indicating when facts occurred.
        """

        # Extract global context from blueprint
        global_context = blueprint.processing_items
        canonical_entities = global_context.get("canonical_entities", {})
        key_patterns = global_context.get("key_patterns", {})
        global_timeline = global_context.get("global_timeline", [])

        # Extract document context from cognitive map (if available)
        cognitive_context = ""
        if document_cognitive_map:
            doc_summary = document_cognitive_map.get("summary", "")
            doc_key_entities = document_cognitive_map.get("key_entities", [])
            doc_themes = document_cognitive_map.get("theme_keywords", [])
            doc_timeline = document_cognitive_map.get("important_timeline", [])

            cognitive_context = f"""**Document Cognitive Map:**
- Summary: {doc_summary}
- Key Entities: {json.dumps(doc_key_entities, ensure_ascii=False)}
- Themes: {json.dumps(doc_themes, ensure_ascii=False)}
- Timeline: {json.dumps(doc_timeline, ensure_ascii=False)}
"""

        # Enhanced extraction prompt with full context
        extraction_prompt = f"""You are an expert knowledge extractor working on {topic_name} documents.

**Global Blueprint (Cross-Document Context):**
- Canonical Entities: {json.dumps(canonical_entities, indent=2, ensure_ascii=False)}
- Key Patterns: {json.dumps(key_patterns, indent=2, ensure_ascii=False)}  
- Global Timeline: {json.dumps(global_timeline, indent=2, ensure_ascii=False)}

**Processing Instructions:**
{blueprint.processing_instructions}

**Document Cognitive Map:**
{cognitive_context}

**IMPORTANT EXTRACTION GUIDELINES:**
1. Use canonical entity names from global blueprint when available
2. Align extracted facts with global patterns and timeline
3. Focus on relationships that provide business insights

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
