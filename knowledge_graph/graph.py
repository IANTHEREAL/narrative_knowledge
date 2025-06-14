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
from utils.json_utils import extract_json, extract_json_array
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

    def generate_skeletal_graph_from_summaries(
        self, topic_name: str, topic_docs: List[Dict], force_regenerate: bool = False
    ) -> Dict:
        """
        Stage 1: Generate skeletal knowledge graph from document summaries.
        Creates the structural backbone with core entities and relationships.
        """
        logger.info(f"Generating skeletal graph for {topic_name}")

        # Check if skeletal graph already exists
        if not force_regenerate:
            existing_skeletal = self.get_skeletal_graph_for_topic(topic_name)
            if existing_skeletal:
                logger.info(f"Using cached skeletal graph for {topic_name}")
                return existing_skeletal

        # Prepare document summaries
        doc_summaries = []
        for doc in topic_docs:
            doc_summary = f"Document: {doc['source_name']}\n\n{doc['source_content']}"
            doc_summaries.append(doc_summary)

        combined_summaries = "\n" + "=" * 50 + "\n".join(doc_summaries)

        skeletal_prompt = f"""You are an expert knowledge architect analyzing document summaries to build a skeletal knowledge graph for {topic_name}.

Your task is to identify the CORE entities and their PRIMARY relationships that form the structural backbone of {topic_name}.

<document_summaries>
{combined_summaries}
</document_summaries>

### Guiding Principles
1.  **Identify Core Nodes**: First, identify all critical entities and assign a unique `entity_name`.
2.  **Define Relationships**: Connect these entities, describing the relationship with a concise, standardized `relationship_desc`.
3.  **Focus on the Backbone**: Extract only the most structurally important elements. Details will be added later.

Based on these principles, create a skeletal knowledge graph in the following JSON format. Return only the JSON object.

```json
{{
    "skeletal_entities": [
        {{
            "name": "Entity name",
            "description": "Core description of this entity",
            "entity_type": "Business Goal|Key Personnel|Technical Component|Requirement|Challenge|Project|etc.",
        }}
    ],
    "skeletal_relationships": [
        {{
            "source_entity": "Source entity name", 
            "target_entity": "Target entity name",
            "relationship": "Core relationship description",
        }}
    ]
}}
```

Focus on:
1. The most important entities that define {topic_name}
2. Core relationships that explain the main story for {topic_name}

Now, please generate the skeletal graph for {topic_name}.
"""

        try:
            response = self.llm_client.generate(skeletal_prompt, max_tokens=8192)
            json_str = extract_json(response)
            if not json_str:
                raise ValueError(f"Failed to parse skeletal graph response: {response}")

            skeletal_data = json.loads(json_str)
            logger.info(
                f"Generated skeletal graph with {len(skeletal_data.get('skeletal_entities', []))} entities and {len(skeletal_data.get('skeletal_relationships', []))} relationships"
            )

            return skeletal_data

        except Exception as e:
            logger.error(f"Error generating skeletal graph: {e}")
            raise RuntimeError(f"Error generating skeletal graph: {e}")

    def generate_analysis_blueprint(
        self,
        topic_name: str,
        topic_docs: List[Dict],
        skeletal_context: Optional[str] = None,
        skeletal_graph: Optional[Dict] = None,
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

            # Stage 2 prompt: Analysis blueprint generation (enhanced with skeletal graph)
            blueprint_prompt = f"""
You are an expert strategist analyzing a complete "context package" for {topic_name}. 
Your task is to generate a strategic analysis blueprint that will guide the detailed extraction of narrative knowledge from documents.
<skeletal_graph_context>
{skeletal_context}
</skeletal_graph_context>

<documents_summary>
{combined_docs_summary}
</documents_summary>

Based on the skeletal graph structure and document summaries, create a JSON analysis blueprint for detailed knowledge extraction:

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

Now, please generate the analysis blueprint for {topic_name}.
"""

            try:
                logger.info(f"Generating analysis blueprint for {topic_name}")
                response = self.llm_client.generate(blueprint_prompt, max_tokens=4096)

                # Extract JSON from response
                json_str = extract_json(response)
                if not json_str:
                    raise ValueError(
                        f"Failed to parse LLM response as JSON: {response}"
                    )

                blueprint_data = json.loads(json_str)

                # Create and save blueprint with skeletal graph from previous stage
                attributes = {"document_count": len(topic_docs)}

                # Save skeletal graph if provided
                if skeletal_graph:
                    attributes["skeletal_graph"] = skeletal_graph
                    logger.info(
                        f"Saving skeletal graph with blueprint for {topic_name}"
                    )

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

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse LLM response as JSON: {e}\nResponse: {response}"
                )
            except Exception as e:
                raise RuntimeError(f"Error generating analysis blueprint: {e}")

    def extract_triplets_from_document(
        self,
        topic_name: str,
        document: Dict,
        blueprint: AnalysisBlueprint,
        skeletal_context: Optional[str] = None,
    ) -> List[Dict]:
        """
        Stage 3: Extract enhanced narrative triplets from entire document to enrich skeletal graph.
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
            f"Document: {document['source_name']}\n\n{document['source_content']}"
        )

        try:
            # 1. Extract semantic triplets from entire document (enriching skeletal graph)
            semantic_triplets = self.extract_narrative_triplets_from_document_content(
                topic_name, document_content, blueprint, skeletal_context
            )

            for triplet in semantic_triplets:
                logger.info(f"semantic triplet: {triplet}")

            # 2. Extract structural triplets from entire document (enriching skeletal graph)
            structural_triplets = (
                self.extract_structural_triplets_from_document_content(
                    topic_name, document_content, blueprint, skeletal_context
                )
            )

            for triplet in structural_triplets:
                logger.info(f"structural triplet: {triplet}")

            # Combine both types of triplets
            all_triplets = semantic_triplets + structural_triplets

            logger.info(
                f"Document({document['source_name']}): {len(semantic_triplets)} semantic + {len(structural_triplets)} structural triplets. Extracted {len(all_triplets)} total triplets."
            )

            return all_triplets

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
        skeletal_context: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract enhanced narrative triplets from entire document content to enrich the skeletal graph.
        Each triplet includes rich entity descriptions and attributes.
        """

        # Prepare skeletal graph context
        context_block = ""
        if skeletal_context:
            context_block = f"""
<skeletal_graph_context>
{skeletal_context}
</skeletal_graph_context>
"""

        # Enhanced extraction prompt
        extraction_prompt = f"""You are an expert knowledge extractor working on {topic_name} documents.

Understand the following skeletal graph context for {topic_name}:

{context_block}

Use this analysis blueprint to guide extraction:

**Entity Types to Focus On:**
{json.dumps(blueprint.suggested_entity_types, indent=2)}

**Narrative Themes:**
{json.dumps(blueprint.key_narrative_themes, indent=2)}

**Instructions:**
{blueprint.processing_instructions}

Extract enhanced narrative triplets from this document. Focus on:
1. ENRICHING the skeletal graph with detailed narrative context
2. Finding WHY, HOW, WHEN details for existing skeletal relationships
3. Discovering new supporting relationships that add depth

Each triplet should include:
- Rich entity descriptions and attributes
- Detailed narrative relationships
- Proper categorization

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
                "entity_type": "one of the suggested types",
            }}
        }},
        "predicate": "Rich narrative relationship with WHO, WHAT, WHEN, WHERE, WHY context",
        "object": {{
            "name": "Entity name", 
            "description": "Detailed contextual description of the entity",
            "attributes": {{
                "entity_type": "one of the suggested types",
            }}
        }},
        "relationship_attributes": {{
            "timestamp": "time context if available",
            "sentiment": "positive/negative/neutral"
        }}
    }}
]
```

Focus on extracting meaningful relationships that reveal business insights. 
Only extract triplets if they contain valuable knowledge.

Now, please generate the narrative triplets for {topic_name}.
"""

        try:
            response = self.llm_client.generate(extraction_prompt, max_tokens=16384)

            # Extract and parse JSON
            json_str = extract_json_array(response)
            json_str = "".join(
                char for char in json_str if ord(char) >= 32 or char in "\r\t"
            )

            triplets = json.loads(json_str)

            # Add metadata to each triplet
            for triplet in triplets:
                triplet.update({"topic_name": topic_name, "category": "narrative"})

            return triplets

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from document content: {e}")
            raise RuntimeError(f"Failed to parse JSON from document content: {e}")
        except Exception as e:
            logger.error(f"Error processing document content: {e}")
            raise RuntimeError(f"Error processing document content: {e}")

    def extract_structural_triplets_from_document_content(
        self,
        topic_name: str,
        document_content: str,
        blueprint: AnalysisBlueprint,
        skeletal_context: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract structural triplets that connect the topic to its key aspects and build hierarchical relationships.
        Focuses on predefined structural entities from the blueprint.
        """

        context_block = ""
        if skeletal_context:
            context_block = f"""
<skeletal_graph_context>
{skeletal_context}
</skeletal_graph_context>
"""

        # Structural extraction prompt
        structural_prompt = f"""You are an expert knowledge extractor expanding the detailed structural backbone of a knowledge graph for {topic_name}.

Understand the following skeletal graph context for {topic_name}:

{context_block}

<document_content>
{document_content}
</document_content>

Extract structural triplets that build the topic-centric hierarchy. Look for:
- How this document content relates to the skeletal graph context
- High-level to detailed relationships within the content

Return a JSON array of structural triplets:

```json
[
    {{
        "subject": {{
            "name": "Entity name (could be {topic_name} or one of the structural entities)",
            "description": "Detailed contextual description",
            "attributes": {{
                "entity_type": "one of the suggested types",
            }}
        }},
        "predicate": "Rich structural relationship describing HOW the subject relates to the object in context of {topic_name}",
        "object": {{
            "name": "Entity name",
            "description": "Detailed contextual description",
            "attributes": {{
                "entity_type": "one of the suggested types",
            }}
        }},
        "relationship_attributes": {{
            "timestamp": "time context if available",
            "sentiment": "positive/negative/neutral"
            "hierarchy_level": "topic_to_aspect|aspect_to_component|component_to_detail",
        }}
    }}
]
```

Focus on building clear topic-centric structure. Only extract triplets if they reveal structural relationships.
Now, please generate the structural triplets for {topic_name}.
"""

        try:
            response = self.llm_client.generate(structural_prompt, max_tokens=16384)

            # Extract and parse JSON
            json_str = extract_json_array(response)
            json_str = "".join(
                char for char in json_str if ord(char) >= 32 or char in "\r\t"
            )

            triplets = json.loads(json_str)

            # Add metadata to each triplet
            for triplet in triplets:
                triplet.update({"topic_name": topic_name, "category": "skeletal"})

            return triplets

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse structural JSON from document content: {e}. response {json_str}"
            )
            raise RuntimeError(
                f"Failed to parse structural JSON from document content: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error processing structural extraction for document content: {e}"
            )
            raise RuntimeError(
                f"Error processing structural extraction for document content: {e}"
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

    def get_skeletal_graph_for_topic(self, topic_name: str) -> Optional[Dict]:
        """Retrieve saved skeletal graph for a topic."""
        with self.SessionLocal() as db:
            blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if blueprint and blueprint.attributes:
                return blueprint.attributes.get("skeletal_graph")
            return None

    def convert_skeletal_graph_to_entities_relationships(
        self, skeletal_graph: Dict, topic_name: str, source_id: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Convert skeletal graph entities and relationships to actual database records.

        Args:
            skeletal_graph: Dict containing skeletal_entities and skeletal_relationships
            topic_name: Topic name for context
            source_id: Optional source ID for mappings

        Returns:
            Tuple of (entities_created, relationships_created)
        """
        if not skeletal_graph:
            return 0, 0

        entities_created = 0
        relationships_created = 0
        entity_id_cache = {}  # Map entity names to entity IDs

        # Convert skeletal entities to actual entities
        skeletal_entities = skeletal_graph.get("skeletal_entities", [])
        for entity_data in skeletal_entities:
            entity_name = entity_data.get("name", "")
            if not entity_name:
                continue

            logger.info(f"Converting skeletal entity {entity_data} to actual entity")

            def process_single_entity():
                nonlocal entities_created

                with self.SessionLocal() as db:
                    try:
                        # Check if entity already exists
                        existing_entity = (
                            db.query(Entity)
                            .filter(
                                Entity.name == entity_name,
                                Entity.attributes["topic_name"] == topic_name,
                            )
                            .first()
                        )

                        if not existing_entity:
                            # Create new entity
                            entity = Entity(
                                name=entity_name,
                                description=entity_data.get(
                                    "description", f"Skeletal entity for {topic_name}"
                                ),
                                description_vec=self.embedding_func(
                                    entity_data.get("description", entity_name)
                                ),
                                attributes={
                                    "entity_type": entity_data.get(
                                        "entity_type", "unknown"
                                    ),
                                    "category": "skeletal",
                                    "topic_name": topic_name,
                                },
                            )
                            db.add(entity)
                            db.flush()
                            entities_created += 1
                            entity_id_cache[entity_name] = entity.id

                            # Create source mapping if source_id provided
                            if source_id:
                                self._create_source_mapping(
                                    db, source_id, entity.id, "entity", topic_name
                                )
                        else:
                            entity_id_cache[entity_name] = existing_entity.id

                        db.commit()

                    except Exception as e:
                        db.rollback()
                        raise e

            try:
                self._simple_retry(process_single_entity)
            except Exception as e:
                logger.error(f"Error creating skeletal entity {entity_name}: {e}")
                raise RuntimeError(f"Error creating skeletal entity {entity_name}: {e}")

        # Convert skeletal relationships to actual relationships
        skeletal_relationships = skeletal_graph.get("skeletal_relationships", [])
        for rel_data in skeletal_relationships:
            source_name = rel_data.get("source_entity", "")
            target_name = rel_data.get("target_entity", "")
            relationship_desc = rel_data.get("relationship", "")

            logger.info(
                f"Converting skeletal relationship {rel_data} to actual relationship"
            )

            if not all([source_name, target_name, relationship_desc]):
                logger.warning(f"skip relationship which has missing data: {rel_data}")
                continue

            def process_single_relationship():
                nonlocal relationships_created

                with self.SessionLocal() as db:
                    try:
                        # Get entity IDs from cache or database
                        source_entity_id = entity_id_cache.get(source_name)
                        target_entity_id = entity_id_cache.get(target_name)

                        if not source_entity_id:
                            source_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == source_name,
                                    Entity.attributes["topic_name"] == topic_name,
                                )
                                .first()
                            )
                            if source_entity:
                                source_entity_id = source_entity.id

                        if not target_entity_id:
                            target_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == target_name,
                                    Entity.attributes["topic_name"] == topic_name,
                                )
                                .first()
                            )
                            if target_entity:
                                target_entity_id = target_entity.id

                        if not source_entity_id or not target_entity_id:
                            logger.warning(
                                f"Missing entities for relationship: {source_name} -> {target_name}"
                            )
                            return

                        # Check if relationship already exists
                        existing_rel = (
                            db.query(Relationship)
                            .filter(
                                Relationship.source_entity_id == source_entity_id,
                                Relationship.target_entity_id == target_entity_id,
                                Relationship.relationship_desc == relationship_desc,
                            )
                            .first()
                        )

                        if not existing_rel:
                            # Create new relationship
                            relationship = Relationship(
                                source_entity_id=source_entity_id,
                                target_entity_id=target_entity_id,
                                relationship_desc=relationship_desc,
                                relationship_desc_vec=self.embedding_func(
                                    relationship_desc
                                ),
                                attributes={
                                    "topic_name": topic_name,
                                    "category": "skeletal",
                                },
                            )
                            db.add(relationship)
                            db.flush()
                            relationships_created += 1

                            # Create source mapping if source_id provided
                            if source_id:
                                self._create_source_mapping(
                                    db,
                                    source_id,
                                    relationship.id,
                                    "relationship",
                                    topic_name,
                                )

                        db.commit()

                    except Exception as e:
                        db.rollback()
                        raise e

            try:
                self._simple_retry(process_single_relationship)
            except Exception as e:
                logger.error(
                    f"Error creating skeletal relationship {source_name} -> {target_name}: {e}"
                )
                raise RuntimeError(
                    f"Error creating skeletal relationship {source_name} -> {target_name}: {e}"
                )

        logger.info(
            f"Skeletal graph converted: {entities_created} entities, {relationships_created} relationships"
        )
        return entities_created, relationships_created
