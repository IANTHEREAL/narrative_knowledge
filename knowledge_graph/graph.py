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
from knowledge_graph.query import query_existing_knowledge
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
                        # Handle both string and list values
                        if isinstance(value, list):
                            # Convert list to formatted string with bullet points
                            for item in value:
                                processing_instructions_parts.append(f"  - {item}")
                        else:
                            processing_instructions_parts.append(str(value))
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
4. **MANDATORY FACT SUPPORT:** Every entity attribute and relationship MUST be directly supported by explicit text from the document

Extract enhanced narrative triplets from this document. Focus on:
1. Finding WHY, HOW, WHEN details for existing relationships
2. Discovering new supporting relationships that add depth
3. **Only extract facts that have clear textual evidence**

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

**ENTITY vs RELATIONSHIP SEPARATION:**
- Entity description = What IS this entity? (intrinsic properties, characteristics, context about the entity itself)
- Relationship description = How entities interact? (all WHO, WHEN, WHERE, HOW, WHY context)

Return a JSON array of enhanced triplets (surround with ```json and ```):

```json
[
    {{
        "subject": {{
            "name": "Entity name",
            "description": "ENTITY-FOCUSED: Detailed description of what this entity IS (intrinsic properties, characteristics, context about the entity itself). EXCLUDE relationships with other entities.",
            "attributes": {{
                "entity_type": "one of the suggested types"
            }}
        }},
        "predicate": "RELATIONSHIP-FOCUSED: Rich narrative describing HOW entities interact (who, what, when, where, why, how context)",
        "object": {{
            "name": "Entity name", 
            "description": "ENTITY-FOCUSED: Detailed description of what this entity IS (intrinsic properties, characteristics, context about the entity itself). EXCLUDE relationships with other entities.",
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

**CRITICAL FACT-BASED REQUIREMENTS:**
1. Every entity description must be based on explicit text about that entity
2. Every relationship must be clearly evidenced by specific text spans
3. Do NOT infer or assume information not directly stated in the text
4. **Rich Entity Descriptions**: Detailed descriptions of what entities ARE, not how they relate to others
5. **Rich Relationship Descriptions**: Detailed descriptions of how entities interact, not just relationship types

Focus on extracting meaningful relationships that reveal business insights WITH their temporal context.
Only extract triplets if they contain valuable knowledge AND have clear textual support.

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

    def enhance_knowledge_graph(
        self,
        topic_name: str,
        document: Dict,
        blueprint: AnalysisBlueprint = None,
        document_cognitive_map: Dict = None,
    ) -> Dict[str, List[Dict]]:
        """
        Enhance knowledge graph through reasoning and inference.

        Acts like a detective to find hidden connections, complete missing knowledge gaps,
        and enrich existing entities with additional insights.

        Args:
            topic_name: Topic to focus the enhancement on
            document: Document dict with source_id, source_name, source_content, etc.
            blueprint: Optional analysis blueprint for global context
            document_cognitive_map: Optional cognitive map for document context

        Returns:
            Dict containing:
            - enhanced_relationships: List of new/enhanced relationship triplets
            - enhanced_entities: List of new/enhanced entity objects
            - reasoning_summary: Summary of the reasoning process
        """
        logger.info(
            f"Starting knowledge graph reasoning enhancement for document: {document['source_name']}"
        )

        # Step 1: Query existing knowledge related to this document
        with self.SessionLocal() as db:
            existing_knowledge = query_existing_knowledge(
                db, document["source_id"], topic_name
            )

        # Step 2: Build context for reasoning
        reasoning_context = self._build_reasoning_context(
            topic_name, document, blueprint, document_cognitive_map, existing_knowledge
        )

        # Step 3: Perform reasoning to discover implicit relationships and entities
        reasoning_results = self._perform_knowledge_reasoning(
            topic_name, document, reasoning_context
        )

        logger.info(
            f"Knowledge reasoning completed for {document['source_name']}: "
            f"{len(reasoning_results.get('enhanced_relationships', []))} enhanced relationships, "
        )

        return reasoning_results

    def _build_reasoning_context(
        self,
        topic_name: str,
        document: Dict,
        blueprint: AnalysisBlueprint = None,
        document_cognitive_map: Dict = None,
        existing_knowledge: Dict = None,
    ) -> str:
        """
        Build comprehensive context for knowledge reasoning.
        """
        context_parts = []

        # Document information
        context_parts.append(f"**Document Information:**")
        context_parts.append(f"- Name: {document['source_name']}")
        context_parts.append(f"- Content: {document['source_content']}")
        context_parts.append(f"- Attributes: {document.get('source_attributes', {})}")
        context_parts.append("")

        # Global blueprint context (if available)
        if blueprint:
            context_parts.append("**Global Blueprint Context:**")
            context_parts.append(
                f"- Processing Instructions: {blueprint.processing_instructions}"
            )
            context_parts.append(
                f"- Processing Items: {json.dumps(blueprint.processing_items, indent=2, ensure_ascii=False)}"
            )
            context_parts.append("")

        # Document cognitive map context (if available)
        if document_cognitive_map:
            context_parts.append("**Document Cognitive Map:**")
            context_parts.append(
                f"{json.dumps(document_cognitive_map, indent=2, ensure_ascii=False)}"
            )
            context_parts.append("")

        # Existing knowledge context
        if existing_knowledge:
            context_parts.append("**Existing Knowledge in Graph:**")
            context_parts.append(
                f"- Total Entities: {existing_knowledge['total_entities']}"
            )
            context_parts.append(
                f"- Total Relationships: {existing_knowledge['total_relationships']}"
            )

            if existing_knowledge["existing_entities"]:
                context_parts.append("- Entities:")
                for entity in existing_knowledge["existing_entities"]:
                    context_parts.append(
                        f"  * {entity['name']}: {entity['description']}"
                    )

            if existing_knowledge["existing_relationships"]:
                context_parts.append("- Relationships:")
                for rel in existing_knowledge["existing_relationships"]:
                    context_parts.append(
                        f"  * {rel['source_entity']['name']} -> {rel['relationship_desc']} -> {rel['target_entity']['name']}"
                    )
            context_parts.append("")

        return "\n".join(context_parts)

    def _perform_knowledge_reasoning(
        self, topic_name: str, document: Dict, reasoning_context: str
    ) -> Dict:
        """
        Use LLM to perform knowledge reasoning and discover implicit relationships.
        """
        reasoning_prompt = f"""You are an expert knowledge detective working on "{topic_name}" analysis.

Your primary mission is to **complete missing information** and **enhance entity descriptions** through **logical reasoning and inference**. 
Crucially, all of your reasoning must be **strictly grounded in the explicit facts** presented in the text. Your goal is not to invent, but to **synthesize and connect scattered information** to create a more complete and valuable knowledge graph.

{reasoning_context}

**FACT-BASED REASONING PRINCIPLES:**

1.  **Entity-Only Descriptions:** Enhance entity descriptions with intrinsic properties only
2.  **Evidence-Based:** Every enhancement must be supported by explicit text
3.  **No Speculation:** Do not infer missing information or assume details not present
4.  **Clear Separation:** Entity properties in descriptions, interactions in relationships

**ANALYTICAL TASKS:**

1.  **Enhance Entity Descriptions**: Add ONLY explicitly stated properties about entities. Focus on what the entity IS (definition, characteristics, features) based on direct text evidence.
2.  **Discover Explicit Relationships**: Find relationships that are clearly stated or directly evidenced in specific text spans. Each relationship must cite supporting text.
3.  **Connect Clear Facts**: Link entities only when connections are explicitly stated or logically unavoidable from direct evidence.

**OUTPUT FORMAT:**

Return a JSON object with your reasoning discoveries in the following format (surround with ```json and ```):

```json
{{
    "enhanced_relationships": [
        {{
            "subject": {{
                "name": "Entity name",
                "description": "ENTITY-FOCUSED: What IS this entity? (intrinsic properties only, no relationships)", 
                "attributes": {{
                    "entity_type": "Organization|Person|System|Concept|Event|Process", 
                }},
                "requires_description_update": true/false,
                "update_justification": "Explanation of why the description should be updated (required if requires_description_update is true)"
            }},
            "predicate": "RELATIONSHIP-FOCUSED: How entities interact (all WHO, WHEN, WHERE, HOW context)",
            "object": {{
                "name": "Entity name",
                "description": "ENTITY-FOCUSED: What IS this entity? (intrinsic properties only, no relationships)",
                "attributes": {{
                    "entity_type": "Person|Organization|System|Concept|Event|Process",
                }},
                "requires_description_update": true/false,
                "update_justification": "Explanation of why the description should be updated (required if requires_description_update is true)"
            }},
            "relationship_attributes": {{
                "fact_time": "when this relationship/fact occurred or was true",
                "time_expression": "original time expression from text if any",
                "sentiment": "positive|negative|neutral",
                "confidence": "high|medium|low",
                "justification": "Clear explanation of reasoning process with reference to supporting evidence in the text",
            }}
        }}
    ]
}}
```

**CRITICAL REQUIREMENTS:**

1. **Evidence-Based Reasoning**: Every enhancement must reference supporting evidence from the text.
2. **Temporal Accuracy**: Capture temporal context precisely based on text evidence.
3. **Rich Entity Descriptions**: Provide detailed descriptions of what entities ARE, not how they relate to others.
4. **Rich Relationship Descriptions**: Provide detailed descriptions of how entities interact, not just the relationship type.
5. **Confidence Scoring**: Include confidence levels with clear justification.
6. **Entity Enhancement Tracking**:
    - For **existing entities**: Set `requires_description_update` to `true` only when adding intrinsic properties about the entity itself.
    - For **new entities**: Always set `requires_description_update` to `false`.

Begin your detective work and discover the hidden knowledge!
"""

        try:
            logger.info(f"Performing knowledge reasoning for {document['source_name']}")
            response = self.llm_client.generate(reasoning_prompt, max_tokens=16384)
        except Exception as e:
            logger.error(f"Error performing knowledge reasoning: {e}")
            raise RuntimeError(f"Error performing knowledge reasoning: {e}")

        try:
            reasoning_results = self._parse_llm_json_response(response, "object")

            # Add metadata to discoveries
            for rel in reasoning_results.get("enhanced_relationships", []):
                rel.update({"topic_name": topic_name})

            return reasoning_results

        except Exception as e:
            logger.error(
                f"Error parsing knowledge reasoning results: {e}, response: {response}"
            )
            raise RuntimeError(f"Error parsing knowledge reasoning results: {e}")

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

    def convert_reasoning_results_to_graph(
        self, reasoning_results: Dict, source_id: str
    ) -> Tuple[int, int]:
        """
        Convert reasoning enhancement results to Entity/Relationship objects in the database.

        Args:
            reasoning_results: Results from enhance_knowledge_graph_with_reasoning
            source_id: Source document ID for mapping

        Returns:
            (entities_created_or_updated, relationships_created)
        """
        entities_created = 0
        relationships_created = 0
        entity_id_cache = {}

        logger.info(
            f"Converting reasoning results to graph: "
            f"{len(reasoning_results.get('enhanced_relationships', []))} relationships"
        )

        for rel_data in reasoning_results.get("enhanced_relationships", []):

            def process_enhanced_relationship():
                nonlocal relationships_created
                nonlocal entities_created

                with self.SessionLocal() as db:
                    try:
                        subject_name = rel_data["subject"]["name"]
                        object_name = rel_data["object"]["name"]
                        topic_name = rel_data["topic_name"]

                        # Get or create subject entity
                        if subject_name in entity_id_cache:
                            subject_entity_id = entity_id_cache[subject_name]
                        else:
                            subject_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == subject_name,
                                    Entity.attributes["topic_name"] == topic_name,
                                )
                                .first()
                            )

                            if not subject_entity:
                                # Create subject entity from relationship
                                subject_entity = Entity(
                                    name=subject_name,
                                    description=rel_data["subject"]["description"],
                                    description_vec=self.embedding_func(
                                        rel_data["subject"]["description"]
                                    ),
                                    attributes={
                                        **rel_data["subject"]["attributes"],
                                        "topic_name": topic_name,
                                    },
                                )
                                db.add(subject_entity)
                                db.flush()
                                entities_created += 1
                            elif rel_data["subject"]["requires_description_update"]:
                                # Enhance existing entity description and attributes
                                subject_entity.description = rel_data["subject"][
                                    "description"
                                ]
                                subject_entity.description_vec = self.embedding_func(
                                    subject_entity.description
                                )

                                enhanced_attributes = {
                                    **subject_entity.attributes,
                                    **rel_data["subject"]["attributes"],
                                }
                                subject_entity.attributes = enhanced_attributes
                                logger.info(
                                    f"Enhanced existing entity: {subject_name}. New description: {subject_entity.description}, justification: {rel_data['subject']['update_justification']}"
                                )

                            subject_entity_id = subject_entity.id
                            entity_id_cache[subject_name] = subject_entity_id

                            # Create source mapping
                            self._create_source_mapping(
                                db, source_id, subject_entity_id, "entity", topic_name
                            )

                        # Get or create object entity
                        if object_name in entity_id_cache:
                            object_entity_id = entity_id_cache[object_name]
                        else:
                            object_entity = (
                                db.query(Entity)
                                .filter(
                                    Entity.name == object_name,
                                    Entity.attributes["topic_name"] == topic_name,
                                )
                                .first()
                            )

                            if not object_entity:
                                # Create object entity from relationship
                                object_entity = Entity(
                                    name=object_name,
                                    description=rel_data["object"]["description"],
                                    description_vec=self.embedding_func(
                                        rel_data["object"]["description"]
                                    ),
                                    attributes={
                                        **rel_data["object"]["attributes"],
                                        "topic_name": topic_name,
                                    },
                                )
                                db.add(object_entity)
                                db.flush()
                                entities_created += 1
                            elif rel_data["object"]["requires_description_update"]:
                                object_entity.description = rel_data["object"][
                                    "description"
                                ]
                                object_entity.description_vec = self.embedding_func(
                                    object_entity.description
                                )

                                enhanced_attributes = {
                                    **object_entity.attributes,
                                    **rel_data["object"]["attributes"],
                                }
                                object_entity.attributes = enhanced_attributes
                                logger.info(
                                    f"Enhanced existing entity: {object_name}. New description: {object_entity.description}, justification: {rel_data['object']['update_justification']}"
                                )

                            object_entity_id = object_entity.id
                            entity_id_cache[object_name] = object_entity_id

                            # Create source mapping
                            self._create_source_mapping(
                                db, source_id, object_entity_id, "entity", topic_name
                            )

                        # Create enhanced relationship
                        relationship_desc = rel_data["predicate"]

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
                            # Create new reasoning-enhanced relationship
                            rel_attributes = {
                                **rel_data["relationship_attributes"],
                                "topic_name": topic_name,
                            }

                            new_relationship = Relationship(
                                source_entity_id=subject_entity_id,
                                target_entity_id=object_entity_id,
                                relationship_desc=relationship_desc,
                                relationship_desc_vec=self.embedding_func(
                                    relationship_desc
                                ),
                                attributes=rel_attributes,
                            )
                            db.add(new_relationship)
                            db.flush()
                            relationships_created += 1

                            # Create source mapping
                            self._create_source_mapping(
                                db,
                                source_id,
                                new_relationship.id,
                                "relationship",
                                topic_name,
                            )

                            logger.info(
                                f"Created reasoning-enhanced relationship: {subject_name} -> {relationship_desc} -> {object_name}"
                            )
                        else:
                            # Update existing relationship with reasoning insights
                            enhanced_attributes = {
                                **existing_rel.attributes,
                                **rel_data["relationship_attributes"],
                            }
                            existing_rel.attributes = enhanced_attributes

                            logger.info(
                                f"Enhanced existing relationship: {subject_name} -> {relationship_desc} -> {object_name}"
                            )

                        db.commit()

                    except Exception as e:
                        db.rollback()
                        raise e

            try:
                self._simple_retry(process_enhanced_relationship)
            except Exception as e:
                logger.error(f"Error processing enhanced relationship: {e}")
                raise RuntimeError(f"Error processing enhanced relationship: {e}")

        logger.info(
            f"Reasoning results conversion completed: "
            f"{entities_created} entities created/enhanced, "
            f"{relationships_created} relationships created"
        )

        return entities_created, relationships_created
