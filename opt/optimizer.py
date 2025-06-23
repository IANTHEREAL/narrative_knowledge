import json
import os
import logging

from setting.db import db_manager
from utils.json_utils import extract_json
from utils.token import calculate_tokens
from opt.graph_retrieval import (
    query_entities_by_ids,
    get_relationship_by_entity_ids,
    get_relationship_by_ids,
    get_source_data_by_entity_ids,
    get_source_data_by_relationship_ids,
)
from llm.embedding import (
    get_entity_description_embedding,
    get_text_embedding,
)
from llm.factory import LLMInterface

session_factory = db_manager.get_session_factory(os.getenv("GRAPH_DATABASE_URI"))

##### refine entity

def refine_entity(llm_client, issue, entity, relationships, source_data_list):
    format_relationships = []
    consumed_tokens = 0
    for relationship in relationships.values():
        relationship_str = f"""{relationship['source_entity_name']} -> {relationship['target_entity_name']}: {relationship['relationship_desc']}"""
        consumed_tokens += calculate_tokens(relationship_str)
        if consumed_tokens > 30000:
            break
        format_relationships.append(relationship_str)

    consumed_tokens = calculate_tokens(json.dumps(format_relationships, indent=2))

    # make the token won't exceed 65536
    selected_source_data = []
    for source_data in source_data_list:
        consumed_tokens += calculate_tokens(source_data["content"])
        if consumed_tokens > 70000:
            selected_source_data = selected_source_data[:-1]
            break
        selected_source_data.append(source_data)

    improve_entity_quality_prompt = f"""You are an expert assistant specializing in technologies and knowledge graph curation, tasked with rectifying quality issues within a single entity.

## Objective

Your primary goal is to **transform a problematic entity into an accurate, coherent, meaningful, and self-contained representation**. This involves correcting identified flaws, enriching its details using available context, and ensuring it becomes a high-quality, usable piece of information. The improved entity must be clear and easily understood by a knowledgeable audience (which may include those not deeply expert in every specific nuance).

## Input Data

You will be provided with the following information:

1.  **Entity Quality Issue (`issue`):** Describes the specific quality problem(s) with the entity that needs to be addressed.
    ```json
    {json.dumps(issue, indent=2)}
    ```

2.  **Entity to Improve (`entity_to_improve`):** The entity object refered in the issue.
    ```json
    {json.dumps(entity, indent=2)}
    ```

3.  **Background Information:** Use this to gain a deeper understanding, resolve inconsistencies/ambiguities, and enrich the entity, ensuring all *genuinely relevant* context informs the improvement process.
    * **Relevant Relationships (`relationships`):** Describes how the problematic entity relates to other entities. Use this to understand its functional role, dependencies, and interactions to clarify its identity and purpose.
        ```json
        {json.dumps(format_relationships, indent=2)}
        ```
    * **Relevant Source Knowledge (`source_data`):** Text snippets related to the entity. Identify and extract *truly valuable details* from these source data to correct, clarify, and enhance the entity's description and metadata. Prioritize information that resolves the identified quality issues.
        ```json
        {json.dumps(selected_source_data, indent=2)}
        ```

## Core Principles for Entity Improvement

Rely on your expert judgment to achieve the following:

1.  **Meaningful Correction and Enhancement:**
    * Prioritize creating a **factually accurate, clear, and high-quality representation** that effectively addresses the identified quality flaws.
    * Preserve and integrate information that is **genuinely significant for rectifying the issues, adding crucial context, or improving understanding**.
    * Resolve discrepancies and ambiguities thoughtfully, aiming for a coherent narrative. If conflicts cannot be definitively resolved with the given information, this should be noted if critical, or the most probable interpretation chosen with justification.
    * All corrections and enhancements MUST be directly supported by the provided background information - never invent or assume facts not present in the input data.
    
2.  **Accuracy, Clarity, and Completeness:**
    * Ensure the improved entity is **unambiguous, logically structured, and easily digestible**.
    * Strive for an optimal balance: comprehensive enough to be authoritative and address the quality flaw, yet concise enough for practical use. **Avoid information overload.**

## Improvement Guidelines (Applying Principles with Strategic Judgment)

Apply the Core Principles to make informed decisions for each aspect of the entity:

1.  **Name Refinement (`name`):**
    * Choose/refine the name to be **precise, unambiguous, and accurately reflecting the entity's now-clarified identity and purpose.**
    * If the original name was a significant identifier despite being flawed, or if other common names exist, document them as aliases in `meta.aliases` to aid discoverability.

2.  **Description Enhancement (`description`):**
    * **Synthesize a new, coherent narrative** that integrates corrections, clarifications, and enriched details from all relevant sources (`entity_to_improve`'s original data, `source_data`, `relationships`).
    * Focus on delivering a **clear, accurate, and comprehensive understanding** of the entity, ensuring it directly addresses and resolves the identified quality issue.
    * Ensure a logical flow and highlight key characteristics.
    * Every statement in the description must be traceable to the background information provided.

3.  **Attributes Augmentation/Correction (`attributes`):**
    * Consolidate and correct attributes. Select, update, or add fields that provide **essential context, provenance, or defining attributes** for the improved entity.
    * Correct any erroneous values based on `source_data` or `relationships`.
    * Add new attributes if they are critical for understanding the entity's corrected definition or provide important context (e.g., a more specific `entity_type`, `data_source_reliability`).
    * Ensure each attribute is meaningful, accurate, and supports the improved entity.
    * All attributes must be derived from or supported by the background information.

## Output Requirements

Return a single JSON object (surrounded by ```json and ```) representing the improved entity. The structure MUST be as follows:

```json
{{
    "name": "...",
    "description": "...",
    "attributes": {{}}
}}
```

Final Check: Before finalizing, review the improved entity:

- Is it a high-quality, useful piece of information?
- Are the original quality issues demonstrably resolved?
- Is it clear, concise, accurate, yet comprehensive?
- Does it truly represent the best understanding of the underlying concept based on the provided information?
- Are all technical terms, identifiers, and features sufficiently contextualized or explained to be understood by a reasonably knowledgeable audience in technologies?

Based on all the provided information and guidelines, exercising your expert judgment, generate the improved entity.
"""

    try:
        token_count = calculate_tokens(improve_entity_quality_prompt)
        print(f"improve entity quality prompt token count: {token_count}")
        response = llm_client.generate(
            improve_entity_quality_prompt, max_tokens=token_count + 1024
            )
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("Failed to improve entity quality", e)
        return None


def process_entity_quality_issue(
    llm_client, entity_model, relationship_model, row_index, row_issue
):
    print(f"start to process entity {row_index}")
    with session_factory() as session:
        try:
            for affected_id in row_issue["affected_ids"]:
                entity_quality_issue = {
                    "issue_type": row_issue["issue_type"],
                    "reasoning": row_issue["reasoning"],
                    "affected_ids": [affected_id],
                }

                print(f"process entity({row_index}), {entity_quality_issue}")

                entities = query_entities_by_ids(
                    session, entity_quality_issue["affected_ids"]
                )
                print(f"Pendding entities({row_index})", entities)
                if len(entities) == 0:
                    print(f"Failed to find entity({row_index}) {affected_id}")
                    return False

                relationships = get_relationship_by_entity_ids(
                    session, entity_quality_issue["affected_ids"]
                )

                source_data_list = get_source_data_by_entity_ids(
                    session, entity_quality_issue["affected_ids"]
                )

                updated_entity = refine_entity(
                    llm_client,
                    entity_quality_issue,
                    entities,
                    relationships,
                    source_data_list,
                )
                print("updated entity", updated_entity)

                if (
                    updated_entity is not None
                    and isinstance(updated_entity, dict)
                    and "name" in updated_entity
                    and "description" in updated_entity
                    and "attributes" in updated_entity
                ):
                    existing_entity = (
                        session.query(entity_model)
                        .filter(entity_model.id == affected_id)
                        .first()
                    )
                    if existing_entity is not None:
                        existing_entity.name = updated_entity["name"]
                        existing_entity.description = updated_entity["description"]
                        new_attributes = updated_entity.get("attributes", {})
                        if isinstance(new_attributes, str):
                            new_attributes = json.loads(new_attributes)
                        # Safely preserve existing topic_name and category
                        existing_attrs = existing_entity.attributes or {}
                        if "topic_name" in existing_attrs:
                            new_attributes["topic_name"] = existing_attrs["topic_name"]
                        if "category" in existing_attrs:
                            new_attributes["category"] = existing_attrs["category"]
                        existing_entity.attributes = new_attributes
                        existing_entity.description_vec = (
                            get_entity_description_embedding(
                                updated_entity["name"], updated_entity["description"]
                            )
                        )
                        session.add(existing_entity)
                        print(
                            f"Success update entity({row_index}) {affected_id} to {updated_entity}"
                        )
                    else:
                        print(
                            f"Failed to find entity({row_index}) {affected_id} to update"
                        )
                        return False
                else:
                    print(
                        f"Failed to refine entity({row_index}), which is invalid or empty."
                    )
                    return False
            session.commit()
        except Exception as e:
            logging.error(f"Failed to refine entity {row_index}: {e}", exc_info=True)
            session.rollback()
            return False

    return True


##### merge entities


def merge_entity(llm_client, issue, entities, relationships, source_data_list):

    format_relationships = []
    consumed_tokens = 0
    for relationship in relationships.values():
        relationship_str = f"""{relationship['source_entity_name']} -> {relationship['target_entity_name']}: {relationship['relationship_desc']}"""
        consumed_tokens += calculate_tokens(relationship_str)
        if consumed_tokens > 20000:
            break
        format_relationships.append(relationship_str)

    # make the token won't exceed 65536
    selected_source_data = []
    for source_data in source_data_list:
        consumed_tokens += calculate_tokens(source_data["content"])
        if consumed_tokens > 70000:
            selected_source_data = selected_source_data[:-1]
            break
        selected_source_data.append(source_data)

    merge_entity_prompt = f"""You are an expert assistant specializing in technologies and knowledge graph curation, tasked with intelligently consolidating redundant entity information into a single, authoritative, and comprehensive entity representation.

## Objective

Your primary goal is to synthesize a **single, high-quality entity** that:
1. **Eliminates redundancy** while preserving all valuable information across different abstraction levels
2. **Handles complex multi-level scenarios** including common-specific, hierarchical, and peer-level entity combinations
3. **Maintains semantic accuracy** based strictly on provided evidence
4. **Optimizes clarity and utility** for knowledge graph applications

## Input Data Structure

### 1. Redundancy Issue (`issue`)
Describes why these entities are considered redundant and need merging.
```json
{json.dumps(issue, indent=2)}
```

### 2. Entities to Merge (`entities`)
A list of entity objects that require consolidation, potentially spanning different abstraction levels.
```json
{json.dumps(entities, indent=2)}
```

### 3. Background Information
Additional context to enrich the merged entity:

#### Relevant Relationships (`relationships`)
Describes how entities relate to other entities in the knowledge graph.
```json
{json.dumps(format_relationships, indent=2)}
```

#### Relevant Source Knowledge (`source_data`)
Text snippets related to the entities for context enhancement.
```json
{json.dumps(selected_source_data, indent=2)}
```

## Core Principles for Merging

### 1. Abstraction Level Analysis Strategy
- **Level Identification**: Recognize different abstraction levels (concept → product → version → instance)
- **Hierarchical Mapping**: Understand how entities relate across abstraction boundaries
- **Optimal Scope Selection**: Choose the abstraction level that maximizes knowledge graph utility
- **Information Layering**: Preserve valuable details from all levels without losing semantic clarity

### 2. Context-Driven Integration Approach
- **Evidence-Based Prioritization**: Use `source_data` and `relationships` to guide integration decisions
- **Usage Pattern Analysis**: Consider how merged entity will be used in the knowledge graph
- **Connection Density Evaluation**: Prioritize entities with more relationships and contextual relevance
- **Semantic Coherence**: Ensure the merged result represents a coherent, unified concept

### 3. Information Synthesis and Quality Enhancement
- **Comprehensive Integration**: Combine all unique and valuable information from source entities
- **Conflict Resolution**: When information conflicts, prioritize the most specific and evidence-supported version
- **Redundancy Elimination**: Remove duplicate information while preserving unique insights from each source
- **Contextual Enrichment**: Add meaningful structure that enhances understanding across abstraction levels

## Advanced Merging Guidelines

### Multi-Level Entity Analysis Framework

#### Step 1: Entity Relationship Classification
Identify the relationship pattern among entities to merge:
- **Pure Hierarchical**: Concept → Specific → Instance (Database → MySQL → MySQL 8.0 → prod-instance)
- **Peer-Level Variants**: Similar abstraction entities with different focuses
- **Mixed Complexity**: Combination of hierarchical and peer relationships
- **Specialization Chain**: Progressive refinement of the same core concept

#### Step 2: Optimal Abstraction Level Selection
Apply decision criteria to choose the primary abstraction level:

**Selection Priority Rules:**
1. **Relationship Density**: Entity with most connections in the knowledge graph
2. **Information Richness**: Entity providing the most comprehensive and useful context
3. **Usage Frequency**: Most commonly referenced entity in `source_data`
4. **Semantic Centrality**: Entity that best represents the core concept across all levels

#### Step 3: Information Integration Strategy
Based on the selected primary level, apply appropriate integration approach:

**For Concept-Level Primary**: Integrate specific details as contextual examples and variants
**For Product-Level Primary**: Balance general context with specific implementation details
**For Instance-Level Primary**: Maintain specific focus while providing broader conceptual context
**For Mixed Scenarios**: Create layered descriptions that acknowledge multiple valid perspectives

### Advanced Attribute Consolidation

#### Multi-Level Attribute Handling
- **Level-Specific Grouping**: Organize attributes by abstraction level when beneficial
- **Hierarchical Preservation**: Use nested structures to maintain level relationships
- **Conflict Resolution**: Apply evidence-based priority rules for conflicting attribute values
- **Semantic Enhancement**: Add attributes that clarify scope and abstraction boundaries

#### Attribute Integration Strategies
- **Union Strategy**: Combine non-conflicting attributes from all sources
- **Layered Strategy**: Group attributes by abstraction level for complex scenarios
- **Contextual Strategy**: Prioritize attributes that enhance understanding of the merged entity's scope
- **Traceability Strategy**: Maintain connection to original abstraction levels when valuable

## Output Requirements

Return a single JSON object representing the merged entity (surrounding by ```json and ```):

```json
{{
  "name": "Optimally selected name that represents the merged entity's scope and primary abstraction level",
  "description": "Comprehensive, layered description that integrates information from all source entities while maintaining semantic coherence. Should flow logically from general context to specific details, clearly indicating the scope and abstraction boundaries of the merged entity.",
  "attributes": {{
    // Strategically consolidated attributes that enhance understanding
    // May include level-specific groupings for complex multi-level scenarios
    // All attributes must be evidence-supported and add meaningful value
    // Consider using nested structures for hierarchical information when beneficial
  }}
}}
```

Based on all the provided information and guidelines, exercising your expert judgment, generate the merged entity."""

    try:
        token_count = calculate_tokens(merge_entity_prompt)
        print(f"merge entity prompt token count: {token_count}")
        response = llm_client.generate(merge_entity_prompt, max_tokens=token_count + 1024)
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("Failed to merge entity", e, response)
        return None


def process_redundancy_entity_issue(
    llm_client, entity_model, relationship_model, source_graph_mapping_model, row_key, row_issue
):
    print(f"start to merge entity({row_key}) for {row_issue}")
    with session_factory() as session:
        try:
            entities = query_entities_by_ids(session, row_issue["affected_ids"])
            print(f"pending entities({row_key})", entities)
            if len(entities) == 0:
                print(f"Failed to find entity({row_key}) {row_issue['affected_ids']}")
                return False

            relationships = get_relationship_by_entity_ids(
                session, row_issue["affected_ids"]
            )
            source_data_list = get_source_data_by_entity_ids(
                session, row_issue["affected_ids"]
            )

            merged_entity = merge_entity(
                llm_client, row_issue, entities, relationships, source_data_list
            )
            print(f"merged entity({row_key}) {merged_entity}")

            if (
                merged_entity is not None
                and isinstance(merged_entity, dict)
                and "name" in merged_entity
                and "description" in merged_entity
                and "attributes" in merged_entity
            ):
                new_entity = entity_model(
                    name=merged_entity["name"],
                    description=merged_entity["description"],
                    attributes=merged_entity.get("attributes", {}),
                    description_vec=get_entity_description_embedding(
                        merged_entity["name"], merged_entity["description"]
                    ),
                )
                session.add(new_entity)
                session.flush()
                merged_entity_id = new_entity.id
                print(
                    f"Merged entity({row_key}) created with ID: {new_entity.name}({merged_entity_id})"
                )
                original_entity_ids = {entity["id"] for entity in entities.values()}
                # Step 2: Update relationships to reference the merged entity
                # Bulk update source entity IDs
                session.execute(
                    relationship_model.__table__.update()
                    .where(relationship_model.source_entity_id.in_(original_entity_ids))
                    .values(source_entity_id=merged_entity_id)
                )

                # Bulk update target entity IDs
                session.execute(
                    relationship_model.__table__.update()
                    .where(relationship_model.target_entity_id.in_(original_entity_ids))
                    .values(target_entity_id=merged_entity_id)
                )
                # step 3: update source graph mapping table
                session.execute(
                    source_graph_mapping_model.__table__.update()
                    .where(
                        (source_graph_mapping_model.graph_element_id.in_(original_entity_ids)) &
                        (source_graph_mapping_model.graph_element_type == "entity")
                    )
                    .values(graph_element_id=merged_entity_id)
                )

                # step 4: delete original entities after all references are updated
                session.execute(
                    entity_model.__table__.delete().where(
                        entity_model.id.in_(original_entity_ids)
                    )
                )

                print(
                    f"Relationships and source mappings updated, original entities deleted for merged entity({row_key}) {merged_entity_id}"
                )

                session.commit()  # Commit the relationship updates
                print(f"Merged entity({row_key}) processing complete.")
                return True
            else:
                print(f"Failed to merge entity({row_key}), which is invalid or empty.")
                return False
        except Exception as e:
            logging.error(f"Failed to merge entity({row_key}): {e}", exc_info=True)
            session.rollback()
            return False


##### refine relationship quality


def refine_relationship_quality(llm_client, issue, entities, relationships, source_data_list):
    format_relationships = []
    consumed_tokens = 0
    for relationship in relationships.values():
        relationship_str = f"""{relationship['source_entity_name']} -> {relationship['target_entity_name']}: {relationship['relationship_desc']}"""
        consumed_tokens += calculate_tokens(relationship_str)
        if consumed_tokens > 30000:
            break
        format_relationships.append(relationship_str)

    consumed_tokens = calculate_tokens(json.dumps(format_relationships, indent=2))
    selected_source_data = []
    for source_data in source_data_list:
        consumed_tokens += calculate_tokens(source_data["content"])
        if consumed_tokens > 70000:
            selected_source_data = selected_source_data[:-1]
            break
        selected_source_data.append(source_data)

    refine_relationship_quality_prompt = f"""You are an expert assistant specializing in technologies and knowledge graph curation, tasked with rectifying quality issues within a single relationship to ensure its meaning is clear, accurate, and truthful by providing an improved description and optimized attributes.

## Objective

Your primary goal is to analyze a problematic relationship and its surrounding context to craft:
1. **An accurate, coherent, and semantically meaningful textual description** of the connection between source and target entities
2. **Well-curated relationship attributes** that provide valuable metadata and context

Both improvements must correct identified flaws (like vagueness or ambiguity) and be **strictly based on evidence**, avoiding any speculation. The aim is to produce a relationship that is genuinely useful and unambiguous for a knowledgeable audience.

## Input Data

You will be provided with the following information:

1.  **Relationship Quality Issue (`issue`):** Describes the specific quality problem(s) with the relationship's existing description or definition that needs to be addressed. Your primary task is to generate a new description that resolves these problems.
    ```json
    {json.dumps(issue, indent=2)}
    ```

2.  **Relationship to Improve (`relationship_to_improve`):** The relationship object whose description requires quality improvement.
    ```json
    {json.dumps(format_relationships, indent=2)}
    ```

3.  **Background Information:** Use this to gain a deep understanding of the context, resolve ambiguities/contradictions, and formulate the improved description. **The new description MUST be justifiable by this background information.**

    * **Relevant Knowledge (`source_data`):** Text snippets related to the relationship itself or its connected entities. Extract **verifiable details** from these chunks to formulate the improved description.
        ```json
        {json.dumps(selected_source_data, indent=2)}
        ```

## Core Principles for Relationship Improvement

### Description Enhancement Principles

1.  **Meaningful Clarification & Semantic Accuracy**: The description must make the relationship's purpose and the nature of the connection explicit and precise.
2.  **Truthfulness and Evidence-Based Refinement**: This is paramount. The improved description MUST be directly supported by evidence found in the `source_data`.
3.  **Clarity, Unambiguity, and Utility**: Ensure the improved description is easily understandable, its meaning is singular and well-defined.

### Attribute Quality Principles

1. **Relevance Check**: Remove attributes unrelated to the relationship's core meaning
2. **Accuracy Verification**: Ensure attribute values are based on provided evidence
3. **Completeness Enhancement**: Add valuable missing attributes based on source data
4. **Consistency Assurance**: Maintain consistent attribute naming and value formats


## Guidelines for Relationship Improvement

Step 1: Deep Analysis of Quality Issues - Thoroughly understand the specific flaw(s) described in the `Relationship Quality Issue`. Identify what aspects of both description and attributes need improvement

Step 2: Comprehensive Contextual Understanding - Synthesize information from the existing relationship data and relevant `source_data`. Extract verifiable facts that can support both description and attribute improvements

Step 3: Crafting the Improved Description - Create a **clear, concise, and evidence-based narrative** that explains *precisely how* the source entity connects to or interacts with the target entity. Articulate the nature, purpose, and, if applicable, the direction or mechanism of the connection. Directly address and resolve the issues raised in `Relationship Quality Issue`

Step 4: Optimizing Relationship Attributes - Apply the following strategies based on evidence from `source_data`. Attribute Processing Strategies:

- **Retain**: Existing attributes that are evidence-supported and meaningful
- **Correct**: Inaccurate attribute values based on source data
- **Add**: Valuable new attributes extracted from source data
- **Remove**: Attributes lacking evidence support or relevance


## Output Requirements

Return a single JSON object representing the improved relationship. The structure MUST be as follows:

```json
{{
    "source_entity_name": "...", # use the entity name in the `relationship_to_improve`
    "target_entity_name": "...", # use the entity name in the `relationship_to_improve`
    "relationship_desc": "...",
    "attributes": {{
        // Curated attributes based on evidence from source_data
        // Include only attributes that add meaningful value and context
        // If no valid attributes can be derived from evidence, use empty object
    }}
}}
```

Based on all the provided information and guidelines, exercising your expert judgment with a strict adherence to truthfulness, generate **only the new, improved relationship description string.**
"""

    try:
        token_count = calculate_tokens(refine_relationship_quality_prompt)
        print(f"refine relationship quality prompt token count: {token_count}")
        response = llm_client.generate(refine_relationship_quality_prompt, max_tokens=token_count + 1024)
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("Failed to refine relationship quality", e)
        return None


def process_relationship_quality_issue(
    llm_client, relationship_model, row_key, row_issue
):
    print(f"start to process relationship({row_key})")
    with session_factory() as session:
        try:
            for affected_id in row_issue["affected_ids"]:
                relationship_quality_issue = {
                    "issue_type": row_issue["issue_type"],
                    "reasoning": row_issue["reasoning"],
                    "affected_ids": [affected_id],
                }

                print(f"process relationship({row_key}), {relationship_quality_issue}")

                relationships = get_relationship_by_ids(
                    session, relationship_quality_issue["affected_ids"]
                )
                print(f"Pendding relationships({row_key})", relationships)
                if len(relationships) == 0:
                    print(f"Failed to find relationship({row_key}) {affected_id}")
                    return False

                source_data_list = get_source_data_by_relationship_ids(
                    session, relationship_quality_issue["affected_ids"]
                )

                updated_relationship = refine_relationship_quality(
                    llm_client, relationship_quality_issue, [], relationships, source_data_list
                )
                print("updated relationship", updated_relationship)

                if (
                    updated_relationship is not None
                    and isinstance(updated_relationship, dict)
                    and "relationship_desc" in updated_relationship
                ):
                    existing_relationship = (
                        session.query(relationship_model)
                        .filter(relationship_model.id == affected_id)
                        .first()
                    )
                    if existing_relationship is not None:
                        existing_relationship.relationship_desc = updated_relationship[
                            "relationship_desc"
                        ]
                        existing_relationship.relationship_desc_vec = get_text_embedding(
                            updated_relationship["relationship_desc"]
                        )
                        # Update attributes if provided, preserving important existing fields
                        if "attributes" in updated_relationship:
                            new_attributes = updated_relationship["attributes"] or {}
                            if isinstance(new_attributes, str):
                                new_attributes = json.loads(new_attributes)
                            # Safely preserve existing important attributes
                            existing_attrs = existing_relationship.attributes or {}
                            # Preserve common important fields that should not be lost
                            important_fields = ["topic_name", "category"]
                            for field in important_fields:
                                if field in existing_attrs and field not in new_attributes:
                                    new_attributes[field] = existing_attrs[field]
                            existing_relationship.attributes = new_attributes
                        # If no new attributes provided, keep existing ones unchanged
                        session.add(existing_relationship)
                        print(
                            f"Success update relationship({row_key}) {affected_id} to {updated_relationship}"
                        )
                    else:
                        print(f"Failed to find relationship({row_key}) {affected_id}")
                        return False
                else:
                    print(
                        f"Failed to refine relationship({row_key}), which is invalid or empty."
                    )
                    return False
            session.commit()
        except Exception as e:
            logging.error(
                f"Failed to refine relationship {row_key}: {e}", exc_info=True
            )
            session.rollback()
            return False

    return True


##### merge redundancy relationship


def merge_relationship(llm_client, issue, entities, relationships, source_data_list):
    format_relationships = []
    consumed_tokens = 0
    for relationship in relationships.values():
        relationship_str = f"""{relationship['source_entity_name']}(source_entity_id={relationship['source_entity_id']}) -> {relationship['target_entity_name']}(target_entity_id={relationship['target_entity_id']}): {relationship['relationship_desc']}"""
        consumed_tokens += calculate_tokens(relationship_str)
        if consumed_tokens > 30000:
            break
        format_relationships.append(relationship_str)

    consumed_tokens = calculate_tokens(json.dumps(format_relationships, indent=2))

    # make the token won't exceed 65536
    selected_source_data = []
    for source_data in source_data_list:
        consumed_tokens += calculate_tokens(source_data["content"])
        if consumed_tokens > 70000:
            selected_source_data = selected_source_data[:-1]
            break
        selected_source_data.append(source_data)

    merge_relationship_prompt = f"""You are an expert assistant specializing in technologies and knowledge graph curation, tasked with intelligently consolidating redundant relationship information into a single, authoritative, and comprehensive relationship entry.

## Objective

Your primary goal is to synthesize a **single, authoritative, and structured relationship** from multiple redundant entries that:
1. **Eliminates redundancy** while preserving all valuable information
2. **Enhances clarity and comprehensiveness** beyond any individual source entry
3. **Maintains semantic accuracy** based strictly on provided evidence
4. **Provides structured attributes** that add meaningful context

## Input Data

You will be provided with the following information:

### 1. Redundancy Issue (`issue`)

Describes why these relationship entries are considered redundant and need merging.
```json
{json.dumps(issue, indent=2)}
```

### 2. Relationships to Merge (`relationships_to_merge`)

A list of relationship entries that require merging. Each entry contains basic relationship information with potential variations in descriptions and attributes.
```json
{json.dumps(format_relationships, indent=2)}
```

### 3. Background Information (`source_data`)

Text snippets related to the entities and their interactions. Use this as your **sole source of external information** for enriching the merged relationship.
```json
{json.dumps(selected_source_data, indent=2)}
```

## Core Principles for Merging Relationships

### 1. Information Synthesis Strategy

- **Comprehensive Integration**: Combine all unique and valuable information from source relationships
- **Conflict Resolution**: When descriptions conflict, prioritize the most specific and evidence-supported version
- **Evidence-Based Enhancement**: Use `source_data` to resolve ambiguities and add context
- **Conservative Inference**: Only infer details that are clearly supported by provided evidence

### 2. Quality Enhancement Principles

- **Clarity Optimization**: Create descriptions that are more precise and understandable than individual sources
- **Semantic Accuracy**: Ensure the merged relationship accurately represents the underlying connection
- **Contextual Enrichment**: Add meaningful attributes that enhance relationship understanding
- **Redundancy Elimination**: Remove duplicate or overlapping information while preserving unique insights

### 3. Attribute Consolidation Strategy

- **Union**: Combine non-conflicting attributes from all sources
- **Resolution**: For conflicting attribute values, choose the most evidence-supported option
- **Enhancement**: Add new attributes derived from `source_data` when clearly justified
- **Standardization**: Ensure consistent naming and formatting across merged attributes

### 4. Priority Rules for Conflicts

1. **Evidence-supported** information takes precedence
2. **More specific** details override generic ones
3. **Recent or updated** information preferred when timestamps available
4. **Consensus** across multiple sources increases reliability

## Output Requirements

Return a single JSON object representing the merged relationship (surrounding by ```json and ```):

The structure MUST be as follows:

```json
{{
  "source_entity_id": "...", // entity id from input
  "target_entity_id": "...", // entity id from input
  "relationship_desc": "...",      // Merged/synthesized relationship description
  "attributes": {{
    // Merged and optimized attributes from all source relationships
    // Include only attributes that add meaningful value
    // Resolve conflicts based on evidence and priority rules
    // Add new attributes only when clearly supported by source_data
  }}
}}
```

Based on all the provided information and guidelines, exercising your expert judgment to infer and synthesize within the given constraints, generate the merged relationship.
"""

    try:
        token_count = calculate_tokens(merge_relationship_prompt)
        print(f"merge relationship prompt token count: {token_count}")
        response = llm_client.generate(merge_relationship_prompt, max_tokens=token_count + 1024)
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("Failed to merge relationship", e)
        return None


def process_redundancy_relationship_issue(
    llm_client, relationship_model, source_graph_mapping_model, row_key, row_issue
):
    print(f"start to merge relationships {row_key} for {row_issue}")
    with session_factory() as session:
        try:
            relationships = get_relationship_by_ids(session, row_issue["affected_ids"])
            print(f"pending relationships({row_key})", relationships)
            if len(relationships) < 2:
                print(
                    f"skip, not enough relationships to merge - ({row_key}) {row_issue['affected_ids']}"
                )
                return True

            entity_pairs = set()
            for relationship in relationships.values():
                print(f"relationship: {relationship}")
                entity_pairs.add(relationship["source_entity_id"])
                entity_pairs.add(relationship["target_entity_id"])

            if len(entity_pairs) != 1 and len(entity_pairs) != 2:
                print(
                    f"skip, incapabble to merge relationship between different entities - ({row_key}) {relationships}"
                )
                return True

            source_data_list = get_source_data_by_relationship_ids(
                session, row_issue["affected_ids"]
            )

            merged_relationship = merge_relationship(
                llm_client, row_issue, [], relationships, source_data_list
            )
            print("merged relationship", merged_relationship)

            # Get the actual entity IDs from the original relationships (they should all be the same)
            first_relationship = next(iter(relationships.values()))

            if (
                merged_relationship is not None
                and isinstance(merged_relationship, dict)
                and "relationship_desc" in merged_relationship
            ):
                candidate_source_entity_id = first_relationship["source_entity_id"]
                candidate_target_entity_id = first_relationship["target_entity_id"]

                actual_source_entity_id = candidate_source_entity_id
                if merged_relationship[
                    "source_entity_id"
                ] is not None and merged_relationship["source_entity_id"] in (
                    candidate_source_entity_id,
                    candidate_target_entity_id,
                ):
                    actual_source_entity_id = merged_relationship["source_entity_id"]

                # other candidate entity id
                actual_target_entity_id = candidate_target_entity_id
                if actual_source_entity_id == candidate_target_entity_id:
                    actual_target_entity_id = candidate_source_entity_id

                # Merge attributes intelligently
                merged_attributes = merged_relationship.get("attributes", {}) or {}
                new_relationship = relationship_model(
                    source_entity_id=actual_source_entity_id,
                    target_entity_id=actual_target_entity_id,
                    relationship_desc=merged_relationship["relationship_desc"],
                    relationship_desc_vec=get_text_embedding(
                        merged_relationship["relationship_desc"]
                    ),
                    attributes=merged_attributes,
                )
                session.add(new_relationship)
                session.flush()
                merged_relationship_id = new_relationship.id
                print(
                    f"Merged relationship created with ID: {new_relationship.source_entity_id} -> {new_relationship.target_entity_id}({merged_relationship_id})"
                )
                original_relationship_ids = {
                    relationship["id"] for relationship in relationships.values()
                }
                
                # Step 1: Update source graph mapping table before deleting original relationships
                session.execute(
                    source_graph_mapping_model.__table__.update()
                    .where(
                        (source_graph_mapping_model.graph_element_id.in_(original_relationship_ids)) &
                        (source_graph_mapping_model.graph_element_type == "relationship")
                    )
                    .values(graph_element_id=merged_relationship_id)
                )
                
                # Step 2: Remove the original relationships
                session.execute(
                    relationship_model.__table__.delete().where(
                        relationship_model.id.in_(original_relationship_ids)
                    )
                )

                print(f"Source mappings updated and deleted {len(original_relationship_ids)} relationships")
                session.commit()  # Commit the relationship updates
                print(f"Merged relationship {row_key} processing complete.")
                return True
            else:
                print(
                    f"Failed to merge relationship({row_key}), which is invalid or empty."
                )
                return False
        except Exception as e:
            logging.error(f"Failed to merge relationship {row_key}: {e}", exc_info=True)
            session.rollback()
            return False
