import json
import logging

from setting.db import SessionLocal
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

    improve_entity_quality_prompt = f"""You are an expert assistant specializing in database technologies and knowledge graph curation, tasked with rectifying quality issues within a single entity.

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

Return a single JSON object representing the improved entity. The structure MUST be as follows:

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
- Are all technical terms, identifiers, and features sufficiently contextualized or explained to be understood by a reasonably knowledgeable audience in database technologies?

Based on all the provided information and guidelines, exercising your expert judgment, generate the improved entity.
"""

    try:
        token_count = calculate_tokens(improve_entity_quality_prompt)
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
    with SessionLocal() as session:
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
        if consumed_tokens > 30000:
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

    merge_entity_prompt = f"""You are an expert assistant specializing in database technologies, tasked with intelligently consolidating redundant entity information.

    ## Objective

    Your primary goal is to synthesize a **single, authoritative, and high-quality entity** from a group of redundant ones. This merged entity should be more comprehensive, coherent, **meaningful, and self-contained** than any individual source entity. It's not just about combining data, but about creating a **genuinely improved representation** that effectively reduces redundancy while maximizing clarity, utility, and **ease of understanding for a knowledgeable audience (which may include those not deeply expert in every specific nuance).** Prioritize information significance, contextual accuracy, and overall comprehensibility.

    ## Input Data

    You will be provided with the following information:

    1.  **Redundancy Issue (`issue`):** Describes why these entities are considered redundant and need merging.
        ```json
        {json.dumps(issue, indent=2)}
        ```

    2.  **Entities to Merge (`entities`):** A list of the entity objects that require merging.
        ```json
        {json.dumps(entities, indent=2)}
        ```

    3.  **Background Information:** Use this to gain a deeper understanding and to enrich the merged entity, ensuring all *genuinely relevant* context is captured or informs the synthesis.
        * **Relevant Relationships (`relationships`):** Describes how the redundant entities relate to other entities. Use this to understand the broader context and to inform the selection and presentation of relational insights within the description if they add significant value.
            ```json
            {json.dumps(format_relationships, indent=2)}
            ```
        * **Relevant Source Knowledge (`source_data`):** Text snippets related to the entities. Identify and extract *truly valuable details* from these chunks to enhance the merged description and metadata, avoiding trivial or overly specific information unless critical.
            ```json
            {json.dumps(selected_source_data, indent=2)}
            ```

    ## Core Principles for Merging

    Rely on your expert judgment to achieve the following:

    1.  **Meaningful Synthesis for Enhanced Understanding:**
        * Prioritize creating a **holistic, accurate, and high-quality representation** that is more valuable than the sum of its parts.
        * Preserve information that is **genuinely significant, unique, or offers crucial context**. Critically assess if all details add value or if some can be omitted for clarity and conciseness.
        * Resolve discrepancies thoughtfully, aiming for a coherent narrative that explains differing perspectives if they are important for a complete understanding.
        * All enhancements MUST be directly supported by the provided background information - never invent or assume facts not present in the input data.

    2.  **Clarity, Coherence, and Utility:**
        * Ensure the merged entity is **clear, logically structured, and easily digestible**.
        * Strive for an optimal balance: comprehensive enough to be authoritative, yet concise enough for practical use. **Avoid information overload and undue complexity.**

    ## Merging Guidelines (Applying Principles with Strategic Judgment)

    Apply the Core Principles to make informed decisions for each aspect of the entity:

    1.  **Name Selection (`name`):**
        * Choose the **most representative, widely recognized, and unambiguous name**. Document essential aliases in `meta` if they significantly aid discoverability or understanding.

    2.  **Description Crafting (`description`):**
        * **Synthesize a new, coherent narrative** that integrates the most critical and insightful information from all relevant sources (entities, source_data, relationships).
        * Focus on delivering a **clear and comprehensive understanding** of the entity, ensuring a logical flow and highlighting key characteristics.
        * Every statement in the description must be traceable to the background information provided.

    3.  **Attributes Curation (`attributes`):**
        * Consolidate metadata by selecting fields that provide **essential context, provenance, or defining attributes**.
        * Handle differing values by prioritizing what is most current, relevant, or representative for the merged entity, using arrays or notes for important, non-conflicting variations or unavoidable ambiguities. Be selective to ensure metadata supports, rather than clutters, the entity.
        * Ensure each attribute can be understood independently without background information, and is meaningful for the entity.
        * All attributes must be derived from or supported by the background information.

    ## Output Requirements

    Return a single JSON object representing the merged entity. The structure MUST be as follows:

    ```json
    {{
    "name": "...",
    "description": "...",
    "attributes": {{}}
    }}
    ```

    **Final Check:** Before finalizing, review the merged entity:
    * Is it a high-quality, useful piece of information?
    * Is it clear, concise, yet comprehensive?
    * Does it truly represent the best understanding of the underlying concept?
    * **Are all technical terms, identifiers, and features sufficiently contextualized or explained to be understood by a reasonably knowledgeable audience in database technologies?**

    Based on all the provided information and guidelines, exercising your expert judgment, generate the merged entity.
    """

    try:
        token_count = calculate_tokens(merge_entity_prompt)
        print(f"merge entity prompt token count: {token_count}")
        response = llm_client.generate(merge_entity_prompt, max_tokens=8192)
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("Failed to merge entity", e)
        return None


def process_redundancy_entity_issue(
    llm_client, entity_model, relationship_model, source_graph_mapping_model, row_key, row_issue
):
    print(f"start to merge entity({row_key}) for {row_issue}")
    with SessionLocal() as session:
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

    refine_relationship_quality_prompt = f"""You are an expert assistant specializing in database technologies and knowledge graph curation, tasked with rectifying quality issues within a single relationship to ensure its meaning is clear, accurate, and truthful by providing an improved description.

## Objective

Your primary goal is to analyze a problematic relationship and its surrounding context to craft an **accurate, coherent, and semantically meaningful textual description of the connection** between its source and target entities. This improved description must correct identified flaws (like vagueness or ambiguity) and be **strictly based on evidence**, avoiding any speculation. The aim is to produce a description that makes the relationship genuinely useful and unambiguous for a knowledgeable audience.

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

## Core Principles for Crafting the Relationship Description

Rely on your expert judgment, guided by the following principles:

1.  **Meaningful Clarification & Semantic Accuracy:** The description must make the relationship's purpose and the nature of the connection explicit and precise. It should accurately reflect how the entities interact or are associated.
2.  **Truthfulness and Evidence-Based Refinement:** **This is paramount.** The improved description MUST be directly supported by evidence found in the `source_data`. **Do NOT invent details, make assumptions, or infer beyond what the provided context clearly indicates.**
3.  **Clarity, Unambiguity, and Utility:** Ensure the improved description is easily understandable, its meaning is singular and well-defined, and it provides genuine insight into the connection. Avoid overly generic terms if evidence supports specificity.

## Guidelines for Formulating the Improved Description

1.  **Deep Analysis of `Relationship Quality Issue`:** Thoroughly understand the specific flaw(s) described in `Relationship Quality Issue` concerning the relationship's clarity or meaning. This is the problem your new description must solve.
2.  **Comprehensive Contextual Understanding:** Before formulating the description, synthesize information from `relationship_to_improve`'s existing data and relevant `source_data`.
3.  **Crafting the New Relationship `description`:**
    * This is your sole output. It must be a **clear, concise, and evidence-based narrative** that explains *precisely how* the source entity connects to or interacts with the target entity.
    * Clearly articulate the nature, purpose, and, if applicable, the direction or mechanism of the connection. For example, instead of "System A affects System B," a better description (if supported by evidence) might be "System A sends real-time transaction data to System B for fraud analysis."
    * Ensure the new description directly addresses and resolves the issues raised in `Relationship Quality Issue`.


## Output Requirements

Return a single JSON object representing the improved relationship. The structure MUST be as follows:

```json
{{
"source_entity_name": "...", # use the entity name in the `relationship_to_improve`
"target_entity_name": "...", # use the entity name in the `relationship_to_improve`
"relationship_desc": "...",
"attributes": {{}}
}}
```

## Final Check: Before finalizing the description string, mentally review:

* Is the fundamental meaning of the relationship, as conveyed by this description, now clear, precise, and unambiguous?
* Does this description accurately capture the nature of the connection between the source and target entities, based *only* on the provided evidence?
* Is this description truthful and directly verifiable from the input context?
* Does this description directly and thoroughly address all problems outlined in `Relationship Quality Issue` regarding the relationship's clarity or meaning?
* Is this description, on its own, genuinely useful and easily understood by a knowledgeable audience in database technologies without needing to guess the relationship's meaning?
* Has any information been invented or inferred beyond what the evidence supports to create this description? (This should be avoided).

Based on all the provided information and guidelines, exercising your expert judgment with a strict adherence to truthfulness, generate **only the new, improved relationship description string.**
"""

    try:
        response = llm_client.generate(refine_relationship_quality_prompt)
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
    with SessionLocal() as session:
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

    merge_relationship_prompt = f"""You are an expert assistant specializing in database technologies and knowledge graph curation, tasked with intelligently consolidating redundant relationship information that is primarily described through simple textual statements.

## Objective

Your primary goal is to synthesize a **single, authoritative, and structured relationship** from a group of redundant descriptive entries. These entries connect the same source and target entities (identified by name) with what is presumed to be the same underlying semantic meaning. The merged relationship – which will include a common `source_entity_name`, a common `target_entity_name`, a synthesized `description` – should be more comprehensive, coherent, **meaningful, and well-defined** than any individual source entry. This task involves transforming multiple simple textual descriptions of a connection into a single, richer, structured representation, effectively reducing redundancy while maximizing clarity, utility, and **ease of understanding for a knowledgeable audience.** Prioritize information significance, contextual accuracy, and overall comprehensibility of the connection. **All inferences and syntheses must be strictly based on the provided evidence.**

## Input Data

You will be provided with the following information:

1.  **Redundancy Issue (`issue`):** Describes why these relationship entries are considered redundant and need merging.
    ```json
    {json.dumps(issue, indent=2)}
    ```

2.  **Relationships to Merge (`relationships_to_merge`):** A list of relationship entries that require merging. **Each entry in this list is a simple object, identified by `source_entity_name` and `target_entity_name`, and containing a `description` string. These input entries LACK explicit `type`, `source_entity_id`, `target_entity_id`, or structured `properties` fields.** They are presumed to describe the same conceptual connection between the named entities.
    ```json
    {json.dumps(format_relationships, indent=2)}
    ```

3.  **Background Information:** Use this to gain a deeper understanding of the context. This is your **sole source of external information** beyond the `relationships_to_merge` themselves for inferring structural elements and enriching the merged relationship.
    * **Relevant Knowledge (`source_data`):** Text snippets related to the named entities, their potential interactions, or relevant schemas/ontologies. Identify and extract *truly valuable details* from these chunks to help infer the relationship `type`, synthesize the `description`, and derive any relevant `properties`.
        ```json
        {json.dumps(selected_source_data, indent=2)}
        ```

## Core Principles for Merging Relationships

Rely on your expert judgment to achieve the following:

1.  **Meaningful Synthesis and Structuring from Limited Input:**
    * Prioritize creating a **holistic, accurate, and high-quality structured representation of the semantic link**, even from simple descriptive inputs. The merged output should be more valuable than the sum of its parts.
    * Preserve information from all source descriptions that is **genuinely significant, unique, or offers crucial context to the connection itself**.
    * All aspects of the merged relationship MUST be directly supported by the provided `relationships_to_merge` descriptions and `source_data` – **never invent or assume facts not present. Be conservative with inferences if evidence is weak.**

2.  **Clarity, Coherence, and Utility:**
    * Ensure the synthesized `description` is **clear, its semantic meaning well-defined, logically structured, and easily digestible**.
    * Strive for an optimal balance: comprehensive enough to be authoritative, yet concise.

## Output Requirements

Return a single JSON object representing the merged relationship. This object will use the common `source_entity_name` and `target_entity_name` from the input.

The structure MUST be as follows:

```json
{{
  "source_entity_id": "...", // entity id from input
  "target_entity_id": "...", // entity id from input
  "relationship_desc": "...",      // Merged/synthesized relationship description
  "attributes": {{}}               // Merged/synthesized relationship attributes
}}
```

## Final Check: Before finalizing, review the merged relationship:

- Is the inferred type semantically accurate and well-justified by the limited context? Is it appropriately general if specific evidence was lacking?
- Is the synthesized description clear, comprehensive, and faithful to the combined evidence from input descriptions and source_data?
- Does the entire merged relationship accurately represent the single best understanding of the underlying connection, given the input constraints?
- Has redundancy from the input descriptions been effectively consolidated?
- Are all aspects of the merged relationship directly supported by the provided relationships_to_merge and source_data, without invention?

Based on all the provided information and guidelines, exercising your expert judgment to infer and synthesize within the given constraints, generate the merged relationship.
"""

    try:
        response = llm_client.generate(merge_relationship_prompt, max_tokens=8192)
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
    with SessionLocal() as session:
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
