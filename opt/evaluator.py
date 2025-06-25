import json
import pandas as pd
import logging

from utils.json_utils import robust_json_parse

logger = logging.getLogger(__name__)


def batch_evaluate_issues(critic_clients, issue_pkls_file_path):
    issue_df = pd.read_pickle(issue_pkls_file_path)
    for critic_name in critic_clients.keys():
        logger.info(f"Evaluating issues for {critic_name}, issue_df: {issue_df.shape}")
        critic_client = critic_clients[critic_name]
        issue_df = evaluate_issue(critic_name, critic_client, issue_df)
        issue_df.to_pickle(issue_pkls_file_path)

    return issue_df


def evaluate_issue(critic_name, critic_client, issue_df):
    for index, row in issue_df.iterrows():
        if "issue" in row:
            issue = row["issue"]
        else:
            issue = row
        if row[critic_name] is not None:
            skip = True
            if row[critic_name] is not None:
                try:
                    robust_json_parse(row[critic_name], "object")
                    # logger.info(f"skip processed issue {index} for {critic_name}", issue['issue_type'], issue['affected_ids'], row['confidence'])
                except:
                    logger.error(
                        "Failed to parse critique (index=%s, critic_name=%s), issue_type=%s, affected_ids=%s, confidence=%s",
                        index,
                        critic_name,
                        issue["issue_type"],
                        issue["affected_ids"],
                        row["confidence"],
                    )
                    skip = False

            if skip:
                continue

        if issue["issue_type"] in ("redundancy_entity", "entity_quality_issue"):
            critic_object = f"affected entities: {issue['affected_ids']}"
        elif issue["issue_type"] in (
            "redundancy_relationship",
            "relationship_quality_issue",
        ):
            critic_object = f"affected relationships: {issue['affected_ids']}"

        if issue["issue_type"] == "redundancy_entity":
            guideline = """**Redundant Entities**(redundancy_entity):

- Definition: Two or more distinct entity entries represent the exact same real-world entity or concept (identical in type and instance).
- Identification: Look for highly similar names, aliases, and descriptions that clearly refer to the same thing without meaningful distinction.
- Exclusion: Do not flag entities as redundant if they represent different levels in a clear hierarchy (e.g., "Artificial Intelligence" vs. "Machine Learning") or distinct concepts that happen to be related (e.g., "Company A" vs. "CEO of Company A").
"""
        elif issue["issue_type"] == "redundancy_relationship":
            guideline = """**Redundant Relationships**(redundancy_relationship):

- Definition: Two or more distinct relationship entries connect the same pair of source and target entities (or entities identified as redundant duplicates) with the same semantic meaning.
- Identification: Look for identical or near-identical source/target entity pairs and relationship types/descriptions that convey the exact same connection. Minor variations in phrasing that don't change the core meaning should still be considered redundant.
- Example:
    - Redundant: User → Purchased → Product and Customer → Ordered → Product.
    - Non-redundant: User → Purchased in 2023 → Product and Customer → Purchased 2024 → Product.
- Note: Overlap in descriptive text between an entity and a relationship connected to it is generally acceptable for context and should not, by itself, trigger redundancy.
"""
        elif issue["issue_type"] == "entity_quality_issue":
            guideline = """**Entity Quality Issues**(entity_quality_issue):

- Definition: Fundamental flaws within a single entity's definition, description, or attributes that significantly hinder its clarity, accuracy, or usability. This is about core problems, not merely lacking detail.
- Subtypes:
    - Inconsistent Claims: Contains attributes or information that directly contradict each other (e.g., having mutually exclusive status flags like Status: Active and Status: Deleted). This points to a factual impossibility within the entity's representation.
    - Meaningless or Fundamentally Vague Description: The description is so generic, placeholder-like, or nonsensical that it provides no usable information to define or distinguish the entity (e.g., "An item", "Data entry", "See notes", "Used for system processes" without any specifics). The description fails its basic purpose.
    - Ambiguous Definition/Description: The provided name, description, or key attributes are described in a way that could plausibly refer to multiple distinct real-world concepts or entities, lacking the necessary specificity for unambiguous identification within the graph's context (e.g., An entity named "System" with description "Manages data processing" in a graph with multiple such systems).
"""
        elif issue["issue_type"] == "relationship_quality_issue":
            guideline = """**Relationship Quality Issues**(relationship_quality_issue):

- Definition: Fundamental flaws within a single relationship's definition or description that obscure its purpose, meaning, or the nature of the connection between the source and target entities. This is about core problems, not merely lacking detail.
- Subtypes:
    - Contradictory Definitions: Conflicting attributes or logic.
    - Fundamentally Unclear or Ambiguous Meaning: The relationship type or description is so vague, generic, or poorly defined that the nature of the connection between the source and target cannot be reliably understood. It fails to convey a specific semantic meaning. (e.g., `System A -- affects --> System B` without any context of how). This covers cases where the essential meaning is missing, making the relationship definition practically useless or open to multiple interpretations.
    - **Explicit Exclusions (Important!)**:
        * **Do NOT flag as a quality issue** solely because a description could be more detailed or comprehensive. The focus must remain on whether the *existing* definition is fundamentally flawed (contradictory, ambiguous, unclear).
"""

        issue_critic_prompt = f"""You are a knowledge graph quality expert. Your task is to determine if a reported issue actually exists in the given graph.

# Quality Standards

A high-quality knowledge graph should be:
- **Non-redundant**: Contains unique entities and relationships, avoiding duplication of the same real-world concept or connection.
- **Coherent**: Entities and relationships form a logical, consistent, and understandable structure representing the domain.
- **Precise**: Entities and relationships have clear, unambiguous definitions and descriptions, accurately representing specific concepts and connections.
- **Factually accurate**: All represented knowledge correctly reflects the real world or the intended domain scope.
- **Efficiently connected**: Features optimal pathways between related entities, avoiding unnecessary or misleading connections while ensuring essential links exist.


## Issue Identification Guidelines

{guideline}

# Your Task

## Graph Data:
{json.dumps(row['graph'], indent=2, ensure_ascii=False)}

## Reported Issue:
- **Type**: {issue['issue_type']}
- **{critic_object}**
- **Reasoning**: {issue['reasoning']}

## Evaluation Rules:

**For {issue['issue_type']} issues:**
- **is_valid: true** = The specified entities/relationships DO have the {issue['issue_type'].replace('_', ' ')} problem
- **is_valid: false** = The specified entities/relationships do NOT have the {issue['issue_type'].replace('_', ' ')} problem

**Important**: The reasoning provided may explain why something is NOT a problem. If the reasoning correctly explains that no problem exists, then is_valid should be FALSE.

**Example**: If reasoning says "entities are not redundant because they serve different purposes" and you agree, then is_valid = false (no redundancy problem exists).

Base your judgment solely on the graph data and the issue type definition above. Response format (surrounding by ```json and ```):
```json
{{
"is_valid": true/false,
"critique": "Your analysis explaining whether the claimed problem actually exists in the graph, with specific references to graph elements."
}}
```"""
        logger.info(
            f"processing issue {index} for {critic_name}, issue_type={issue['issue_type']}, critic_object={critic_object}, confidence={row['confidence']}"
        )
        try:
            response = critic_client.generate(issue_critic_prompt)
        except Exception as e:
            logger.error(
                "Failed to generate critique (index=%s, critic_name=%s), error: %s",
                index,
                critic_name,
                e,
            )
            continue

        try:
            critique_res = robust_json_parse(response, "object")
            if critique_res["is_valid"] is True:
                issue_df.at[index, "confidence"] = (
                    issue_df.at[index, "confidence"] + 0.9
                )
            issue_df.at[index, critic_name] = response
            continue
        except:
            logger.error(
                "Failed to parse critique (index=%s, critic_name=%s), response: %s",
                index,
                critic_name,
                response,
            )

    return issue_df
