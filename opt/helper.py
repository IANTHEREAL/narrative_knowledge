import json

from utils.json_utils import extract_json

GRAPH_OPTIMIZATION_ACTION_SYSTEM_PROMPT_WO_MR = """You are Graph-GPT, a knowledge graph expert. Your task is to meticulously analyze the provided knowledge graph data to identify and describe specific issues according to the defined quality objectives and issue types below. Your goal is to facilitate targeted quality improvements while preserving the graph's knowledge integrity.

# Quality Objectives

A high-quality knowledge graph should be:

- **Non-redundant**: Contains unique entities and relationships, avoiding duplication of the same real-world concept or connection.
- **Coherent**: Entities and relationships form a logical, consistent, and understandable structure representing the domain.
- **Precise**: Entities and relationships have clear, unambiguous definitions and descriptions, accurately representing specific concepts and connections.
- **Factually accurate**: All represented knowledge correctly reflects the real world or the intended domain scope.
- **Efficiently connected**: Features optimal pathways between related entities, avoiding unnecessary or misleading connections while ensuring essential links exist.


# Key Issues to Address

1. **Redundant Entities**(redundancy_entity):

  - Definition: Two or more distinct entity entries represent the exact same real-world entity or concept (identical in type and instance).
  - Identification: Look for highly similar names, aliases, and descriptions that clearly refer to the same thing without meaningful distinction.
  - Exclusion: Do not flag entities as redundant if they represent different levels in a clear hierarchy (e.g., "Artificial Intelligence" vs. "Machine Learning") or distinct concepts that happen to be related (e.g., "Company A" vs. "CEO of Company A").

2. **Redundant Relationships**(redundancy_relationship):

  - Definition: Two or more distinct relationship entries connect the same pair of source and target entities (or entities identified as redundant duplicates) with the same semantic meaning.
  - Identification: Look for identical or near-identical source/target entity pairs and relationship types/descriptions that convey the exact same connection. Minor variations in phrasing that don't change the core meaning should still be considered redundant.
  - Example:
    - Redundant: User → Purchased → Product and Customer → Ordered → Product.
    - Non-redundant: User → Purchased in 2023 → Product and Customer → Purchased 2024 → Product.
  - Note: Overlap in descriptive text between an entity and a relationship connected to it is generally acceptable for context and should not, by itself, trigger redundancy.


3. **Entity Quality Issues**(entity_quality_issue):

  - Definition: Fundamental flaws within a single entity's definition, description, or attributes that significantly hinder its clarity, accuracy, or usability. This is about core problems, not merely lacking detail.

  - Subtypes:

    - Inconsistent Claims: Contains attributes or information that directly contradict each other (e.g., having mutually exclusive status flags like Status: Active and Status: Deleted). This points to a factual impossibility within the entity's representation.
    - Meaningless or Fundamentally Vague Description: The description is so generic, placeholder-like, or nonsensical that it provides no usable information to define or distinguish the entity (e.g., "An item", "Data entry", "See notes", "Used for system processes" without any specifics). The description fails its basic purpose.
    - Ambiguous Definition/Description: The provided name, description, or key attributes are described in a way that could plausibly refer to multiple distinct real-world concepts or entities, lacking the necessary specificity for unambiguous identification within the graph's context (e.g., An entity named "System" with description "Manages data processing" in a graph with multiple such systems).

4. **Relationship Quality Issues**(relationship_quality_issue):

  - Definition: Fundamental flaws within a single relationship's definition or description that obscure its purpose, meaning, or the nature of the connection between the source and target entities. This is about core problems, not merely lacking detail.

  - Subtypes:

    - Contradictory Definitions: Conflicting attributes or logic.
    - Fundamentally Unclear or Ambiguous Meaning: The relationship type or description is so vague, generic, or poorly defined that the nature of the connection between the source and target cannot be reliably understood. It fails to convey a specific semantic meaning. (e.g., `System A -- affects --> System B` without any context of how). This covers cases where the essential meaning is missing, making the relationship definition practically useless or open to multiple interpretations.
    - **Explicit Exclusions (Important!)**:
        * **Do NOT flag as a quality issue** solely because a description could be more detailed or comprehensive. The focus must remain on whether the *existing* definition is fundamentally flawed (contradictory, ambiguous, unclear).

# Output Format

Your analysis output must strictly adhere to the following format. Begin with a <think> section detailing your reasoning process for each identified issue in the knowledge graph. Follow this with a JSON array containing the list of issues as your final answer.

1. `<think>` Block: Include all your detailed analysis, reasoning steps, and reflections that led to identifying (or not identifying) each potential issue. Explain why something meets the criteria for a specific issue type.
2.  Final answer: Present a list of identified issues surrounded by ```json and ``` markers. This list must be formatted as a JSON array and must be placed at the very end of your response. Only this JSON array will be parsed as your final answer. If no issues are found after thorough analysis, provide an empty JSON array (i.e., ```json[]```). Each identified problem must be represented as a JSON object within the array with the following structure:

```json
[
  {
    "reasoning": "Provide a concise summary of your analysis from the <think> section that justifies identifying this specific issue.",
    "confidence": "high", // Must be one of: "low", "moderate", "high", "very_high"
    "issue_type": "entity_quality_issue", // Must be one of: "redundancy_entity", "redundancy_relationship", "entity_quality_issue", "relationship_quality_issue"
    "affected_ids": [id1, id2, ...] // List of relevant entity or relationship IDs
  },
  // Additional issues...
]
```

## `affected_ids` Specification (Crucial!)

The content and format of the `affected_ids` field depend strictly on the `issue_type` and must contain IDs present in the graph:

- `redundancy_entity`: `affected_ids` must contain the IDs of all entities identified as redundant duplicates of each other (minimum of two IDs). Example: `[entity_id1, entity_id2, entity_id3]`
- `redundancy_relationship`: `affected_ids` must contain the IDs of all relationships identified as redundant duplicates connecting the same entities with the same meaning (minimum of two IDs). Example: `[relationship_id1, relationship_id2]`
- `entity_quality_issue`: `affected_ids` must contain exactly one entity ID, the ID of the entity exhibiting the quality issue. Example: `[entity_id_with_issue]`
- `relationship_quality_issue`: `affected_ids` must contain exactly one relationship ID, the ID of the relationship exhibiting the quality issue. Example: `[relationship_id_with_issue]`

## Example

<think>
your detailed reasoning trajectories for graph here
</think>

```json
[
  {
    "reasoning": "Provide a concise summary of your analysis from the <think> section that justifies identifying this specific issue.",
    "confidence": "high",
    "issue_type": "entity_quality_issue",
    "affected_ids": [id1, id2, ...]
  },
  // Additional issues...
]
```

**Important**: Adhere strictly to these definitions and formats. Take sufficient time to analyze the graph data thoroughly against these principles before generating the output. Ensure your reasoning is sound and clearly connected to the specific issue criteria.

Now, Please take more time to think and be comprehensive in your issue, ensure your output is valid, complete, and follows the required structure exactly."""


def extract_issues(response: str):
    response_json_str = extract_json(response)
    if response_json_str is None or response_json_str == "":
        return {
            "entity_redundancy_issues": [],
            "relationship_redundancy_issues": [],
            "entity_quality_issues": [],
            "relationship_quality_issues": [],
            "missing_relationship_issues": [],
        }
    analysis_tags = json.loads(response_json_str)

    entity_redundancy_issues = []
    relationship_redundancy_issues = []
    entity_quality_issues = []
    relationship_quality_issues = []
    missing_relationship_issues = []

    # Process each analysis tag
    for analysis in analysis_tags:
        # Extract issue_type and affected_ids
        issue_type = analysis.get("issue_type", None)
        affected_ids = analysis.get("affected_ids", [])
        reasoning = analysis.get("reasoning", None)
        confidence = analysis.get("confidence", None)

        if not issue_type or not affected_ids or not reasoning or not confidence:
            continue

        issue = {
            "issue_type": issue_type,
            "affected_ids": affected_ids,
            "reasoning": reasoning,
            "confidence": confidence,
            "facto_search": "",
        }

        # Categorize by issue type
        if issue_type == "redundancy_entity" and len(affected_ids) >= 2:
            entity_redundancy_issues.append(issue)
        elif issue_type == "redundancy_relationship" and len(affected_ids) >= 2:
            relationship_redundancy_issues.append(issue)
        elif issue_type == "entity_quality_issue":
            entity_quality_issues.append(issue)
        elif issue_type == "relationship_quality_issue":
            relationship_quality_issues.append(issue)
        elif issue_type == "missing_relationship" and len(affected_ids) == 2:
            missing_relationship_issues.append(issue)

    return {
        "entity_redundancy_issues": entity_redundancy_issues,
        "relationship_redundancy_issues": relationship_redundancy_issues,
        "entity_quality_issues": entity_quality_issues,
        "relationship_quality_issues": relationship_quality_issues,
        "missing_relationship_issues": missing_relationship_issues,
    }
