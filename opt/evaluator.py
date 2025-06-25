import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from utils.json_utils import robust_json_parse

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """Issue data structure"""

    issue_type: str
    affected_ids: List[str]
    reasoning: str
    source_graph: Dict[str, Any]
    analysis_context: str = "graph quality analysis"
    validation_score: float = 0.0
    critic_evaluations: Dict[str, str] = None
    is_resolved: bool = False

    def __post_init__(self):
        if self.critic_evaluations is None:
            self.critic_evaluations = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            "issue_type": self.issue_type,
            "affected_ids": self.affected_ids,
            "reasoning": self.reasoning,
            "source_graph": self.source_graph,
            "analysis_context": self.analysis_context,
            "validation_score": self.validation_score,
            "critic_evaluations": self.critic_evaluations,
            "is_resolved": self.is_resolved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Issue":
        """Create Issue from dictionary"""
        return cls(
            issue_type=data["issue_type"],
            affected_ids=data["affected_ids"],
            reasoning=data["reasoning"],
            source_graph=data["source_graph"],
            analysis_context=data.get("analysis_context", "graph quality analysis"),
            validation_score=data.get("validation_score", 0.0),
            critic_evaluations=data.get("critic_evaluations", {}),
            is_resolved=data.get("is_resolved", False),
        )


def batch_evaluate_issues(
    critic_clients: Dict[str, Any], issues: List[Issue]
) -> List[Issue]:
    """
    Evaluate a list of issues using critic clients.

    Args:
        critic_clients: Dictionary of critic name -> LLM client
        issues: List of Issue objects to evaluate

    Returns:
        List of evaluated Issue objects with updated critic_evaluations and validation_score
    """
    logger.info(f"Evaluating {len(issues)} issues with {len(critic_clients)} critics")

    for critic_name, critic_client in critic_clients.items():
        logger.info(f"Evaluating issues with {critic_name}")

        for issue in issues:
            # Skip if already evaluated by this critic
            if (
                critic_name in issue.critic_evaluations
                and issue.critic_evaluations[critic_name]
            ):
                try:
                    robust_json_parse(issue.critic_evaluations[critic_name], "object")
                    continue  # Skip if valid evaluation exists
                except:
                    logger.warning(
                        f"Invalid critique found for {critic_name}, re-evaluating"
                    )

            # Evaluate this issue
            try:
                evaluation = evaluate_single_issue(critic_name, critic_client, issue)
                if evaluation:
                    issue.critic_evaluations[critic_name] = evaluation

                    # Update validation score if critique is positive
                    try:
                        critique_res = robust_json_parse(evaluation, "object")
                        if critique_res.get("is_valid") is True:
                            issue.validation_score += 0.9
                    except:
                        logger.error(
                            f"Failed to parse critique for validation score update"
                        )

            except Exception as e:
                logger.error(f"Failed to evaluate issue with {critic_name}: {e}")

    return issues


def evaluate_single_issue(
    critic_name: str, critic_client: Any, issue: Issue
) -> Optional[str]:
    """
    Evaluate a single issue with a specific critic.

    Args:
        critic_name: Name of the critic
        critic_client: LLM client for the critic
        issue: Issue object to evaluate

    Returns:
        Evaluation result as JSON string or None if failed
    """

    # Determine critic object based on issue type
    if issue.issue_type in ("redundancy_entity", "entity_quality_issue"):
        critic_object = f"affected entities: {issue.affected_ids}"
    elif issue.issue_type in ("redundancy_relationship", "relationship_quality_issue"):
        critic_object = f"affected relationships: {issue.affected_ids}"
    else:
        critic_object = f"affected items: {issue.affected_ids}"

    # Get appropriate guideline based on issue type
    guideline = get_issue_guideline(issue.issue_type)

    # Build evaluation prompt
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
{json.dumps(issue.source_graph, indent=2, ensure_ascii=False)}

## Reported Issue:
- **Type**: {issue.issue_type}
- **{critic_object}**
- **Reasoning**: {issue.reasoning}

## Evaluation Rules:

**For {issue.issue_type} issues:**
- **is_valid: true** = The specified entities/relationships DO have the {issue.issue_type.replace('_', ' ')} problem
- **is_valid: false** = The specified entities/relationships do NOT have the {issue.issue_type.replace('_', ' ')} problem

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
        f"Evaluating {issue.issue_type} issue with {critic_name}, validation_score={issue.validation_score}"
    )

    try:
        response = critic_client.generate(issue_critic_prompt)
        return response
    except Exception as e:
        logger.error(f"Failed to generate critique with {critic_name}: {e}")
        return None


def get_issue_guideline(issue_type: str) -> str:
    """Get the evaluation guideline for a specific issue type"""

    guidelines = {
        "redundancy_entity": """**Redundant Entities**(redundancy_entity):

- Definition: Two or more distinct entity entries represent the exact same real-world entity or concept (identical in type and instance).
- Identification: Look for highly similar names, aliases, and descriptions that clearly refer to the same thing without meaningful distinction.
- Exclusion: Do not flag entities as redundant if they represent different levels in a clear hierarchy (e.g., "Artificial Intelligence" vs. "Machine Learning") or distinct concepts that happen to be related (e.g., "Company A" vs. "CEO of Company A").
""",
        "redundancy_relationship": """**Redundant Relationships**(redundancy_relationship):

- Definition: Two or more distinct relationship entries connect the same pair of source and target entities (or entities identified as redundant duplicates) with the same semantic meaning.
- Identification: Look for identical or near-identical source/target entity pairs and relationship types/descriptions that convey the exact same connection. Minor variations in phrasing that don't change the core meaning should still be considered redundant.
- Example:
    - Redundant: User → Purchased → Product and Customer → Ordered → Product.
    - Non-redundant: User → Purchased in 2023 → Product and Customer → Purchased 2024 → Product.
- Note: Overlap in descriptive text between an entity and a relationship connected to it is generally acceptable for context and should not, by itself, trigger redundancy.
""",
        "entity_quality_issue": """**Entity Quality Issues**(entity_quality_issue):

- Definition: Fundamental flaws within a single entity's definition, description, or attributes that significantly hinder its clarity, accuracy, or usability. This is about core problems, not merely lacking detail.
- Subtypes:
    - Inconsistent Claims: Contains attributes or information that directly contradict each other (e.g., having mutually exclusive status flags like Status: Active and Status: Deleted). This points to a factual impossibility within the entity's representation.
    - Meaningless or Fundamentally Vague Description: The description is so generic, placeholder-like, or nonsensical that it provides no usable information to define or distinguish the entity (e.g., "An item", "Data entry", "See notes", "Used for system processes" without any specifics). The description fails its basic purpose.
    - Ambiguous Definition/Description: The provided name, description, or key attributes are described in a way that could plausibly refer to multiple distinct real-world concepts or entities, lacking the necessary specificity for unambiguous identification within the graph's context (e.g., An entity named "System" with description "Manages data processing" in a graph with multiple such systems).
""",
        "relationship_quality_issue": """**Relationship Quality Issues**(relationship_quality_issue):

- Definition: Fundamental flaws within a single relationship's definition or description that obscure its purpose, meaning, or the nature of the connection between the source and target entities. This is about core problems, not merely lacking detail.
- Subtypes:
    - Contradictory Definitions: Conflicting attributes or logic.
    - Fundamentally Unclear or Ambiguous Meaning: The relationship type or description is so vague, generic, or poorly defined that the nature of the connection between the source and target cannot be reliably understood. It fails to convey a specific semantic meaning. (e.g., `System A -- affects --> System B` without any context of how). This covers cases where the essential meaning is missing, making the relationship definition practically useless or open to multiple interpretations.
    - **Explicit Exclusions (Important!)**:
        * **Do NOT flag as a quality issue** solely because a description could be more detailed or comprehensive. The focus must remain on whether the *existing* definition is fundamentally flawed (contradictory, ambiguous, unclear).
""",
    }

    return guidelines.get(
        issue_type, "No specific guideline available for this issue type."
    )
