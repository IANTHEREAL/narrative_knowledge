import json

from utils.json_utils import extract_json


def evaluate_issue(critic_clients, issue_df):
    for critic_name in critic_clients.keys():
        critic_client = critic_clients[critic_name]
        for index, row in issue_df.iterrows():
            if "issue" in row:
                issue = row["issue"]
            else:
                issue = row
            if row[critic_name] is not None:
                skip = True
                if row[critic_name] is not None:
                    critique_json_str = extract_json(row[critic_name])
                    if critique_json_str is not None:
                        try:
                            json.loads(critique_json_str)
                            # print(f"skip processed issue {index} for {critic_name}", issue['issue_type'], issue['affected_ids'], row['confidence'])
                        except:
                            print(
                                f"Failed to parse critique for {index} for {critic_name}",
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

            if issue["issue_type"] in "redundancy_entity":
                guideline = """**Redundant Entities**(redundancy_entity):

    - Definition: Two or more distinct entity entries represent the exact same real-world entity or concept (identical in type and instance).
    - Identification: Look for highly similar names, aliases, and descriptions that clearly refer to the same thing without meaningful distinction.
    - Exclusion: Do not flag entities as redundant if they represent different levels in a clear hierarchy (e.g., "Artificial Intelligence" vs. "Machine Learning") or distinct concepts that happen to be related (e.g., "Company A" vs. "CEO of Company A").
    """
            elif issue["issue_type"] in "redundancy_relationship":
                guideline = """**Redundant Relationships**(redundancy_relationship):

    - Definition: Two or more distinct relationship entries connect the same pair of source and target entities (or entities identified as redundant duplicates) with the same semantic meaning.
    - Identification: Look for identical or near-identical source/target entity pairs and relationship types/descriptions that convey the exact same connection. Minor variations in phrasing that don't change the core meaning should still be considered redundant.
    - Example:
        - Redundant: User → Purchased → Product and Customer → Ordered → Product.
        - Non-redundant: User → Purchased in 2023 → Product and Customer → Purchased 2024 → Product.
    - Note: Overlap in descriptive text between an entity and a relationship connected to it is generally acceptable for context and should not, by itself, trigger redundancy.
    """
            elif issue["issue_type"] in "entity_quality_issue":
                guideline = """**Entity Quality Issues**(entity_quality_issue):

    - Definition: Fundamental flaws within a single entity's definition, description, or attributes that significantly hinder its clarity, accuracy, or usability. This is about core problems, not merely lacking detail.
    - Subtypes:
        - Inconsistent Claims: Contains attributes or information that directly contradict each other (e.g., having mutually exclusive status flags like Status: Active and Status: Deleted). This points to a factual impossibility within the entity's representation.
        - Meaningless or Fundamentally Vague Description: The description is so generic, placeholder-like, or nonsensical that it provides no usable information to define or distinguish the entity (e.g., "An item", "Data entry", "See notes", "Used for system processes" without any specifics). The description fails its basic purpose.
        - Ambiguous Definition/Description: The provided name, description, or key attributes are described in a way that could plausibly refer to multiple distinct real-world concepts or entities, lacking the necessary specificity for unambiguous identification within the graph's context (e.g., An entity named "System" with description "Manages data processing" in a graph with multiple such systems).
    """
            elif issue["issue_type"] in "relationship_quality_issue":
                guideline = """**Relationship Quality Issues**(relationship_quality_issue):

    - Definition: Fundamental flaws within a single relationship's definition or description that obscure its purpose, meaning, or the nature of the connection between the source and target entities. This is about core problems, not merely lacking detail.
    - Subtypes:
        - Contradictory Definitions: Conflicting attributes or logic.
        - Fundamentally Unclear or Ambiguous Meaning: The relationship type or description is so vague, generic, or poorly defined that the nature of the connection between the source and target cannot be reliably understood. It fails to convey a specific semantic meaning. (e.g., `System A -- affects --> System B` without any context of how). This covers cases where the essential meaning is missing, making the relationship definition practically useless or open to multiple interpretations.
        - **Explicit Exclusions (Important!)**:
            * **Do NOT flag as a quality issue** solely because a description could be more detailed or comprehensive. The focus must remain on whether the *existing* definition is fundamentally flawed (contradictory, ambiguous, unclear).
    """

            issue_critic_prompt = f"""You are a critic expert. You are given a graph and an issue. Please analyze the issue and provide a critique.

    # Quality Objectives

    A high-quality knowledge graph should be:

    - **Non-redundant**: Contains unique entities and relationships, avoiding duplication of the same real-world concept or connection.
    - **Coherent**: Entities and relationships form a logical, consistent, and understandable structure representing the domain.
    - **Precise**: Entities and relationships have clear, unambiguous definitions and descriptions, accurately representing specific concepts and connections.
    - **Factually accurate**: All represented knowledge correctly reflects the real world or the intended domain scope.
    - **Efficiently connected**: Features optimal pathways between related entities, avoiding unnecessary or misleading connections while ensuring essential links exist.


    ## Issue Identification Guidelines

    {guideline}

    # Issue to critique

    ## Graph:
    {json.dumps(row['graph'], indent=2, ensure_ascii=False)}

    ## Issue Description:

    issue type: {issue['issue_type']}
    {critic_object}
    reasoning: {issue['reasoning']}

    Please provide a critical analysis of this issue. Determine whether the issue is valid based on the graph data and the reasoning provided.

    Your response should be a JSON with the following format (if is_valid is true, the issue makes sense, otherwise the issue does not make sense):
    ```json
    {{
    "is_valid": true/false,
    "critique": "Your detailed critique explaining why the issue is valid or invalid. Include specific references to the graph data where applicable."
    }}
        """
            print(
                f"processing issue {index} for {critic_name}",
                issue["issue_type"],
                critic_object,
                row["confidence"],
            )
            try:
                response = critic_client.generate(issue_critic_prompt)
            except Exception as e:
                print(
                    f"Failed to generate critique for {row} in {critic_name}, error: {e}"
                )
                continue

            print(response)
            critique_json_res = extract_json(response)
            if critique_json_res is not None:
                try:
                    critique_json_res = "".join(
                        char
                        for char in critique_json_res
                        if ord(char) >= 32 or char in "\r\t"
                    )
                    critique_res = json.loads(critique_json_res)
                    if critique_res["is_valid"] is True:
                        issue_df.at[index, "confidence"] = (
                            issue_df.at[index, "confidence"] + 0.9
                        )
                    issue_df.at[index, critic_name] = response
                    continue
                except:
                    print(
                        f"Failed to parse critique for {row} in {critic_name}, response: {response}"
                    )

    return issue_df
