# Unified Quality Standard for Knowledge Graph Entities and Relationships

## Introduction

This standard defines the ideal state for entities and relationships within a high-quality knowledge graph. It integrates the core requirements from both the graph construction (extraction) and optimization phases, providing **specific technical specifications** for critical components. Its purpose is to serve as a unified, actionable guide for creating precise, unambiguous, coherent, and reliable knowledge representations.

## Core Quality Principles

A high-quality knowledge graph must adhere to the following five core principles:

1.  **Non-redundant**: The graph must not contain duplicate entities representing the same real-world concept or duplicate relationships expressing the same semantics.
2.  **Precise**: Entities and relationships must have clear, unambiguous definitions that accurately represent specific concepts and connections.
3.  **Factually Accurate**: All information must reflect real-world facts or facts within a specific domain scope and must be verifiable.
4.  **Coherent**: Entities and relationships must form a logically consistent and understandable knowledge network, free from internal contradictions.
5.  **Efficiently Connected**: Pathways between entities should be optimal, avoiding unnecessary or misleading connections while ensuring all essential links exist.

---

## Entity Quality Standard

### 1. Entity Name
- **Specific and Unambiguous**: Must uniquely identify the entity within its domain context.
- **Descriptive**: The name should intuitively reflect the entity's core identity or function.
- **Canonical**: When multiple name variants exist, the most recognized or official name must be used.
- **Consistent**: Must follow established naming conventions, avoiding meaningless prefixes or suffixes.

### 2. Entity Description
- **Entity-Focused**: The core of the description must answer the question: **"What IS this entity?"**
- **Self-Contained**: The description must provide enough information for a user to understand the entity without needing to consult external context.
- **Precise Definition**: Clearly state the entity's category, purpose, and key characteristics.
- **Contextually Relevant**: May include a summary of the entity's core role in the knowledge system, but specific interaction details **must** be carried by relationships.
- **Appropriately Detailed**: Must be detailed enough to avoid ambiguity but concise enough for practical use.

### 3. Entity Attributes
- **Essential Only**: Include only those attributes that provide critical background, defining characteristics, or classification information.
- **Non-Contradictory**: Attribute values within the same entity must not conflict with one another.
- **Consistent Format**: The naming and value structure of attributes should follow a unified, standardized format.
- **Relevant Metadata**: Where applicable, should include metadata such as provenance, confidence scores, or update timestamps.
- **Entity Type (Highly Recommended)**: To enhance LLM reasoning, entities SHOULD include classification attributes.
    - **`entity_type`**: A mandatory classification from a predefined enumeration (e.g., `Person`, `Concept`, `Technology`).
    - **`domain`**: A standardized domain label (e.g., `database`, `finance`).
- **Search Enhancement (Highly Recommended)**: To improve retrieval accuracy, entities SHOULD include search-related attributes.
    - **`searchable_keywords`**: An array of domain-specific terms and synonyms.
    - **`aliases`**: An array of alternative names or abbreviations.
- **Contextual Information (Optional)**: For richer understanding, entities MAY include supplementary details.
    - **`usage_context`**: A natural language description of primary use cases.

---

## Relationship Quality Standard

The core structure of a relationship is `Source Entity -> Relationship Description -> Target Entity`. The entire semantic meaning and context of a relationship are carried within its **Description**.

### 1. Relationship Description - Core Specification
The relationship's description is its soul and must strictly adhere to the following requirements:

- **Natural Language Narrative**: Must be one or more grammatically complete and fluent **natural language sentences**, not simple labels or keywords. It should read like a factual statement.
- **Self-Contained Context**: The description must contain sufficient information (e.g., method, purpose, conditions, results) for a reader to fully understand the specific interaction between the source and target entities **without needing to look up additional information**.
- **Deep Interaction Details**: Must exhaustively explain **"How"** the entities interact, not merely state **"That"** a relationship exists.
    - **Bad Practice**: `Company A -- influences --> Company B`
    - **Good Practice**: `Company A -- significantly lowered the technical barrier and R&D costs for Company B's development of its image recognition product by releasing its open-source deep learning framework, "TensorFlow". --> Company B`
- **Clarity and Single Responsibility**: Each relationship description should focus on a single, specific interaction or event. If multiple distinct types of interaction exist between a source and target, multiple unique relationships should be created.

### 2. Relationship Attributes
The attributes of a relationship provide structured metadata and context for the narrative description.

- **Connection-Relevant**: Attributes must provide additional information specifically about the connection itself.
- **Relationship Type**: An atomic classification label for the relationship (e.g., `develops`, `acquires`, `uses`) **can be stored as an optional attribute** for fast filtering or classification, but it does not carry the core semantic meaning.
    - **Example**: `attributes: { "type": "develops" }`
- **Strength/Confidence Indicators**: May include quantitative or qualitative attributes to indicate the importance or certainty of the relationship.
- **Contextual Conditions (Highly recommended, if available.)**: To clarify the relationship's applicability for LLMs, attributes SHOULD describe its context.
    - **`condition`**: A natural language description of circumstances under which the relationship holds.
    - **`scope`**: A natural language description of the applicable range or context (e.g., versioning, environment).
- **Execution Context (Optional)**: For deeper insight, attributes MAY provide details about the relationship's manifestation.
    - **`prerequisite`**: A natural language description of required preconditions.
    - **`impact`**: A natural language description of the effects or consequences.

### 3. Specific Specification for Relationship Attributes: Temporal Information
To ensure the accuracy, consistency, and computability of temporal information, attributes related to time in a relationship must follow this **mandatory specification**:

- **`fact_time` (Single Point in Time)**
    - **Definition**: Use this attribute for events that occur at a **single, specific point in time**.
    - **Format**: The value must be an **ISO 8601 formatted UTC timestamp string**.
    - **Example**: `"fact_time": "2025-06-25T11:30:00Z"`

- **`fact_time_range` (Time Range)**
    - **Definition**: Use this attribute for events or states that persist over a **duration of time**.
    - **Format**: The value must be a **dictionary** containing `start` and `end` keys. The values for `start` and `end` can be an ISO 8601 timestamp string or `null` (if one end of the range is open or unknown).
    - **Example**:
        - A duration of one year: `{"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z"}`
        - Starting from a specific time: `{"start": "2025-01-01T00:00:00Z", "end": null}`

- **`temporal_context` (Original Time Context)**
    - **Definition**: Used to store the **raw, unprocessed original time expression** as extracted directly from the source text. This attribute is critical for traceability and understanding context.
    - **Format**: String.
    - **Example**: `"temporal_context": "in the fourth quarter of last year"` or `"temporal_context": "since the company was founded"`

---

## The Golden Rule: The Evidence Principle

- **Evidence-Based Mandate**: Every single claim in the knowledge graph—be it an entity, a relationship, or any of its attributes—must be backed by explicit, traceable evidence.
- **No Speculation**: It is strictly forbidden to infer or assume information that is not explicitly stated in the source data.
- **Verifiable**: All information should be verifiable. For instance, the `temporal_context` attribute serves as direct evidentiary support for the structured `fact_time` and `fact_time_range` attributes.