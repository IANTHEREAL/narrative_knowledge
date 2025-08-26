# Pipeline Parameters Documentation

## Overview

This document provides comprehensive information about input/output parameters for each component in the knowledge graph pipeline, including how parameters flow between components and support for both single and batch file processing, plus the memory pipeline.

## Architecture Overview

The system uses a pipeline orchestrator (`PipelineOrchestrator`) that chains tools together based on the processing scenario. The main entry point is `process_request()` in `api_integration.py`.

### Pipeline Types

1. **Knowledge Graph Pipelines**:
   - `single_doc_existing_topic`: `["etl", "graph_build"]`
   - `batch_doc_existing_topic`: `["etl", "blueprint_gen", "graph_build"]`
   - `new_topic_batch`: `["etl", "blueprint_gen", "graph_build"]`
   - `text_to_graph`: `["graph_build"]`

2. **Memory Pipelines**:
   - `memory_direct_graph`: `["memory_graph_build"]`
   - `memory_single`: `["memory_graph_build"]`

## Component Parameter Flow

### Entry Point: process_request()

**Location**: `tools/api_integration.py:31`

#### Document Processing Flow

**Input Parameters**:
```python
{
    "request_data": {
        "target_type": "knowledge_graph",
        "metadata": {
            "topic_name": str,          # required
            "is_new_topic": bool,       # use if provided, else check from database
            "link": str,                # optional: for single file
            "database_uri": str         # optional
        },
        "process_strategy": {
            "pipeline": ["etl", "blueprint_gen", "graph_build"]  # optional
        },
        "llm_client": object,
        "embedding_func": object,
        "force_regenerate": bool
    },
    "files": [                          # List[Dict]
        {
            "path": str,                # required for each file
            "filename": str,            # required: original filename
            "metadata": dict,           # optional: file-specific metadata
            "link": str                 # required: file URL for single/batch files
        }
    ]
}
```

#### Memory Processing Flow

**Input Parameters**:
```python
{
    "request_data": {
        "target_type": "personal_memory",
        "user_id": str,                 # required: user identifier
        "chat_messages": [              # required: conversation data
            {
                "message_content": str,     # required: message text
                "role": str,                # required: "user" | "assistant"
                "session_id": str,          # optional: conversation session
                "conversation_title": str,  # optional: title
                "date": str                 # optional: ISO timestamp
            }
        ],
        "metadata": {
            "user_id": str,             # alternative location for user_id
            "database_uri": str         # optional
        },
        "process_strategy": {},         # memory uses fixed pipeline
        "llm_client": object,
        "embedding_func": object,
        "force_regenerate": bool
    },
    "files": []                        # empty for memory processing
}
```

**Output**:
```python
{
    "success": bool,
    "data": {
        "results": {
            "etl": {...},
            "blueprint_gen": {...},
            "graph_build": {...}
        },
        "pipeline": [...],
        "duration_seconds": float
    },
    "error_message": str,
    "execution_id": str
}
```

## 1. DocumentETLTool (ETL)

**Purpose**: Processes raw document files into structured SourceData

### Single File Processing

**Input Parameters**:
```python
{
    "file_path": str,           # Required: Path to document file
    "topic_name": str,          # Required: Topic name for grouping
    "metadata": dict,           # Custom metadata
    "force_regenerate": bool,   # Force reprocessing
    "link": str,                # Document URL
    "original_filename": str    # Original filename
}
```

**Output**:
```python
{
    "source_data_ids": [str],   # List of created SourceData IDs
    "results": [
        {
            "source_data_id": str,
            "content_hash": str,
            "content_size": int,
            "source_type": str,
            "reused_existing": bool,
            "status": str,
            "file_path": str
        }
    ],
    "batch_summary": {
        "total_files": int,
        "processed_files": int,
        "reused_files": int,
        "failed_files": int
    }
}
```

### Batch File Processing

**Input Parameters**:
```python
{
    "files": [                  # Required: List of files
        {
            "path": str,        # Required: File path
            "metadata": dict,   #File-specific metadata
            "link": str,        # Document URL
            "filename": str     # Original filename
        }
    ],
    "topic_name": str,          # Required: Topic name
    "request_metadata": dict,   # Global metadata for all files
    "force_regenerate": bool    # Force reprocessing
}
```

**Output**: Same structure as single file, with batch statistics

## 2. BlueprintGenerationTool

**Purpose**: Creates analysis blueprints by analyzing all documents in a topic

### Input Parameters

```python
{
    "topic_name": str,          # Required: Topic name to generate blueprint
    "source_data_ids": [str],   # Optional: Specific source data IDs to include
    "force_regenerate": bool,   # Optional: Force regeneration
    "llm_client": object,       # Required: LLM client instance
    "embedding_func": object    # Optional: Embedding function
}
```

### Output

```python
{
    "blueprint_id": str,                # ID of created/updated AnalysisBlueprint
    "source_data_version_hash": str,    # Hash of contributing source data versions
    "contributing_source_data_count": int,  # Number of source data records used
    "blueprint_summary": {
        "canonical_entities_count": int,
        "key_patterns_count": int,
        "global_timeline_events": int,
        "processing_instructions_length": int,
        "cognitive_maps_used": int
    },
    "reused_existing": bool             # Whether existing blueprint was reused
}
```

## 3. GraphBuildTool

**Purpose**: Extracts knowledge from documents and adds to the global knowledge graph

### Single Document Processing

**Input Parameters**:
```python
{
    "source_data_id": str,      # Required: ID of SourceData to process
    "blueprint_id": str,        # Required: ID of AnalysisBlueprint to use
    "force_regenerate": bool,   # Optional: Force regeneration
    "llm_client": object,       # Required: LLM client instance
    "embedding_func": object    # Optional: Embedding function
}
```

**Output**:
```python
{
    "source_data_id": str,      # ID of processed SourceData
    "blueprint_id": str,        # ID of used AnalysisBlueprint
    "entities_created": int,    # Number of entities created
    "relationships_created": int, # Number of relationships created
    "triplets_extracted": int,  # Number of triplets extracted
    "reused_existing": bool,    # Whether existing processing was reused
    "status": str               # Processing status
}
```

### Batch Document Processing

**Input Parameters**:
```python
{
    "source_data_ids": [str],   # Required: Array of SourceData IDs
    "blueprint_id": str,        # Required: ID of AnalysisBlueprint to use
    "force_regenerate": bool,   # Optional: Force regeneration
    "llm_client": object,       # Required: LLM client instance
    "embedding_func": object    # Optional: Embedding function
}
```

### Topic-Based Batch Processing

**Input Parameters**:
```python
{
    "topic_name": str,          # Required: Name of topic to process
    "source_data_ids": [str],   # Optional: Specific source data IDs
    "force_regenerate": bool,   # Optional: Force regeneration
    "llm_client": object,       # Required: LLM client instance
    "embedding_func": object    # Optional: Embedding function
}
```

**Output** (for all batch modes):
```python
{
    "topic_name": str,          # Topic name processed
    "blueprint_id": str,        # ID of used AnalysisBlueprint
    "processed_count": int,     # Number of successfully processed documents
    "failed_count": int,        # Number of failed documents
    "total_entities_created": int,
    "total_relationships_created": int,
    "total_triplets_extracted": int,
    "results": [                # Individual results for each document
        {
            "source_data_id": str,
            "status": str,
            "entities_created": int,
            "relationships_created": int,
            "triplets_extracted": int,
            "error": str  # Only if status == "failed"
        }
    ]
}
```

## 4. MemoryGraphBuildTool (Memory Pipeline)

**Purpose**: Processes chat messages and builds personal knowledge graph

### Input Parameters

```python
{
    "chat_messages": [          # Required: List of chat messages
        {
            "message_content": str,     # Required: The message content
            "session_id": str,          # Optional: Conversation session ID
            "conversation_title": str,  # Optional: Title of the conversation
            "date": str,                # Optional: ISO format timestamp
            "role": str                 # Required: "user" | "assistant"
        }
    ],
    "user_id": str,             # Required: User identifier
    "force_regenerate": bool,   # Optional: Force reprocessing
    "llm_client": object,       # Required: LLM client instance
    "embedding_func": object    # Optional: Embedding function
}
```

### Output

```python
{
    "user_id": str,             # User identifier
    "topic_name": str,          # Generated topic name for this user
    "source_data_id": str,      # ID of created SourceData
    "knowledge_block_id": str,  # ID of created KnowledgeBlock
    "entities_created": int,    # Number of entities created
    "relationships_created": int, # Number of relationships created
    "triplets_extracted": int,  # Number of triplets extracted
    "status": str               # Processing status
}
```

## Parameter Flow Between Components

### Knowledge Graph Pipeline Flow

```
process_request()
    └── DocumentETLTool
        ├── Input: files + topic_name + metadata
        └── Output: source_data_ids[]
            └── BlueprintGenerationTool (OR use_existing)
                ├── Input: topic_name + source_data_ids
                └── Output: blueprint_id
                    └── GraphBuildTool
                        ├── Input: blueprint_id + source_data_ids
                        └── Output: graph building results
```

### Memory Pipeline Flow

```
process_request()
    └── MemoryGraphBuildTool
        ├── Input: chat_messages + user_id + metadata
        └── Output: memory processing results
```

## Processing Scenarios

### 1. Single Document to Existing Topic
- **Pipeline**: `["etl", "graph_build"]`
- **ETL Output**: `source_data_ids[0]` (single ID)
- **GraphBuild Input**: Uses existing blueprint from topic

### 2. Batch Documents to Existing Topic
- **Pipeline**: `["etl", "blueprint_gen", "graph_build"]`
- **ETL Output**: `source_data_ids[]` (multiple IDs)
- **BlueprintGen Input**: `topic_name` (accumulates all documents)
- **GraphBuild Input**: Uses newly created blueprint

### 3. New Topic with Batch Documents
- **Pipeline**: `["etl", "blueprint_gen", "graph_build"]`
- Same as #2, but creates new topic

### 4. Direct Text to Graph
- **Pipeline**: `["graph_build"]`
- **GraphBuild Input**: `topic_name` (processes all pending documents)

### 5. Personal Memory Processing
- **Pipeline**: `["memory_graph_build"]`
- **MemoryGraphBuild Input**: `chat_messages + user_id`
- **Output**: Personal knowledge graph for user

## Common Parameters Across All Tools

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `llm_client` | object | LLM client instance for processing | Yes |
| `embedding_func` | object | Embedding function for vector operations | Yes (except ETL) |
| `force_regenerate` | bool | Force regeneration even if already processed | No |
| `topic_name` | string | Topic name for grouping documents | Context-dependent |

## Error Handling

All components return standardized `ToolResult` objects:

```python
{
    "success": bool,
    "data": dict,           # Results on success
    "error_message": str,   # Error details on failure
    "execution_id": str,    # Unique execution identifier
    "duration_seconds": float
}
```