# General Processing Flow

## Memory Processing Flow

### Current Memory Flow Architecture

```
POST /api/v1/save (JSON)
    ↓
_handle_json_data() in api/ingest.py
    ↓
_process_json_for_personal_memory()
    ↓
PipelineAPIIntegration.process_request()
    ↓
PipelineOrchestrator.select_default_pipeline() → "memory_direct_graph"
    ↓
PipelineOrchestrator.execute_pipeline() → ["memory_graph_build"]
    ↓
MemoryGraphBuildTool.execute()
    ↓
PersonalMemorySystem.process_chat_batch() [via MemoryGraphBuildTool]
    ↓
KnowledgeGraphBuilder.build_knowledge_graph()
    ↓
Graph triplets in database
```

### Detailed Flow Functions

#### 1. API Entry Point:
- `api/ingest.py:_process_json_for_personal_memory()` – Lines 192-229
- Uses `PipelineAPIIntegration` instead of standalone `PersonalMemorySystem`

#### 2. Pipeline Selection:
- `tools/orchestrator.py:select_default_pipeline()` – Returns `"memory_direct_graph"` for memory processing

#### 3. Memory Processing Tool:
- `tools/memory_graph_build_tool.py:MemoryGraphBuildTool.execute()` – Lines 101-150
- **Key**: This tool wraps the `PersonalMemorySystem` within the pipeline architecture

#### 4. Memory System Integration:
```python
# In MemoryGraphBuildTool.execute():
memory_system = PersonalMemorySystem(...)  # Same as standalone
result = memory_system.process_chat_batch(
    chat_messages=input_data["chat_messages"],
    user_id=input_data["user_id"]
)
```

### How Memory System is Used Within Pipeline

`MemoryGraphBuildTool` acts as an adapter that:
1. Receives pipeline-compatible input (chat messages + user_id)  
2. Delegates to `PersonalMemorySystem` for actual memory processing  
3. Returns pipeline-compatible output (with graph triplets)  

### API Usage Examples

**Memory Processing Request:**
```
POST /api/v1/save
Content-Type: application/json
```

```json
{
  "target_type": "personal_memory",
  "metadata": {"user_id": "user123"},
  "input": [
    {
      "message_content": "I love Python programming",
      "role": "user",
      "date": "2024-01-01T10:00:00Z"
    }
  ]
}
```

**Flow Path:**
1. API: `_process_json_for_personal_memory()` → `PipelineAPIIntegration`  
2. Orchestrator: `select_default_pipeline()` → `"memory_direct_graph"`  
3. Tool: `MemoryGraphBuildTool.execute()` → Uses `PersonalMemorySystem`  
4. Result: Graph triplets stored under topic `The personal information of user123`

---

## Document Processing Flow

### Document Processing Architecture

```
POST /api/v1/save (multipart/form-data or JSON)
    ↓
_handle_form_data() or _handle_json_data() in api/ingest.py
    ↓
_process_file_with_pipeline() or _process_json_for_knowledge_graph()
    ↓
PipelineAPIIntegration.process_request()
    ↓
PipelineOrchestrator.select_default_pipeline() → [pipeline selection]
    ↓
PipelineOrchestrator.execute_pipeline() → [tool sequence]
    ↓
DocumentETLTool → BlueprintGenerationTool (if needed) → GraphBuildTool
    ↓
Graph triplets in database
```

### Document Processing Scenarios Flow

#### Scenario 1: Single Document, Existing Topic (3.1.1)

- **Input:** `POST /api/v1/save` with single file  
- **Parameters:** `target_type="knowledge_graph"`, `topic_name="python_tutorial"`, `is_new_topic=false`, `file_count=1`

**Flow:**
1. `_process_file_with_pipeline()`  
2. `select_default_pipeline()` → `"single_doc_existing_topic"`  
3. `execute_pipeline(["etl", "graph_build"])`  
4. `DocumentETLTool.extract_knowledge()`  
5. `GraphBuildTool.build_knowledge_graph()`  
6. **Result:** Python tutorial knowledge graph under `"python_tutorial"` topic

---

#### Scenario 2: Batch Documents, Existing Topic (3.1.2)

- **Input:** `POST /api/v1/save` with multiple files  
- **Parameters:** `target_type="knowledge_graph"`, `topic_name="machine_learning"`, `is_new_topic=false`, `file_count>1`

**Flow:**
1. `_process_file_with_pipeline()` (for each file)  
2. `select_default_pipeline()` → `"batch_doc_existing_topic"`  
3. `execute_pipeline(["etl", "blueprint_gen", "graph_build"])`  
4. `DocumentETLTool.extract_knowledge()` (for each file)  
5. `BlueprintGenerationTool.generate_blueprint()`  
6. `GraphBuildTool.build_knowledge_graph()`  
7. **Result:** ML knowledge graph under `"machine_learning"` topic

---

#### Scenario 3: New Topic, Batch Documents (3.1.3)

- **Input:** `POST /api/v1/save` with multiple files  
- **Parameters:** `target_type="knowledge_graph"`, `topic_name="new_ai_research"`, `is_new_topic=true`, `file_count>1`

**Flow:**
1. `_process_file_with_pipeline()` (for each file)  
2. `select_default_pipeline()` → `"new_topic_batch"`  
3. `execute_pipeline(["etl", "blueprint_gen", "graph_build"])`  
4. `DocumentETLTool.extract_knowledge()` (for each file)  
5. `BlueprintGenerationTool.generate_blueprint()`  
6. `GraphBuildTool.build_knowledge_graph()`  
7. **Result:** New AI research knowledge graph under `"new_ai_research"` topic

---

### Detailed Flow Functions

#### 1. API Entry Points:
- `api/ingest.py:_process_file_with_pipeline()` – Lines 37-116 (file uploads)  
- `api/ingest.py:_process_json_for_knowledge_graph()` – Lines 232-303 (JSON input)

#### 2. Pipeline Selection Logic:
```python
# tools/orchestrator.py:select_default_pipeline()
if target_type == "knowledge_graph":
    if input_type == "document":
        if is_new_topic:
            return "new_topic_batch"
        elif file_count == 1:
            return "single_doc_existing_topic"
        else:
            return "batch_doc_existing_topic"
```

#### 3. Tool Execution Sequence:
- `single_doc_existing_topic`: `["etl", "graph_build"]`  
- `batch_doc_existing_topic`: `["etl", "blueprint_gen", "graph_build"]`  
- `new_topic_batch`: `["etl", "blueprint_gen", "graph_build"]`

---

### API Usage Examples

#### Single Document Upload

```
POST /api/v1/save
Content-Type: multipart/form-data
```

```
file: @python_guide.pdf
metadata: {"topic_name": "python_tutorial", "link": "docs/python_guide.pdf"}
target_type: "knowledge_graph"
```

#### Batch Documents Upload

```
POST /api/v1/save
Content-Type: multipart/form-data
```

```
files[]: @ml_intro.pdf
files[]: @ml_advanced.pdf
metadata: {"topic_name": "machine_learning", "link": "docs/ml_materials/"}
target_type: "knowledge_graph"
```

#### JSON Document Processing

```
POST /api/v1/save
Content-Type: application/json
```

```json
{
  "target_type": "knowledge_graph",
  "metadata": {"topic_name": "new_ai_research", "link": "inline_text"},
  "input": "Artificial intelligence is transforming industries..."
}
```

---

### Flow Path by Scenario

#### Single Document, Existing Topic:
1. API: `_process_file_with_pipeline()` → `PipelineAPIIntegration`  
2. Orchestrator: `select_default_pipeline()` → `"single_doc_existing_topic"`  
3. Tools: `DocumentETLTool` → `GraphBuildTool`  
4. Result: Knowledge graph under specified topic

#### Batch Documents, Existing Topic:
1. API: `_process_file_with_pipeline()` → `PipelineAPIIntegration`  
2. Orchestrator: `select_default_pipeline()` → `"batch_doc_existing_topic"`  
3. Tools: `DocumentETLTool` → `BlueprintGenerationTool` → `GraphBuildTool`  
4. Result: Knowledge graph under specified topic

#### New Topic, Batch Documents:
1. API: `_process_file_with_pipeline()` → `PipelineAPIIntegration`  
2. Orchestrator: `select_default_pipeline()` → `"new_topic_batch"`  
3. Tools: `DocumentETLTool` → `BlueprintGenerationTool` → `GraphBuildTool`  
4. Result: New knowledge graph under specified topic
