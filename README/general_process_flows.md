# General Processing Flow

## Memory Processing Flow

### Updated Memory Flow Architecture (Tool-Based)

#### Overall Architecture
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
PersonalMemorySystem.process_chat_batch() [modified GraphBuild]
    ├─ _store_chat_batch_as_source() 
    ├─ _create_summary_knowledge_block() 
    ├─ create_personal_blueprint()  
    └─ _build_graph_from_memory()  # Direct tool-based graph building
        └─ GraphBuildTool._process_single_document()
    ↓
Graph triplets in database
```

#### Detailed Architecture
```
POST /api/v1/save (JSON)
    ↓
api/ingest.py:_handle_json_data() [lines 1-191]
    ↓
api/ingest.py:_process_json_for_personal_memory() [lines 192-229]
    ├─ Validates user_id presence: metadata.get("user_id")
    ├─ Validates input_data type: isinstance(input_data, list)
    └─ PipelineAPIIntegration.process_request() [tools/api_integration.py]
        └─ Prepares unified request: {"target_type": "personal_memory", "metadata": {"user_id": user_id}, "input": chat_messages}
            ↓
PipelineOrchestrator.select_default_pipeline() → "memory_direct_graph" [tools/orchestrator.py]
    └─ Decision logic: if target_type == "personal_memory" → "memory_direct_graph"
        ↓
PipelineOrchestrator.execute_pipeline() → ["memory_graph_build"] [tools/orchestrator.py]
    └─ Tool sequence: ["memory_graph_build"]
        ↓
MemoryGraphBuildTool.execute() [tools/memory_graph_build_tool.py:101-227]
    ├─ Input validation: chat_messages, user_id, force_reprocess
    ├─ Deduplication check: _check_existing_processing() [lines 229-277]
    │   └─ Content hash generation: hashlib.sha256(json.dumps(chat_messages, sort_keys=True))
    ├─ PersonalMemorySystem initialization: PersonalMemorySystem(llm_client, embedding_func, session_factory)
    ├─ Memory processing: memory_system.process_chat_batch() [memory_system.py:165-220]
    │   ├─ _store_chat_batch_as_source() [memory_system.py:222-327]
    │   │   ├─ ContentStore creation: ContentStore(content_hash, content_json)
    │   │   ├─ SourceData creation: SourceData(name, link, source_type, attributes)
    │   │   └─ Deduplication: Check existing SourceData by link/content_hash
    │   ├─ _create_summary_knowledge_block() [memory_system.py:329-457]
    │   │   ├─ LLM summary generation: llm_client.generate(summary_prompt, max_tokens=4096)
    │   │   ├─ KnowledgeBlock creation: KnowledgeBlock(name, knowledge_type, content, content_vec)
    │   │   └─ BlockSourceMapping creation: BlockSourceMapping(block_id, source_id)
    │   ├─ create_personal_blueprint() [memory_system.py:459-497]
    │   │   └─ AnalysisBlueprint creation: AnalysisBlueprint(topic_name, processing_instructions)
    │   └─ _build_graph_from_memory() [memory_system.py:499-580]
    │       ├─ GraphBuildTool initialization: GraphBuildTool(session_factory, llm_client, embedding_func)
    │       ├─ Blueprint retrieval: AnalysisBlueprint.query.filter_by(topic_name)
    │       └─ Graph processing: graph_tool._process_single_document(blueprint_id, source_data_id)
    └─ Result aggregation: Combines memory and graph results
        ↓
Graph triplets in database [final storage]
```

### Detailed Flow Functions

#### 1. API Entry Point:
- `api/ingest.py:_process_json_for_personal_memory()` – Lines 192-229
- Uses `PipelineAPIIntegration` instead of standalone `PersonalMemorySystem`

#### 2. Pipeline Selection:
- `tools/orchestrator.py:select_default_pipeline()` – Returns `"memory_direct_graph"` for memory processing

#### 3. Memory Processing Tool:
- `tools/memory_graph_build_tool.py:MemoryGraphBuildTool.execute()` – Lines 101-227
- **Key**: This tool wraps the `PersonalMemorySystem` within the pipeline architecture

#### 4. Memory System Integration:
```python
# In MemoryGraphBuildTool.execute():
memory_system = PersonalMemorySystem(...)  # Same as standalone
result = memory_system.process_chat_batch(
    chat_messages=input_data["chat_messages"],
    user_id=input_data["user_id"]
)
# Direct processing replaces GraphBuild background daemon
```

### How Memory System is Used Within Pipeline

`MemoryGraphBuildTool` acts as an adapter that:
1. Receives pipeline-compatible input (chat messages + user_id)  
2. Delegates to `PersonalMemorySystem` for actual memory processing 
3. Uses `GraphBuildTool` for immediate graph building
4. Returns pipeline-compatible output with graph triplets/results 

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
3. Tool: `MemoryGraphBuildTool.execute()` → Uses `PersonalMemorySystem` with direct graph building
4. Result: Graph triplets stored under topic `The personal information of user123`

---

## Document Processing Flow

### Document Processing Architecture

#### Overall Architecture
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

#### Detailed Architecture
```
POST /api/v1/save (multipart/form-data or JSON)
    ↓
api/ingest.py:_handle_form_data() [lines 308-336] OR _handle_json_data() [lines 1-191]
    ├─ Content-Type detection: multipart/form-data vs application/json
    ├─ Metadata extraction: json.loads(metadata_str)
    └─ Route dispatch: target_type == "knowledge_graph" → _process_file_with_pipeline()/_process_json_for_knowledge_graph()
        ↓
api/ingest.py:_process_file_with_pipeline() [lines 37-116] OR _process_json_for_knowledge_graph() [lines 232-303]
    ├─ File validation: UploadFile validation and temp storage
    ├─ Strategy preparation: {"is_new_topic": ..., "file_count": ...}
    └─ PipelineAPIIntegration.process_request() [tools/api_integration.py]
        └─ Unified request: {"target_type": "knowledge_graph", "metadata": metadata, "process_strategy": strategy}
            ↓
PipelineOrchestrator.select_default_pipeline() [tools/orchestrator.py:172-181]
    ├─ Decision tree based on input parameters:
    │   ├─ if is_new_topic == true → "new_topic_batch"
    │   ├─ elif file_count == 1 → "single_doc_existing_topic"
    │   └─ else → "batch_doc_existing_topic"
    └─ Returns appropriate pipeline key
        ↓
PipelineOrchestrator.execute_pipeline() [tools/orchestrator.py]
    └─ Tool sequence execution based on pipeline type:
        ├─ "single_doc_existing_topic": ["etl", "graph_build"]
        ├─ "batch_doc_existing_topic": ["etl", "blueprint_gen", "graph_build"]
        └─ "new_topic_batch": ["etl", "blueprint_gen", "graph_build"]
            ↓
DocumentETLTool.execute() [tools/document_etl_tool.py:143-320]
    ├─ Input validation: file_path existence, topic_name format
    ├─ File processing: extract_source_data(str(file_path)) [etl/extract.py]
    ├─ ContentStore management: ContentStore(content_hash, content, content_type)
    ├─ SourceData creation: SourceData(name, topic_name, raw_data_source_id, attributes)
    └─ Status tracking: raw_data_source.status updates (pending → etl_processing → etl_completed)
        ↓
BlueprintGenerationTool.execute() [tools/blueprint_generation_tool.py:145-326] (conditional)
    ├─ Source data aggregation: query SourceData by topic_name
    ├─ Version hash calculation: hashlib.sha256("|".join(source_data_versions))
    ├─ Cognitive map generation: cm_generator.batch_generate_cognitive_maps()
    ├─ Blueprint creation: graph_builder.generate_analysis_blueprint()
    └─ AnalysisBlueprint updates: status, processing_instructions, source_data_version_hash
        ↓
GraphBuildTool.execute() [tools/graph_build_tool.py:162-221]
    ├─ Mode selection: single document vs batch processing
    ├─ Document processing: _process_document_with_blueprint() [lines 444-533]
    │   ├─ Document conversion: _convert_source_data_to_document() [lines 535-552]
    │   ├─ Cognitive map retrieval: cm_generator.get_cognitive_maps_for_topic()
    │   ├─ Triplet extraction: graph_builder.extract_triplets_from_document()
    │   └─ Graph persistence: graph_builder.convert_triplets_to_graph()
    └─ Status updates: SourceData.status (graph_processing → graph_completed/failed)
        ↓
Graph triplets in database [final storage]
```

### Document Processing Scenarios Flow

#### Scenario 1: Single Document, Existing Topic (4.1.1)

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

#### Scenario 2: Batch Documents, Existing Topic (4.1.2)

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

#### Scenario 3: New Topic, Batch Documents (4.1.3)

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
