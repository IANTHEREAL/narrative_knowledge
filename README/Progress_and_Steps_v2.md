# Progress and Next Steps v2

## Completed Works

### Available Tools
We have three main tools working together in flexible pipelines:

1. **DocumentETLTool** - Processes individual documents to SourceData
2. **BlueprintGenerationTool** - Creates topic analysis blueprints  
3. **GraphBuildTool** - Builds knowledge graph from documents using blueprints

### Unified API Integration
The `/api/v1/save` endpoint now uses PipelineOrchestrator to dynamically select appropriate pipelines based on context.

Current `/api/v1/save` endpoint:
- Migration to tool-based pipeline execution
- Support for process_strategy parameter
- Integration with PipelineOrchestrator

## Example Processing Pipelines

### Scenario 1: Single Document to Existing Topic
**Flow:** DocumentETLTool → GraphBuildTool (using existing blueprint)

```mermaid
graph TD
    A[New Document] --> B(DocumentETLTool);
    B --> C[SourceData];
    D[Existing Topic Blueprint] --> E(GraphBuildTool);
    C --> E;
    E --> F((Global Knowledge Graph));
```

### Scenario 2: Batch Documents to Existing Topic
**Flow:** DocumentETLTool (parallel) → BlueprintGenerationTool → GraphBuildTool (parallel)

```mermaid
graph TD
    subgraph "Step 1: ETL"
        A[New Docs Batch] --> B(DocumentETLTool);
        B --> C[New SourceData];
    end
    subgraph "Step 2: Update Context"
        C --> D(BlueprintGenerationTool);
        E[Old SourceData] --> D;
        D --> F[Updated Topic Blueprint];
    end
    subgraph "Step 3: Build Graph"
        F --> G(GraphBuildTool);
        C --> G;
        G --> H((Global Knowledge Graph));
    end
```

### Scenario 3: New Topic with Batch Documents
**Flow:** DocumentETLTool (parallel) → BlueprintGenerationTool → GraphBuildTool (parallel)

```mermaid
graph TD
    subgraph "Step 1: ETL"
        A[New Docs Batch] --> B(DocumentETLTool);
        B --> C[SourceData];
    end
    subgraph "Step 2: Create Context"
        C --> D(BlueprintGenerationTool);
        D --> E[New Topic Blueprint];
    end
    subgraph "Step 3: Build Graph"
        E --> F(GraphBuildTool);
        C --> F;
        F --> G((Global Knowledge Graph));
    end
```

### Scenario 4: Memory Dialogues
**Flow:** DocumentETLTool → MemoryPipeline → Personal Memory Storage

```mermaid
graph TD
    A[Dialogue/Memory File] --> B(DocumentETLTool);
    B --> C[Memory SourceData];
    C --> D(MemoryPipeline);
    D --> E[Personal Memory Context];
    E --> F((Personal Memory Graph));
```
### Scenario 5: Memory Single Processing
**Flow:** DocumentETLTool → GraphBuildTool (single memory document)

```mermaid
graph TD
    A[Single Memory] --> B(DocumentETLTool);
    B --> C[SourceData];
    C --> D(GraphBuildTool);
    D --> E((Knowledge Graph));
```

### Current State

The system basically follows the pipeline design as follows:
- SmartSave API → PipelineAPIIntegration →
PipelineOrchestrator → Tools → Knowledge Graph

**Pipeline strategies available:**
- `knowledge_graph_single`: ["etl", "graph_build"]
- `knowledge_graph_batch_new`: ["etl", "blueprint_gen", "graph_build"]
- `knowledge_graph_batch_existing`: ["etl", "blueprint_gen", "graph_build"]
- `memory_direct_graph`: ["graph_build"]
- `memory_single`: ["graph_build"]

## Next Steps to Consider
### 0. Verification
- Verify the consistencies between different functions, as well as the correctness of functions calling.

### 1. Testing & Examples
- Create sample scripts for all pipeline scenarios
    (1) Single document to existing topic
    (2) Batch documents to existing topic
    (3) New topic with batch documents
    (4) Memory Dialogues
- Add unit tests for tool combinations
- Document edge cases (empty documents, duplicate processing, etc.)

### 2. Error Handling Enhancement
- Add retry mechanisms with exponential backoff
- Better error categorization (validation, network, parsing)
- Automatic retry on transient failures

### 3. Performance Monitoring
- Track processing time per tool
- Memory usage monitoring
- Batch processing efficiency metrics

### 4. Document Format Expansion
- Add support for: DOCX, XLSX, PPTX, HTML, XML, JSON, CSV, TSV
- Each format needs specialized extraction handlers
- Update DocumentETLTool format detection