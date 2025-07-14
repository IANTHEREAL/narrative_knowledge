# Knowledge Graph Daemons CLI

A unified command-line interface for managing both knowledge extraction and graph building daemons.

## Quick Start

### Start Individual Daemons

```bash
# Start knowledge extraction daemon (processes documents → SourceData)
python graph_daemon_cli.py extraction

# Start knowledge graph daemon (SourceData → Graph mappings)  
python graph_daemon_cli.py graph

# Start both daemons concurrently
python graph_daemon_cli.py both
```

### Check Status

```bash
# Show status of both daemons
python graph_daemon_cli.py status

# Show status including failed tasks
python graph_daemon_cli.py status --show-failed
```

## Commands

### `extraction`
Starts the knowledge extraction daemon that:
- Monitors GraphBuild table for pending tasks
- Extracts knowledge from documents
- Creates SourceData entries
- Marks tasks as completed

```bash
python graph_daemon_cli.py extraction [options]

Options:
  --check-interval INT    Check interval in seconds (default: 60)
  --llm-provider STR      LLM provider (default: openai_like)
  --llm-model STR         LLM model (default: qwen3-32b)  
  --log-level LEVEL       Log level: DEBUG|INFO|WARNING|ERROR (default: INFO)
```

### `graph`
Starts the knowledge graph daemon that:
- Finds SourceData without graph mappings
- Only processes completed topics
- Builds knowledge graphs (entities + relationships)
- Creates SourceGraphMapping entries

```bash
python graph_daemon_cli.py graph [options]

Options:
  --check-interval INT    Check interval in seconds (default: 120)
  --worker-count INT      Number of workers (default: 5)
  --llm-provider STR      LLM provider (default: openai_like)
  --llm-model STR         LLM model (default: qwen3-32b)
  --log-level LEVEL       Log level: DEBUG|INFO|WARNING|ERROR (default: INFO)
```

### `both`
Starts both daemons concurrently in separate threads:

```bash
python graph_daemon_cli.py both [options]

Options:
  --extraction-interval INT   Extraction daemon interval (default: 60)
  --graph-interval INT        Graph daemon interval (default: 120)
  --worker-count INT          Number of workers for graph daemon (default: 5)
  --llm-provider STR          LLM provider (default: openai_like)
  --llm-model STR             LLM model (default: qwen3-32b)
  --log-level LEVEL           Log level: DEBUG|INFO|WARNING|ERROR (default: INFO)
```

### `status`
Shows current status of both daemons:

```bash
python graph_daemon_cli.py status [options]

Options:
  --show-failed          Always show failed tasks section
  --failed-limit INT     Max failed tasks to show (default: 5)
```

## Status Output

The status command shows:

**Knowledge Extraction Daemon:**
- Running status and check interval
- Pending, processing, completed, and failed task counts
- Details of pending tasks
- Recent failed tasks (if any)

**Knowledge Graph Daemon:**
- Running status and configuration
- Total unmapped sources across all completed topics
- Graph mapping statistics (entities/relationships)
- Breakdown by completed topics

## Workflow

1. **Upload documents** → GraphBuild entries with "pending" status
2. **Extraction daemon** processes pending tasks → SourceData + "completed" status
3. **Graph daemon** finds SourceData from completed topics → Entities + Relationships + SourceGraphMapping

## Environment Variables

Make sure these are set in your `.env` file:

```
OPENAI_LIKE_BASE_URL=http://your-llm-endpoint
EMBEDDING_BASE_URL=http://your-embedding-endpoint
# ... other database and API settings
```

## Logs

Logs are written to both console and `knowledge_graph_daemons.log` file.

## Examples

```bash
# Production setup - start both with custom intervals
python graph_daemon_cli.py both \
  --extraction-interval 30 \
  --graph-interval 300 \
  --worker-count 10 \
  --log-level INFO

# Development - start extraction daemon with debug logging
python graph_daemon_cli.py extraction \
  --check-interval 10 \
  --log-level DEBUG

# Check what's happening
python graph_daemon_cli.py status --show-failed
``` 