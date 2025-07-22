# SmartSave API v1

## SmartSave API Documentation

### Overview

This API provides a flexible and extensible interface for saving structured or unstructured data, including natural language input, documents, URLs, and file uploads. It supports intelligent downstream routing, chunking, metadata storage, and LLM/agent-based processing.

---

### Endpoint

```http
POST /api/save
```

---

### Supported Content-Types

- `application/json`
- `multipart/form-data`

---

### Request Parameters


| Field         | Type               | Required | Description |
|---------------|--------------------|----------|-------------|
| input         | string or object   | No       | Raw content such as natural language, markdown, HTML, or structured JSON. |
| file          | file or string     | No       | File upload or file path/URL. Supports PDF, DOCX, TXT, etc. |
| url           | string             | No       | Publicly accessible URL for the document. |
| metadata      | object             | No       | Additional metadata like title, author, source, timestamp, language. |
| tags          | array of strings   | No       | Tags for organization, e.g., ["report", "weekly"]. |
| format_hint   | string             | No       | Format indicator: "markdown", "pdf", "json", etc. |
| parse_mode    | string             | No       | Parsing strategy: "auto" (default), "raw", "chunk", "structured". |
| pipeline_hint | string or array    | No       | Suggested downstream processing steps: "insert-tidb", "embedding", "summary", etc. Final routing is determined by LLM. |
| llm_hint      | string             | No       | Instruction to LLM about how to interpret or summarize the input. |
| sync          | boolean            | No       | If true, waits for processing (recommended only for small payloads). |
| callback_url  | string             | No       | Optional webhook URL that will be called when processing is complete. |

---

### Example Requests

#### Example 1: Natural Language Input

```json
{
  "input": "Today’s meeting covered three key decisions...",
  "llm_hint": "Summarize the key points",
  "pipeline_hint": ["summary", "insert-tidb"],
  "callback_url": "https://your-service.com/webhook/task-complete"
}
```

#### Example 2: File Upload via URL

```json
{
  "file": "https://example.com/financial-report.pdf",
  "metadata": {
    "title": "2025 Q3 Financial Report",
    "author": "Finance Dept",
    "lang": "en"
  },
  "parse_mode": "structured",
  "pipeline_hint": ["ocr", "chunk", "vectorize", "insert-tidb"],
  "llm_hint": "Extract key revenue and profit figures",
  "callback_url": "https://example.com/hooks/processing_done"
}
```

#### Example 3: Structured JSON Document

```json
{
  "input": {
    "schema": "blog",
    "title": "Product Launch",
    "body": "We launched a new flexible API today..."
  },
  "format_hint": "json",
  "pipeline_hint": ["insert-tidb"],
  "callback_url": "https://your-api.com/notify"
}
```

---

### Response Format

```json
{
  "task_id": "task_abc123",
  "status": "queued",
  "result_url": "/api/result/task_abc123",
  "estimated_time": "5s"
}
```

---

### Webhook Callback Format

If a `callback_url` is provided, the system will send a `POST` request to that URL upon task completion with the following payload:

```json
{
  "task_id": "task_abc123",
  "status": "success",
  "result": {
    "parsed_content": "...",
    "saved_location": "tidb://documents/abc123",
    "summary": "...optional summary..."
  },
  "errors": null
}
```

#### In case of error:

```json
{
  "task_id": "task_abc123",
  "status": "failed",
  "errors": ["Parse error: file format not supported"]
}
```

---

### Notes

- All inputs are optional but at least one of `input`, `file`, or `url` must be present.
- The server determines the processing pipeline dynamically using LLM and other heuristics, optionally guided by `pipeline_hint`.
- Downstream tools like TiDB, Pandoc, embedding models, summarizers, or agent routers can be configured in the backend.
- If `callback_url` is provided, a webhook will be sent upon task completion with the full result.
