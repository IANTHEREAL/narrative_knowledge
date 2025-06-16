# Narrative Knowledge API

This module provides a REST API for the Narrative Knowledge system, enabling document upload, processing, and querying for knowledge graph building.

## Quick Start

### 1. Install Dependencies

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 2. Start the API Server

```bash
poetry run python -m api.main
```

The API will be available at `http://localhost:8000`

### 3. View API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Upload Documents

**POST** `/api/v1/knowledge/upload`

Upload and process documents for knowledge graph building with batch support.

**Parameters:**
- `files`: One or more files (PDF, Markdown, TXT, SQL)
- `links`: List of links to original documents (must match number of files)
  - Recommended to use accessible links
  - If not available, you can use custom unique addresses
  - Must ensure uniqueness
- `topic_name`: Topic name for knowledge graph building (required)
- `database_uri`: Database connection string (optional, uses local if not provided)

**Example using curl (single file):**

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/upload" \
  -F "files=@document.pdf" \
  -F "links=https://docs.google.com/document/d/abc123" \
  -F "topic_name=project-alpha"
```

**Example using curl (batch upload):**

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.md" \
  -F "links=https://docs.google.com/document/d/abc123" \
  -F "links=https://github.com/repo/readme" \
  -F "topic_name=project-alpha"
```

**Response:**
```json
{
  "status": "success",
  "message": "Batch upload completed: 2/2 documents processed successfully",
  "data": {
    "uploaded_count": 2,
    "total_count": 2,
    "success_rate": 1.0,
    "documents": [
      {
        "id": "cc5b92b6-ef73-4c4d-8b54-60c610d3443d",
        "name": "document1",
        "file_path": "uploads/project-alpha/document1/document1.pdf",
        "doc_link": "https://docs.google.com/document/d/abc123",
        "file_type": "pdf",
        "status": "processed"
      },
      {
        "id": "dd6c03c7-fg84-5d5e-9c65-71d721e4554e",
        "name": "document2",
        "file_path": "uploads/project-alpha/document2/document2.md",
        "doc_link": "https://github.com/repo/readme",
        "file_type": "markdown",
        "status": "processed"
      }
    ],
    "failed": []
  }
}
```

### Trigger Document Processing

**POST** `/api/v1/knowledge/trigger-processing`

Manually trigger processing for documents in a specific topic.

**Parameters:**
- `topic_name`: Name of the topic to trigger processing for (required)
- `database_uri`: Database URI to filter tasks (optional)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/trigger-processing" \
  -F "topic_name=project-alpha"
```

**Response:**
```json
{
  "status": "success",
  "message": "Processing triggered for topic: project-alpha",
  "data": {
    "topic_name": "project-alpha",
    "triggered_count": 2,
    "total_documents": 2
  }
}
```

### List Topics

**GET** `/api/v1/knowledge/topics`

Retrieve a list of all topics with their processing status summary.

**Query Parameters:**
- `database_uri`: Filter topics by database URI (optional)
  - Empty string or not provided: Local database topics
  - Specific URI: External database topics

**Example:**
```bash
curl "http://localhost:8000/api/v1/knowledge/topics"
```

**Example with database filter:**
```bash
curl "http://localhost:8000/api/v1/knowledge/topics?database_uri=postgresql://user:pass@host:5432/db"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "topics": [
      {
        "topic_name": "project-alpha",
        "total_documents": 5,
        "uploaded_count": 5,
        "pending_count": 1,
        "processing_count": 0,
        "completed_count": 3,
        "failed_count": 1,
        "latest_update": "2024-01-15T10:35:00",
        "database_uri": ""
      },
      {
        "topic_name": "project-beta",
        "total_documents": 3,
        "uploaded_count": 3,
        "pending_count": 0,
        "processing_count": 1,
        "completed_count": 2,
        "failed_count": 0,
        "latest_update": "2024-01-15T11:20:00",
        "database_uri": "postgresql://user:pass@host:5432/external_db"
      }
    ],
    "total_topics": 2,
    "filter_database_uri": null,
    "source": "local_database"
  }
}
```

## Configuration

### Supported File Types
- PDF (`.pdf`) → document
- Markdown (`.md`) → document  
- Text (`.txt`) → document
- SQL (`.sql`) → code

### File Size Limits
- Maximum total file size for batch upload: 30MB
- Individual file size limit: 30MB

### Document Storage
Documents are stored in a versioned directory structure:
- Base path: `uploads/<topic_name>/<filename>/`
- Each document directory contains:
  - The original file
  - `document_metadata.json` with document information
- If a document with the same metadata exists, it will reuse the existing directory
- If a document with different metadata exists, a new versioned directory will be created (e.g., `filename_v2/`)

## Error Handling

The API uses standardized error responses for all endpoints:

**HTTP Error Response:**
```json
{
  "error": {
    "code": "HTTP_ERROR",
    "message": "Error message here",
    "status_code": 400
  }
}
```

**Internal Error Response:**
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred"
  }
}
```