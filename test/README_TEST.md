# Test Script for process_request() Function

This directory contains a comprehensive test suite for testing the `process_request()` function in different scenarios.

## Files Created

1. `test_process_request.py` - Main test script with example scenarios

## Test Scenarios

### Scenario 1: Adding Single Document to Existing Topic
- **Purpose**: Test processing a single document into an existing knowledge graph topic
- **Configuration**: `is_new_topic: False`
- **Files**: 1 test document

### Scenario 2: Adding Batch Documents to Existing Topic  
- **Purpose**: Test processing multiple documents into an existing topic
- **Configuration**: `is_new_topic: False`, `batch_processing: True`
- **Files**: 3 test documents

### Scenario 3: Creating New Topic with Batch Documents
- **Purpose**: Test creating a new topic with multiple documents
- **Configuration**: `is_new_topic: True`, `batch_processing: True` 
- **Files**: 2 test documents with unique topic name

### Scenario 4: Memory Processing

## How to Run the Tests

### Method 1: Direct Python
```bash
python3 test_process_request.py
```

## Mock Objects

The test script includes mock implementations for:
- **MockLLMClient**: Simulates LLM responses
- **MockEmbeddingFunc**: Generates deterministic embeddings

## Test Output

Each test scenario provides:
- ✅/❌ Success status
- Execution ID for tracking
- Processing messages
- Data structure information
- Files processed count

## Cleanup

The test script automatically:
- Creates temporary test files
- Cleans up files after execution
- Provides detailed error messages for debugging
