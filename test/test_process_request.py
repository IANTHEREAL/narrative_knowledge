#!/usr/bin/env python3
"""
Test script for process_request() function with four scenarios:
1. Adding single document to existing topic
2. Adding batch documents to existing topic
3. Creating new topic with batch documents
4. Processing memory stream (chat messages) with personal memory system
"""

import os
import tempfile
from typing import Dict, Any, List
import uuid

# Import the pipeline integration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_integration import PipelineAPIIntegration

# Mock classes for testing
class MockLLMClient:
    """Mock LLM client for testing"""
    def __init__(self):
        self.name = "mock_llm"
    
    def generate(self, prompt: str) -> str:
        return f"Mock response for: {prompt[:50]}..."

class MockEmbeddingFunc:
    """Mock embedding function for testing"""
    def __init__(self):
        self.name = "mock_embedding"
    
    def embed(self, text: str) -> List[float]:
        # Generate deterministic embedding based on text length for testing
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [float((hash_val + i) % 100) / 100.0 for i in range(5)]

def create_test_files(num_files: int = 1) -> List[Dict[str, Any]]:
    """Create temporary test files for testing"""
    files = []
    for i in range(num_files):
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix=f'test_file_{i+1}_')
        with os.fdopen(fd, 'w') as f:
            f.write(f"This is test document number {i+1}.\n")
            f.write("It contains sample content for testing purposes.\n")
            f.write("The content is designed to test the processing pipeline.\n")
        
        files.append({
            "path": temp_path,
            "filename": os.path.basename(temp_path),
            "link": f"file_link {i+1}",     # suggested
            "metadata": {
                "file_type": "text",
                "size": os.path.getsize(temp_path),
                "title": f"Test Document {i+1}",
                # "link": f"file_meta_link {i+1}"
            }
        })
    return files

def clean_test_files(files: List[Dict[str, Any]]):
    """Clean up temporary test files"""
    for file_info in files:
        try:
            if os.path.exists(file_info["path"]):
                os.unlink(file_info["path"])
        except Exception as e:
            print(f"Warning: Could not clean up {file_info['path']}: {e}")

def test_scenario_1_single_document_existing_topic():
    """
    Scenario 1: Adding single document to existing topic
    """
    print("\n" + "="*60)
    print("SCENARIO 1: Adding single document to existing topic")
    print("="*60)
    
    # Initialize the pipeline
    integration = PipelineAPIIntegration()
    
    # Create test file
    test_files = create_test_files(1)
    
    try:
        # Prepare request data - ensure file_path is in context
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {
                "topic_name": "existing_topic_python",
                "is_new_topic": False,  # Existing topic
                "database_uri": "sqlite:///test_knowledge_graph.db",
                "link": f"request_meta_link"    # optional
            },
            "process_strategy": {
                # "pipeline": ["etl", "blueprint_gen", "graph_build"]
            },
            "llm_client": MockLLMClient(),
            "embedding_func": MockEmbeddingFunc(),
            "force_regenerate": False
        }
        
        # Show processing summary
        print(f"   Processing {len(test_files)} file(s)...")
            
        # Execute the request
        result = integration.process_request(request_data, test_files)
        
        print(f"✅ Scenario 1 completed")
        print(f"   Execution ID: {result.execution_id}")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.error_message or 'Success'}")
        if result.data:
            print(f"   Data keys: {list(result.data.keys())}")
            print(f"   Duration: {result.data.get('duration_seconds', 'N/A')}s")
            if 'pipeline' in result.data:
                print(f"   Pipeline: {result.data['pipeline']}")
            if 'results' in result.data:
                print("   Detailed steps:")
                for step_id, step_result in result.data['results'].items():
                    print(f"     - {step_id}: {step_result.success}")
                    if step_result.data:
                        print(f"       Data: {step_result.data}")
                    if step_result.metadata:
                        print(f"       Metadata: {step_result.metadata}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Scenario 1 failed: {e}")
        return False
    finally:
        clean_test_files(test_files)

def test_scenario_2_batch_documents_existing_topic():
    """
    Scenario 2: Adding batch documents to existing topic
    """
    print("\n" + "="*60)
    print("SCENARIO 2: Adding batch documents to existing topic")
    print("="*60)
    
    # Initialize the pipeline
    integration = PipelineAPIIntegration()
    
    # Create multiple test files
    test_files = create_test_files(3)
    
    try:
        # Prepare request data
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {
                "topic_name": "existing_topic_batch",
                "is_new_topic": False,  # Existing topic
                "database_uri": "sqlite:///test_knowledge_graph.db",
                "link": f"request_meta_link"
            },
            "llm_client": MockLLMClient(),
            "embedding_func": MockEmbeddingFunc(),
            "force_regenerate": False
        }
        
        # Show processing summary
        print(f"   Processing {len(test_files)} file(s)...")
            
        # Execute the request
        result = integration.process_request(request_data, test_files)
        
        print(f"✅ Scenario 2 completed")
        print(f"   Execution ID: {result.execution_id}")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.error_message or 'Success'}")
        print(f"   Files processed: {len(test_files)}")
        if result.data:
            print(f"   Data keys: {list(result.data.keys())}")
            print(f"   Duration: {result.data.get('duration_seconds', 'N/A')}s")
            if 'pipeline' in result.data:
                print(f"   Pipeline: {result.data['pipeline']}")
            if 'results' in result.data:
                print("   Detailed steps:")
                for step_id, step_result in result.data['results'].items():
                    print(f"     - {step_id}: {step_result.success}")
                    if step_result.data:
                        print(f"       Data: {step_result.data}")
                    if step_result.metadata:
                        print(f"       Metadata: {step_result.metadata}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Scenario 2 failed: {e}")
        return False
    finally:
        clean_test_files(test_files)

def test_scenario_3_new_topic_batch_documents():
    """
    Scenario 3: Creating new topic with batch documents
    """
    print("\n" + "="*60)
    print("SCENARIO 3: Creating new topic with batch documents")
    print("="*60)
    
    # Initialize the pipeline
    integration = PipelineAPIIntegration()
    
    # Create multiple test files
    test_files = create_test_files(2)
    
    try:
        # Generate unique topic name to ensure it's new
        new_topic_name = f"new_topic_{uuid.uuid4().hex[:8]}"
        
        # Prepare request data
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {
                "topic_name": new_topic_name,
                "is_new_topic": True,  # New topic
                "database_uri": "sqlite:///test_knowledge_graph.db",
                "link": f"request_meta_link",
                "description": f"New topic created for testing with UUID: {new_topic_name}"
            },
            "llm_client": MockLLMClient(),
            "embedding_func": MockEmbeddingFunc(),
            "force_regenerate": False
        }
        
        # Show processing summary
        print(f"   Processing {len(test_files)} file(s)...")
            
        # Execute the request
        result = integration.process_request(request_data, test_files)
        
        print(f"✅ Scenario 3 completed")
        print(f"   Execution ID: {result.execution_id}")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.error_message or 'Success'}")
        print(f"   Topic created: {new_topic_name}")
        print(f"   Files processed: {len(test_files)}")
        if result.data:
            print(f"   Data keys: {list(result.data.keys())}")
            print(f"   Duration: {result.data.get('duration_seconds', 'N/A')}s")
            if 'pipeline' in result.data:
                print(f"   Pipeline: {result.data['pipeline']}")
            if 'results' in result.data:
                print("   Detailed steps:")
                for step_id, step_result in result.data['results'].items():
                    print(f"     - {step_id}: {step_result.success}")
                    if step_result.data:
                        print(f"       Data: {step_result.data}")
                    if step_result.metadata:
                        print(f"       Metadata: {step_result.metadata}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Scenario 3 failed: {e}")
        return False
    finally:
        clean_test_files(test_files)

def test_scenario_4_memory_process_chat_messages():
    """
    Scenario 4: Processing memory stream (chat messages) with personal memory system
    """
    print("\n" + "="*60)
    print("SCENARIO 4: Processing memory stream with personal memory system")
    print("="*60)
    
    # Initialize the pipeline
    integration = PipelineAPIIntegration()
    
    try:
        # Prepare chat messages for memory processing
        chat_messages = [
            {
                "message_content": "Hi, I'm learning Python and want to build a web application.",
                "session_id": "session_001",
                "conversation_title": "Learning Python Web Development",
                "date": "2024-01-15T10:30:00Z",
                "role": "user"
            },
            {
                "message_content": "That's great! Python has excellent web frameworks like Flask and Django. What kind of application do you want to build?",
                "session_id": "session_001", 
                "conversation_title": "Learning Python Web Development",
                "date": "2024-01-15T10:31:00Z",
                "role": "assistant"
            },
            {
                "message_content": "I want to build a personal blog with user authentication and a comment system.",
                "session_id": "session_001",
                "conversation_title": "Learning Python Web Development", 
                "date": "2024-01-15T10:32:00Z",
                "role": "user"
            },
            {
                "message_content": "For that, I'd recommend Django because it has built-in authentication and admin panel. You'll need models for blog posts, users, and comments.",
                "session_id": "session_001",
                "conversation_title": "Learning Python Web Development",
                "date": "2024-01-15T10:33:00Z", 
                "role": "assistant"
            }
        ]
        
        # Prepare request data for memory processing
        request_data = {
            "target_type": "personal_memory",
            "user_id": "user_12345",
            "chat_messages": chat_messages,
            "metadata": {
                "database_uri": "sqlite:///test_personal_memory.db",
                "description": "Personal memory processing for Python learning conversation"
            },
            "llm_client": MockLLMClient(),
            "embedding_func": MockEmbeddingFunc(),
            "force_regenerate": False
        }
        
        # Show processing summary
        print(f"   Processing {len(chat_messages)} chat messages...")
        print(f"   User ID: user_12345")
        print(f"   Conversation: Learning Python Web Development")
        
        # Execute the request
        result = integration.process_request(request_data, [])
        
        print(f"✅ Scenario 4 completed")
        print(f"   Execution ID: {result.execution_id}")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.error_message or 'Success'}")
        if result.data:
            print(f"   Data keys: {list(result.data.keys())}")
            print(f"   Duration: {result.data.get('duration_seconds', 'N/A')}s")
            if 'pipeline' in result.data:
                print(f"   Pipeline: {result.data['pipeline']}")
            if 'results' in result.data:
                print("   Detailed steps:")
                for step_id, step_result in result.data['results'].items():
                    print(f"     - {step_id}: {step_result.success}")
                    if step_result.data:
                        print(f"       Data: {step_result.data}")
                    if step_result.metadata:
                        print(f"       Metadata: {step_result.metadata}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Scenario 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all four test scenarios"""
    print("🚀 Starting process_request() test suite")
    print("Testing four scenarios for knowledge graph and memory processing")
    
    results = []
    
    # Run each scenario
    results.append(test_scenario_1_single_document_existing_topic())
    results.append(test_scenario_2_batch_documents_existing_topic())
    results.append(test_scenario_3_new_topic_batch_documents())
    results.append(test_scenario_4_memory_process_chat_messages())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    scenario_names = [
        "Single document to existing topic",
        "Batch documents to existing topic", 
        "New topic with batch documents",
        "Memory processing with chat messages"
    ]
    
    for i, (name, success) in enumerate(zip(scenario_names, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check the logs above")
    
    return passed == total

if __name__ == "__main__":
    # Change to the directory containing the test files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)