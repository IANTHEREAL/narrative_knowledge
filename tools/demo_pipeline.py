"""
Demo script for flexible pipeline orchestration.

This script demonstrates the three main scenarios outlined in the design document:
1. Adding a single new document to an existing topic
2. Adding a batch of new documents to an existing topic
3. Creating a new topic with a batch of documents
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.orchestrator import PipelineOrchestrator
from tools.api_integration import PipelineAPIIntegration

# Create some demo files
def create_demo_files():
    """Create demo files for testing."""
    demo_dir = Path("demo_files")
    demo_dir.mkdir(exist_ok=True)
    
    files = []
    
    # Create demo PDF
    pdf_content = """
    # TiDB Architecture Overview
    
    TiDB is an open-source distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads.
    
    ## Key Components
    - TiDB Server: SQL layer
    - TiKV: Distributed key-value storage
    - PD: Placement Driver for metadata management
    - TiFlash: Columnar storage for analytics
    
    ## Architecture Features
    TiDB separates compute from storage, allowing independent scaling of SQL processing and storage capacity.
    """
    
    pdf_file = demo_dir / "tidb_architecture.txt"
    with open(pdf_file, 'w') as f:
        f.write(pdf_content)
    files.append(str(pdf_file))
    
    # Create demo Markdown
    md_content = """
    # TiDB Performance Optimization
    
    ## Query Optimization Techniques
    - Use appropriate indexes
    - Optimize JOIN operations
    - Leverage partition tables
    
    ## System Tuning
    - Configure TiKV parameters
    - Monitor performance metrics
    - Scale horizontally as needed
    """
    
    md_file = demo_dir / "tidb_performance.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    files.append(str(md_file))
    
    return files

def demo_single_document_existing_topic():
    """Demo: Adding a single new document to an existing topic."""
    print("\n=== Demo 1: Single Document to Existing Topic ===")
    
    orchestrator = PipelineOrchestrator()
    
    # Simulate existing topic by processing one file first
    files = create_demo_files()
    
    # First, create the topic with one document
    context = {
        "file_path": files[0],
        "topic_name": "tidb_docs",
        "metadata": {"source": "demo"}
    }
    
    result = orchestrator.execute_pipeline("new_topic_batch", context)
    print(f"Created topic: {result.success}")
    
    # Now add a single document to existing topic
    single_context = {
        "file_path": files[1],
        "topic_name": "tidb_docs",
        "metadata": {"source": "demo_addition"}
    }
    
    result = orchestrator.execute_pipeline("single_doc_existing_topic", single_context)
    print(f"Added single document: {result.success}")
    print(f"Pipeline used: {result.data.get('pipeline', [])}")

def demo_batch_documents_existing_topic():
    """Demo: Adding a batch of new documents to an existing topic."""
    print("\n=== Demo 2: Batch Documents to Existing Topic ===")
    
    integration = PipelineAPIIntegration()
    
    files = create_demo_files()
    
    # Process batch using API integration
    request_data = {
        "target_type": "knowledge_graph",
        "metadata": {
            "topic_name": "tidb_docs",
            "is_new_topic": False
        },
        "files": [{"path": f} for f in files]
    }
    
    result = integration.process_request(request_data, files)
    print(f"Batch processing result: {result.success}")
    if result.success:
        print(f"Pipeline used: {result.data.get('pipeline', [])}")
        print(f"Duration: {result.duration_seconds:.2f}s")

def demo_new_topic_batch():
    """Demo: Creating a new topic with a batch of documents."""
    print("\n=== Demo 3: New Topic with Batch Documents ===")
    
    integration = PipelineAPIIntegration()
    
    files = create_demo_files()
    
    # Process new topic with explicit pipeline
    request_data = {
        "target_type": "knowledge_graph",
        "metadata": {
            "topic_name": "new_tidb_topic",
            "is_new_topic": True
        },
        "process_strategy": {
            "pipeline": ["etl", "blueprint_gen", "graph_build"]
        }
    }
    
    result = integration.process_request(request_data, files)
    print(f"New topic creation: {result.success}")
    if result.success:
        print(f"Explicit pipeline used: {result.data.get('pipeline', [])}")

def demo_explicit_pipeline():
    """Demo: Using explicit pipeline configuration."""
    print("\n=== Demo 4: Explicit Pipeline Configuration ===")
    
    integration = PipelineAPIIntegration()
    
    files = create_demo_files()
    
    # Use custom pipeline sequence
    request_data = {
        "target_type": "knowledge_graph",
        "metadata": {"topic_name": "custom_pipeline_demo"},
        "process_strategy": {
            "pipeline": ["etl", "blueprint_gen", "graph_build"]
        }
    }
    
    result = integration.process_request(request_data, files)
    print(f"Custom pipeline execution: {result.success}")
    print(f"Custom pipeline: {result.data.get('pipeline', [])}")

def main():
    """Run all demos."""
    print("Flexible Pipeline Orchestration Demo")
    print("====================================")
    
    # Ensure demo files exist
    demo_files = create_demo_files()
    print(f"Created demo files: {demo_files}")
    
    # Run demos
    demo_single_document_existing_topic()
    demo_batch_documents_existing_topic()
    demo_new_topic_batch()
    demo_explicit_pipeline()
    
    print("\n=== Demo Complete ===")
    print("All pipeline scenarios demonstrated successfully!")

if __name__ == "__main__":
    main()