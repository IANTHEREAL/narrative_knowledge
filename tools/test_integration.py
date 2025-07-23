"""
Integration test for the flexible pipeline system.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os

from tools.orchestrator import PipelineOrchestrator
from tools.api_integration import PipelineAPIIntegration
from tools.base import ToolRegistry

class TestPipelineIntegration(unittest.TestCase):
    """Test the complete pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = PipelineOrchestrator()
        self.integration = PipelineAPIIntegration()
        
    def test_tool_registry_mapping(self):
        """Test that all tools are properly registered with correct keys."""
        from tools.base import TOOL_REGISTRY
        
        # Test tool keys match design document
        expected_tools = {
            "etl": "DocumentETLTool",
            "blueprint_gen": "BlueprintGenerationTool", 
            "graph_build": "GraphBuildTool"
        }
        
        for key, expected_name in expected_tools.items():
            tool = TOOL_REGISTRY.get_tool(expected_name)
            self.assertIsNotNone(tool, f"Tool {expected_name} should be registered")
            
    def test_orchestrator_pipeline_mapping(self):
        """Test pipeline definitions match design document scenarios."""
        expected_pipelines = {
            "single_doc_existing_topic": ["etl", "graph_build"],
            "batch_doc_existing_topic": ["etl", "blueprint_gen", "graph_build"],
            "new_topic_batch": ["etl", "blueprint_gen", "graph_build"],
            "memory_direct_graph": ["graph_build"],
            "memory_single": ["graph_build"],
            "text_to_graph": ["graph_build"]
        }
        
        for pipeline_name, expected_tools in expected_pipelines.items():
            self.assertIn(pipeline_name, self.orchestrator.standard_pipelines)
            self.assertEqual(
                self.orchestrator.standard_pipelines[pipeline_name],
                expected_tools
            )
    
    def test_process_strategy_parsing(self):
        """Test process strategy parameter parsing."""
        # Test explicit pipeline
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {"topic_name": "test"},
            "process_strategy": {
                "pipeline": ["etl", "blueprint_gen", "graph_build"]
            }
        }
        
        tools = [self.orchestrator.tool_key_mapping.get(key, key) 
                for key in request_data["process_strategy"]["pipeline"]]
        
        expected = ["DocumentETLTool", "BlueprintGenerationTool", "GraphBuildTool"]
        self.assertEqual(tools, expected)
    
    def test_default_pipeline_selection(self):
        """Test intelligent pipeline selection with memory support."""
        # Knowledge graph tests
        pipeline = self.orchestrator.select_default_pipeline(
            "knowledge_graph", "existing_topic", 1, False
        )
        self.assertEqual(pipeline, "single_doc_existing_topic")
        
        pipeline = self.orchestrator.select_default_pipeline(
            "knowledge_graph", "existing_topic", 3, False
        )
        self.assertEqual(pipeline, "batch_doc_existing_topic")
        
        pipeline = self.orchestrator.select_default_pipeline(
            "knowledge_graph", "new_topic", 2, True
        )
        self.assertEqual(pipeline, "new_topic_batch")
        
        # Memory pipeline tests
        pipeline = self.orchestrator.select_default_pipeline(
            "personal_memory", "user123", 0, False, input_type="dialogue"
        )
        self.assertEqual(pipeline, "memory_direct_graph")
        
        pipeline = self.orchestrator.select_default_pipeline(
            "personal_memory", "user123", 1, False
        )
        self.assertEqual(pipeline, "memory_single")
    
    def test_api_integration_context_preparation(self):
        """Test API integration context preparation."""
        request_data = {
            "target_type": "knowledge_graph",
            "metadata": {"topic_name": "test_topic"},
            "process_strategy": {"pipeline": ["etl", "graph_build"]}
        }
        files = [{"path": "/test/file.pdf"}]
        
        context = self.integration._prepare_context(request_data, files)
        
        self.assertEqual(context["target_type"], "knowledge_graph")
        self.assertEqual(context["metadata"]["topic_name"], "test_topic")
        self.assertEqual(len(context["files"]), 1)

if __name__ == "__main__":
    unittest.main()