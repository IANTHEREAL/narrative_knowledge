# """
# Demo script showing how to use the three core tools.

# This demonstrates the tool-based architecture in action.
# """

# from tools.base import TOOL_REGISTRY
# from tools.document_etl_tool import DocumentETLTool
# from tools.blueprint_generation_tool import BlueprintGenerationTool
# from tools.graph_build_tool import GraphBuildTool


# def demo_tools():
#     """Demonstrate the three core tools."""
    
#     print("=== Tool-Based Knowledge Graph Construction Demo ===\n")
    
#     # List available tools
#     print("Available tools:")
#     for tool_name in TOOL_REGISTRY.list_tools():
#         tool = TOOL_REGISTRY.get_tool(tool_name)
#         print(f"  - {tool_name}: {tool.tool_description}")
    
#     print("\n=== Tool Usage Examples ===\n")
    
#     # Example 1: DocumentETLTool
#     print("1. DocumentETLTool:")
#     tool = TOOL_REGISTRY.get_tool("DocumentETLTool")
#     print(f"   Required inputs: {tool.get_required_inputs()}")
#     print("   Usage:")
#     print("   ```python")
#     print("   result = TOOL_REGISTRY.execute_tool('DocumentETLTool', {")
#     print("       'file_path': '/path/to/document.pdf',")
#     print("       'topic_name': 'machine-learning',")
#     print("       'metadata': {'author': 'researcher', 'year': 2024}")
#     print("   })")
#     print("   ```")
    
#     # Example 2: BlueprintGenerationTool
#     print("\n2. BlueprintGenerationTool:")
#     tool = TOOL_REGISTRY.get_tool("BlueprintGenerationTool")
#     print(f"   Required inputs: {tool.get_required_inputs()}")
#     print("   Usage:")
#     print("   ```python")
#     print("   result = TOOL_REGISTRY.execute_tool('BlueprintGenerationTool', {")
#     print("       'topic_name': 'machine-learning',")
#     print("       'force_regenerate': False")
#     print("   })")
#     print("   ```")
    
#     # Example 3: GraphBuildTool
#     print("\n3. GraphBuildTool:")
#     tool = TOOL_REGISTRY.get_tool("GraphBuildTool")
#     print(f"   Required inputs: {tool.get_required_inputs()}")
#     print("   Usage:")
#     print("   ```python")
#     print("   result = TOOL_REGISTRY.execute_tool('GraphBuildTool', {")
#     print("       'source_data_id': 'sd_12345',")
#     print("       'blueprint_id': 'bp_67890',")
#     print("       'force_reprocess': False")
#     print("   })")
#     print("   ```")
    
#     print("\n=== Pipeline Examples ===\n")
    
#     print("Scenario 1: Adding a single new document to existing topic")
#     print("Pipeline: [DocumentETLTool → GraphBuildTool]")
#     print("1. Process document with DocumentETLTool")
#     print("2. Use existing blueprint in GraphBuildTool")
    
#     print("\nScenario 2: Creating new topic with batch of documents")
#     print("Pipeline: [DocumentETLTool × N → BlueprintGenerationTool → GraphBuildTool × N]")
#     print("1. Process all documents with DocumentETLTool (parallel)")
#     print("2. Generate blueprint with BlueprintGenerationTool")
#     print("3. Build graph for each document with GraphBuildTool (parallel)")


# if __name__ == "__main__":
#     demo_tools()