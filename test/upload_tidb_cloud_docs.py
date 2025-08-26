#!/usr/bin/env python3
"""
Parse TOC-tidb-cloud.md to extract all referenced file paths and generate
a targeted upload script for only the files listed in the TOC.
"""

import re
import json
from pathlib import Path

class TOCParser:
    def __init__(self, toc_path="/Users/hjy/Downloads/docs-master/TOC-tidb-cloud.md"):
        self.toc_path = Path(toc_path)
        self.docs_base_path = self.toc_path.parent
        self.base_url = "https://github.com/pingcap/docs/blob/master"
        
    def extract_file_paths(self):
        """Extract all file paths from TOC-tidb-cloud.md"""
        file_paths = []
        
        with open(self.toc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match markdown links: [text](/path)
        # Handle both absolute and relative paths
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for match in re.finditer(link_pattern, content):
            link_text, link_path = match.groups()
            
            # Skip external URLs and special cases
            if link_path.startswith('http') or link_path.startswith('#'):
                continue
                
            # Remove leading slash if present
            if link_path.startswith('/'):
                link_path = link_path[1:]
                
            # Only include .md files
            if link_path.endswith('.md'):
                file_paths.append(link_path)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in file_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        return unique_paths
    
    def group_files_by_section(self, file_paths):
        """Group files by their logical sections from TOC with specific topic names"""
        sections = {
            "tidb_cloud_introduction": {
                "files": [],
                "keywords": ["tidb-cloud/tidb-cloud-intro.md", "mysql-compatibility.md"]
            },
            "tidb_cloud_quickstart": {
                "files": [],
                "keywords": ["tidb-cloud/tidb-cloud-quickstart.md", "tidb-cloud/get-started-with-cli.md", "tidb-cloud/tidb-cloud-poc.md"]
            },
            "tidb_cloud_concepts": {
                "files": [],
                "keywords": ["tidb-cloud/key-concepts.md", "tidb-cloud/architecture-concepts.md", "tidb-cloud/database-schema-concepts.md", "tidb-cloud/transaction-concepts.md", "tidb-cloud/sql-concepts.md"]
            },
            "tidb_cloud_ai_features": {
                "files": [],
                "keywords": ["tidb-cloud/ai-feature-concepts.md", "vector-search/vector-search-get-started-using-python.md", "vector-search/vector-search-overview.md"]
            },
            "tidb_development_guide": {
                "files": [],
                "keywords": ["develop/dev-guide-overview.md", "develop/dev-guide-build-cluster-in-cloud.md", "develop/dev-guide-tidb-crud-sql.md"]
            },
            "database_tools_integration": {
                "files": [],
                "keywords": ["develop/dev-guide-gui-"]
            },
            "java_development": {
                "files": [],
                "keywords": ["develop/dev-guide-sample-application-java-"]
            },
            "python_development": {
                "files": [],
                "keywords": ["develop/dev-guide-sample-application-python-"]
            },
            "nodejs_development": {
                "files": [],
                "keywords": ["develop/dev-guide-sample-application-nodejs-"]
            },
            "database_design": {
                "files": [],
                "keywords": ["develop/dev-guide-schema-design-overview.md", "develop/dev-guide-create-database.md", "develop/dev-guide-create-table.md", "develop/dev-guide-create-secondary-indexes.md"]
            },
            "data_operations": {
                "files": [],
                "keywords": ["develop/dev-guide-insert-data.md", "develop/dev-guide-update-data.md", "develop/dev-guide-delete-data.md", "develop/dev-guide-get-data-from-single-table.md"]
            },
            "performance_optimization": {
                "files": [],
                "keywords": ["develop/dev-guide-optimize-sql-overview.md", "develop/dev-guide-optimize-sql.md", "develop/dev-guide-index-best-practice.md"]
            },
            "cluster_management": {
                "files": [],
                "keywords": ["tidb-cloud/create-tidb-cluster-serverless.md", "tidb-cloud/create-tidb-cluster.md", "tidb-cloud/scale-tidb-cluster.md", "tidb-cloud/upgrade-tidb-cluster.md", "tidb-cloud/delete-tidb-cluster.md"]
            },
            "connection_management": {
                "files": [],
                "keywords": ["tidb-cloud/connect-to-tidb-cluster-serverless.md", "tidb-cloud/connect-to-tidb-cluster.md", "tidb-cloud/set-up-private-endpoint-connections-serverless.md", "tidb-cloud/set-up-private-endpoint-connections.md"]
            },
            "backup_restore": {
                "files": [],
                "keywords": ["tidb-cloud/backup-and-restore.md", "tidb-cloud/backup-and-restore-serverless.md"]
            },
            "monitoring_alerts": {
                "files": [],
                "keywords": ["tidb-cloud/monitor-tidb-cluster.md", "tidb-cloud/built-in-monitoring.md", "tidb-cloud/monitor-built-in-alerting.md"]
            },
            "performance_tuning": {
                "files": [],
                "keywords": ["tidb-cloud/tidb-cloud-tune-performance-overview.md", "tidb-cloud/tune-performance.md", "tidb-cloud/index-insight.md"]
            },
            "data_migration": {
                "files": [],
                "keywords": ["tidb-cloud/tidb-cloud-migration-overview.md", "tidb-cloud/migrate-from-mysql-using-data-migration.md", "tidb-cloud/migrate-from-op-tidb.md"]
            },
            "data_import": {
                "files": [],
                "keywords": ["tidb-cloud/import-sample-data.md", "tidb-cloud/import-csv-files.md", "tidb-cloud/import-parquet-files.md", "tidb-cloud/import-with-mysql-cli.md"]
            },
            "tiftlash_analytics": {
                "files": [],
                "keywords": ["tiflash/tiflash-overview.md", "tiflash/create-tiflash-replicas.md", "tiflash/use-tidb-to-read-tiflash.md", "tiflash/use-tiflash-mpp-mode.md"]
            },
            "vector_search_guide": {
                "files": [],
                "keywords": ["vector-search/vector-search-get-started-using-sql.md", "vector-search/vector-search-functions-and-operators.md", "vector-search/vector-search-index.md"]
            },
            "vector_search_integrations": {
                "files": [],
                "keywords": ["vector-search/vector-search-integrate-with-llamaindex.md", "vector-search/vector-search-integrate-with-langchain.md"]
            },
            "data_service_api": {
                "files": [],
                "keywords": ["tidb-cloud/data-service-overview.md", "tidb-cloud/data-service-get-started.md", "tidb-cloud/use-chat2query-api.md"]
            },
            "changefeed_streaming": {
                "files": [],
                "keywords": ["tidb-cloud/changefeed-overview.md", "tidb-cloud/changefeed-sink-to-mysql.md", "tidb-cloud/changefeed-sink-to-apache-kafka.md"]
            },
            "security_authentication": {
                "files": [],
                "keywords": ["tidb-cloud/tidb-cloud-password-authentication.md", "tidb-cloud/tidb-cloud-sso-authentication.md", "tidb-cloud/manage-user-access.md", "tidb-cloud/configure-security-settings.md"]
            },
            "sql_statements_reference": {
                "files": [],
                "keywords": ["sql-statements/"]
            },
            "sql_data_types": {
                "files": [],
                "keywords": ["data-type-overview.md", "data-type-numeric.md", "data-type-date-and-time.md", "data-type-string.md", "data-type-json.md"]
            },
            "sql_functions_operators": {
                "files": [],
                "keywords": ["functions-and-operators/"]
            },
            "tidb_architecture": {
                "files": [],
                "keywords": ["tidb-architecture.md", "tidb-storage.md", "tidb-computing.md", "tidb-scheduling.md", "tso.md"]
            },
            "transaction_management": {
                "files": [],
                "keywords": ["transaction-overview.md", "transaction-isolation-levels.md", "optimistic-transaction.md", "pessimistic-transaction.md"]
            },
            "system_tables_reference": {
                "files": [],
                "keywords": ["information-schema/", "performance-schema/", "mysql-schema/"]
            }
        }
        
        # Assign files to sections
        for file_path in file_paths:
            assigned = False
            
            # Check specific section keywords
            for section_name, section_data in sections.items():
                for keyword in section_data["keywords"]:
                    if keyword in file_path:
                        sections[section_name]["files"].append(file_path)
                        assigned = True
                        break
                if assigned:
                    break
            
            # If not assigned to specific section, use general categorization
            if not assigned:
                # Check file path patterns
                if file_path.startswith('tidb-cloud/'):
                    sections["cluster_management"]["files"].append(file_path)
                elif 'develop/' in file_path:
                    sections["tidb_development_guide"]["files"].append(file_path)
                elif file_path.startswith('sql-statements/'):
                    sections["sql_statements_reference"]["files"].append(file_path)
                elif 'functions-and-operators/' in file_path:
                    sections["sql_functions_operators"]["files"].append(file_path)
                elif file_path.startswith('data-type-'):
                    sections["sql_data_types"]["files"].append(file_path)
                elif 'transaction' in file_path:
                    sections["transaction_management"]["files"].append(file_path)
                elif 'information-schema/' in file_path or 'performance-schema/' in file_path or 'mysql-schema/' in file_path:
                    sections["system_tables_reference"]["files"].append(file_path)
                else:
                    sections["tidb_architecture"]["files"].append(file_path)
        
        # Remove empty sections
        sections = {k: v for k, v in sections.items() if v["files"]}
        
        return sections
    
    def create_upload_script(self, sections):
        """Create upload script for TOC files with specific topic names"""
        script_content = """#!/bin/bash
# TiDB Cloud Documentation Upload Script
# Uploads ONLY files referenced in TOC-tidb-cloud.md with specific topic names

# Set the base path for documentation files
DOCS_PATH="${1:-/Users/hjy/Downloads/docs-master}"

echo "Using documentation path: $DOCS_PATH"
echo "Uploading TiDB Cloud documentation files from TOC..."

"""
        
        total_files = sum(len(section["files"]) for section in sections.values())
        script_content += f'echo "Total TOC files to upload: {total_files}"\n\n'
        
        # Generate upload commands for each section
        for section_name, section_data in sections.items():
            files = section_data["files"]
            if not files:
                continue
            
            # Use the section name as topic name (already specific)
            topic_name = section_name
            
            # Group files into batches of 5, but keep the same topic name
            batch_size = 5
            file_batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
            
            for batch_idx, file_batch in enumerate(file_batches, 1):
                # Use descriptive comment instead of batch number in topic name
                if len(file_batches) == 1:
                    script_content += f"# {section_name.replace('_', ' ').title()} ({len(file_batch)} files)\n"
                else:
                    script_content += f"# {section_name.replace('_', ' ').title()} - Part {batch_idx} ({len(file_batch)} files)\n"
                
                script_content += "curl -X POST \"http://localhost:8000/api/v1/save\" \\\n"
                
                # Add files
                for file_path in file_batch:
                    script_content += f"  -F \"files=@$DOCS_PATH/{file_path}\" \\\n"
                
                # Add links
                links = [f"{self.base_url}/{file_path}" for file_path in file_batch]
                links_json = json.dumps(links)
                script_content += f"  -F 'links={links_json}' \\\n"
                
                # Add metadata - use the specific topic name for all batches
                metadata = {"topic_name": topic_name, "force_regenerate": "False"}
                script_content += f"  -F 'metadata={json.dumps(metadata)}' \\\n"
                script_content += "  -F \"target_type=knowledge_graph\"\n\n"
        
        script_content += """
# Usage:
# ./upload_tidb_cloud_docs.sh
# ./upload_tidb_cloud_docs.sh /custom/path/to/docs
"""
        
        return script_content
    
    def create_summary(self, sections):
        """Create a summary of files organized by section"""
        summary = "# TiDB Cloud TOC Files Summary\n\n"
        
        total_files = sum(len(section["files"]) for section in sections.values())
        summary += f"Total files from TOC: {total_files}\n\n"
        
        for section_name, section_data in sections.items():
            files = section_data["files"]
            if files:
                summary += f"## {section_name.replace('_', ' ').title()} ({len(files)} files)\n"
                for file_path in files:
                    summary += f"- {file_path}\n"
                summary += "\n"
        
        return summary

def main():
    parser = TOCParser()
    
    # Extract file paths from TOC
    file_paths = parser.extract_file_paths()
    print(f"Found {len(file_paths)} files referenced in TOC")
    
    # Group files by section
    sections = parser.group_files_by_section(file_paths)
    
    # Create upload script
    upload_script = parser.create_upload_script(sections)
    
    # Create summary
    summary = parser.create_summary(sections)
    
    # Save files
    with open("upload_tidb_cloud_docs.sh", "w") as f:
        f.write(upload_script)
    
    with open("toc_files_summary.md", "w") as f:
        f.write(summary)
    
    # Make script executable
    import os
    os.chmod("upload_tidb_cloud_docs.sh", 0o755)
    
    print("\nGenerated files:")
    print("1. upload_tidb_cloud_docs.sh - Upload script for TOC files")
    print("2. toc_files_summary.md - Summary of TOC files by section")
    
    print("\nFiles by section:")
    for section_name, section_data in sections.items():
        print(f"  {section_name}: {len(section_data['files'])} files")

if __name__ == "__main__":
    main()