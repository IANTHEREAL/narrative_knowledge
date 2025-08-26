#!/bin/bash
# TiDB Cloud Documentation Upload Script
# Uploads ONLY files referenced in TOC-tidb-cloud.md with specific topic names

# Set the base path for documentation files
DOCS_PATH="${1:-/Users/hjy/Downloads/docs-master}"

echo "Using documentation path: $DOCS_PATH"
echo "Uploading TiDB Cloud documentation files from TOC..."

echo "Total TOC files to upload: 692"

# Tidb Cloud Introduction (2 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-intro.md" \
  -F "files=@$DOCS_PATH/mysql-compatibility.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-intro.md", "https://github.com/pingcap/docs/blob/master/mysql-compatibility.md"]' \
  -F 'metadata={"topic_name": "tidb_cloud_introduction", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Cloud Quickstart (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-quickstart.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/get-started-with-cli.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-poc.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-quickstart.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/get-started-with-cli.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-poc.md"]' \
  -F 'metadata={"topic_name": "tidb_cloud_quickstart", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Cloud Concepts (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/key-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/architecture-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/database-schema-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/transaction-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/sql-concepts.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/key-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/architecture-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/database-schema-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/transaction-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/sql-concepts.md"]' \
  -F 'metadata={"topic_name": "tidb_cloud_concepts", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Cloud Ai Features (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-get-started-using-python.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ai-feature-concepts.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-overview.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/vector-search/vector-search-get-started-using-python.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ai-feature-concepts.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-overview.md"]' \
  -F 'metadata={"topic_name": "tidb_cloud_ai_features", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-overview.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-build-cluster-in-cloud.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-tidb-crud-sql.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-choose-driver-or-orm.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-connection-parameters.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-overview.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-build-cluster-in-cloud.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-tidb-crud-sql.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-choose-driver-or-orm.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-connection-parameters.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-golang-sql-driver.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-golang-gorm.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nextjs.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-aws-lambda.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-ruby-mysql2.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-golang-sql-driver.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-golang-gorm.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nextjs.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-aws-lambda.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-ruby-mysql2.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-ruby-rails.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-cs.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-prepared-statement.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-join-tables.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-subqueries.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-ruby-rails.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-cs.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-prepared-statement.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-join-tables.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-subqueries.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-paginate-results.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-views.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-temporary-tables.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-common-table-expression.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-follower-read.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-paginate-results.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-views.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-temporary-tables.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-common-table-expression.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-follower-read.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-use-stale-read.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-hybrid-oltp-and-olap-queries.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-transaction-restraints.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-transaction-troubleshoot.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-optimize-sql-best-practices.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-use-stale-read.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-hybrid-oltp-and-olap-queries.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-transaction-restraints.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-transaction-troubleshoot.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-optimize-sql-best-practices.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-implicit-type-conversion.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-unique-serial-number-generation.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-troubleshoot-overview.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-unstable-result-set.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-timeouts-in-tidb.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-implicit-type-conversion.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-unique-serial-number-generation.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-troubleshoot-overview.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-unstable-result-set.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-timeouts-in-tidb.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 7 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-object-naming-guidelines.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sql-development-specification.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-bookshop-schema-design.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-third-party-support.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-third-party-tools-compatibility.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-object-naming-guidelines.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sql-development-specification.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-bookshop-schema-design.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-third-party-support.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-third-party-tools-compatibility.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Development Guide - Part 8 (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-aws-appflow-integration.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-playground-gitpod.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-proxysql-integration.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-aws-appflow-integration.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-playground-gitpod.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-proxysql-integration.md"]' \
  -F 'metadata={"topic_name": "tidb_development_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Database Tools Integration (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-gui-datagrip.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-gui-dbeaver.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-gui-vscode-sqltools.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-gui-mysql-workbench.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-gui-navicat.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-gui-datagrip.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-gui-dbeaver.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-gui-vscode-sqltools.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-gui-mysql-workbench.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-gui-navicat.md"]' \
  -F 'metadata={"topic_name": "database_tools_integration", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Java Development (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-java-jdbc.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-java-mybatis.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-java-hibernate.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-java-spring-boot.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-java-jdbc.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-java-mybatis.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-java-hibernate.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-java-spring-boot.md"]' \
  -F 'metadata={"topic_name": "java_development", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Python Development - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-mysqlclient.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-mysql-connector.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-pymysql.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-sqlalchemy.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-peewee.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-mysqlclient.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-mysql-connector.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-pymysql.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-sqlalchemy.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-peewee.md"]' \
  -F 'metadata={"topic_name": "python_development", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Python Development - Part 2 (1 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-python-django.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-python-django.md"]' \
  -F 'metadata={"topic_name": "python_development", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Nodejs Development (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nodejs-mysql2.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nodejs-mysqljs.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nodejs-prisma.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nodejs-sequelize.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-sample-application-nodejs-typeorm.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nodejs-mysql2.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nodejs-mysqljs.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nodejs-prisma.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nodejs-sequelize.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-sample-application-nodejs-typeorm.md"]' \
  -F 'metadata={"topic_name": "nodejs_development", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Database Design (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-schema-design-overview.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-create-database.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-create-table.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-create-secondary-indexes.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-schema-design-overview.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-create-database.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-create-table.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-create-secondary-indexes.md"]' \
  -F 'metadata={"topic_name": "database_design", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Data Operations (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-insert-data.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-update-data.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-delete-data.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-get-data-from-single-table.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-insert-data.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-update-data.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-delete-data.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-get-data-from-single-table.md"]' \
  -F 'metadata={"topic_name": "data_operations", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Performance Optimization (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-optimize-sql-overview.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-optimize-sql.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-index-best-practice.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-optimize-sql-overview.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-optimize-sql.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-index-best-practice.md"]' \
  -F 'metadata={"topic_name": "performance_optimization", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-htap-quickstart.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/scalability-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-high-availability.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/high-availability-with-multi-az.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-htap-quickstart.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/scalability-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-high-availability.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/high-availability-with-multi-az.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitoring-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-streaming-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/backup-and-restore-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/security-concepts.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/dev-guide-bi-looker-studio.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/monitoring-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-streaming-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/backup-and-restore-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/security-concepts.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/dev-guide-bi-looker-studio.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/dev-guide-wordpress.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-driver.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-driver-node-example.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-driver-prisma-example.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-driver-kysely-example.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/dev-guide-wordpress.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-driver.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-driver-node-example.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-driver-prisma-example.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-driver-kysely-example.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-driver-drizzle-example.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/select-cluster-tier.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/size-your-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-performance-reference.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/create-tidb-cluster-serverless.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-driver-drizzle-example.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/select-cluster-tier.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/size-your-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-performance-reference.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/create-tidb-cluster-serverless.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/connect-via-standard-connection-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/branch-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/branch-manage.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/branch-github-integration.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/manage-serverless-spend-limit.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/connect-via-standard-connection-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/branch-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/branch-manage.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/branch-github-integration.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/manage-serverless-spend-limit.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-export.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/create-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connect-via-standard-connection.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/set-up-private-endpoint-connections-on-azure.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/set-up-private-endpoint-connections-on-google-cloud.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-export.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/create-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connect-via-standard-connection.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/set-up-private-endpoint-connections-on-azure.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/set-up-private-endpoint-connections-on-google-cloud.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 7 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/set-up-vpc-peering-connections.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connect-via-sql-shell.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/scale-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/pause-or-resume-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/configure-maintenance-window.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/set-up-vpc-peering-connections.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connect-via-sql-shell.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/scale-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/pause-or-resume-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/configure-maintenance-window.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 8 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-alert-email.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-alert-slack.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-alert-zoom.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-events.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/third-party-monitoring-integrations.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-alert-email.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-alert-slack.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-alert-zoom.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-events.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/third-party-monitoring-integrations.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 9 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-clinic.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-sql-tuning-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-node-group-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-node-group-management.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/upgrade-tidb-cluster.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-clinic.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-sql-tuning-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-node-group-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-node-group-management.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/upgrade-tidb-cluster.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 10 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/delete-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-incremental-data-from-mysql-using-data-migration.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-sql-shards.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-from-mysql-using-aws-dms.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-from-oracle-using-aws-dms.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/delete-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-incremental-data-from-mysql-using-data-migration.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-sql-shards.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-from-mysql-using-aws-dms.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-from-oracle-using-aws-dms.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 11 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-sample-data-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-import-local-files.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-csv-files-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-parquet-files-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-with-mysql-cli-serverless.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/import-sample-data-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-import-local-files.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-csv-files-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-parquet-files-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-with-mysql-cli-serverless.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 12 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/dedicated-external-storage.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-external-storage.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/naming-conventions-for-data-import.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/csv-config-for-import-data.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/troubleshoot-import-access-denied-error.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/dedicated-external-storage.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-external-storage.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/naming-conventions-for-data-import.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/csv-config-for-import-data.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/troubleshoot-import-access-denied-error.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 13 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-dm-precheck-and-troubleshooting.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-connect-aws-dms.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/explore-data-with-chat2query.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/sql-proxy-account.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/vector-search-integrate-with-amazon-bedrock.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-dm-precheck-and-troubleshooting.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-connect-aws-dms.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/explore-data-with-chat2query.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/sql-proxy-account.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/vector-search-integrate-with-amazon-bedrock.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 14 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/vector-search-full-text-search-sql.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/vector-search-full-text-search-python.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/vector-search-hybrid-search.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/vector-search-changelogs.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/use-chat2query-sessions.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/vector-search-full-text-search-sql.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/vector-search-full-text-search-python.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/vector-search-hybrid-search.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/vector-search-changelogs.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/use-chat2query-sessions.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 15 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/use-chat2query-knowledge.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-manage-data-app.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-manage-endpoint.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-api-key.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-custom-domain.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/use-chat2query-knowledge.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-manage-data-app.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-manage-endpoint.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-api-key.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-custom-domain.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 16 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-integrations.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-postman-integration.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-manage-github-connection.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-oas-with-nextjs.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-app-config-files.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-integrations.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-postman-integration.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-manage-github-connection.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-oas-with-nextjs.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-app-config-files.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 17 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-response-and-status-code.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-sink-to-apache-pulsar.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-sink-to-tidb-cloud.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-sink-to-cloud-storage.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/setup-aws-self-hosted-kafka-private-link-service.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-response-and-status-code.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-sink-to-apache-pulsar.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-sink-to-tidb-cloud.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-sink-to-cloud-storage.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/setup-aws-self-hosted-kafka-private-link-service.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 18 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/setup-azure-self-hosted-kafka-private-link-service.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/setup-self-hosted-kafka-private-service-connect.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/recovery-group-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/recovery-group-get-started.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/recovery-group-failover.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/setup-azure-self-hosted-kafka-private-link-service.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/setup-self-hosted-kafka-private-service-connect.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/recovery-group-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/recovery-group-get-started.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/recovery-group-failover.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 19 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/recovery-group-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-org-sso-authentication.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/oauth2.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/configure-serverless-firewall-rules-for-public-endpoints.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/secure-connections-to-serverless-clusters.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/recovery-group-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-org-sso-authentication.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/oauth2.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/configure-serverless-firewall-rules-for-public-endpoints.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/secure-connections-to-serverless-clusters.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 20 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/configure-ip-access-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-tls-connect-to-dedicated.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-encrypt-cmek.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-log-redaction.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-auditing.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/configure-ip-access-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-tls-connect-to-dedicated.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-encrypt-cmek.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-log-redaction.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-auditing.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 21 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-audit-logging.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-console-auditing.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-billing-ticdc-rcu.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-billing-dm.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-billing-recovery-group.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-audit-logging.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-console-auditing.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-billing-ticdc-rcu.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-billing-dm.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-billing-recovery-group.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 22 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-budget.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-airbyte.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-aws-lambda.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-cloudflare.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-datadog-integration.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-budget.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-airbyte.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-aws-lambda.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-cloudflare.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-datadog-integration.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 23 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-dbt.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-n8n.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-netlify.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-new-relic-integration.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-prometheus-and-grafana-integration.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-dbt.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-n8n.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-netlify.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-new-relic-integration.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-prometheus-and-grafana-integration.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 24 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-tidbcloud-provider-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-get-tidbcloud-provider.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-dedicated-cluster-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-dedicated-private-endpoint-connection-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-dedicated-vpc-peering-resource.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-tidbcloud-provider-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-get-tidbcloud-provider.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-dedicated-cluster-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-dedicated-private-endpoint-connection-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-dedicated-vpc-peering-resource.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 25 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-dedicated-network-container-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-serverless-cluster-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-serverless-branch-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-serverless-export-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-sql-user-resource.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-dedicated-network-container-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-serverless-cluster-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-serverless-branch-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-serverless-export-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-sql-user-resource.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 26 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-cluster-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-backup-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-restore-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-use-import-resource.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/terraform-migrate-cluster-resource.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-cluster-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-backup-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-restore-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-use-import-resource.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/terraform-migrate-cluster-resource.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 27 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-vercel.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/integrate-tidbcloud-with-zapier.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/api-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/cli-reference.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auth-login.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-vercel.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/integrate-tidbcloud-with-zapier.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/api-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/cli-reference.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auth-login.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 28 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auth-logout.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auth-whoami.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-cluster-create.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-cluster-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-cluster-describe.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auth-logout.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auth-whoami.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-cluster-create.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-cluster-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-cluster-describe.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 29 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-cluster-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-update.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-spending-limit.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-region.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-shell.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-cluster-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-update.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-spending-limit.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-region.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-shell.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 30 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-branch-create.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-branch-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-branch-describe.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-branch-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-branch-shell.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-branch-create.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-branch-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-branch-describe.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-branch-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-branch-shell.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 31 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-import-cancel.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-import-describe.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-import-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-import-start.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-export-create.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-import-cancel.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-import-describe.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-import-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-import-start.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-export-create.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 32 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-export-describe.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-export-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-export-cancel.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-export-download.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-sql-user-create.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-export-describe.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-export-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-export-cancel.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-export-download.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-sql-user-create.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 33 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-sql-user-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-sql-user-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-sql-user-update.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-config.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-describe.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-sql-user-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-sql-user-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-sql-user-update.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-config.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-describe.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 34 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-download.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-create.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-describe.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-list.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-download.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-create.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-describe.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-list.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 35 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-template.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-auditlog-filter-update.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-authorized-network-create.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-authorized-network-delete.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-authorized-network-list.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-template.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-auditlog-filter-update.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-authorized-network-create.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-authorized-network-delete.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-authorized-network-list.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 36 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-serverless-authorized-network-update.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-ai.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-completion.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-create.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-delete.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-serverless-authorized-network-update.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-ai.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-completion.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-create.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-delete.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 37 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-describe.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-edit.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-set.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-config-use.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-describe.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-edit.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-set.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-config-use.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 38 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-project-list.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-upgrade.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/ticloud-help.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-partners.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/managed-service-provider-customer.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-project-list.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-upgrade.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/ticloud-help.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-partners.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/managed-service-provider-customer.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 39 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/cppo-customer.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/limitations-and-quotas.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-limitations.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/limited-sql-features.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v8.5-performance-highlights.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/cppo-customer.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/limitations-and-quotas.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-limitations.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/limited-sql-features.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v8.5-performance-highlights.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 40 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/v8.5-performance-benchmarking-with-tpcc.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v8.5-performance-benchmarking-with-sysbench.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v8.1-performance-benchmarking-with-tpcc.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v8.1-performance-benchmarking-with-sysbench.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v7.5-performance-benchmarking-with-tpcc.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/v8.5-performance-benchmarking-with-tpcc.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v8.5-performance-benchmarking-with-sysbench.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v8.1-performance-benchmarking-with-tpcc.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v8.1-performance-benchmarking-with-sysbench.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v7.5-performance-benchmarking-with-tpcc.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 41 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/v7.5-performance-benchmarking-with-sysbench.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v7.1-performance-benchmarking-with-tpcc.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v7.1-performance-benchmarking-with-sysbench.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v6.5-performance-benchmarking-with-tpcc.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/v6.5-performance-benchmarking-with-sysbench.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/v7.5-performance-benchmarking-with-sysbench.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v7.1-performance-benchmarking-with-tpcc.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v7.1-performance-benchmarking-with-sysbench.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v6.5-performance-benchmarking-with-tpcc.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/v6.5-performance-benchmarking-with-sysbench.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 42 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/notifications.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-glossary.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-care-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-care-detail.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-ai-chat-in-im.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/notifications.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-glossary.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-care-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-care-detail.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-ai-chat-in-im.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 43 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-slack-ticket-creation.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-lark-ticket-creation.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-slack-ticket-interaction.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connected-lark-ticket-interaction.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-support.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-slack-ticket-creation.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-lark-ticket-creation.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-slack-ticket-interaction.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connected-lark-ticket-interaction.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-support.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 44 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-faq.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/serverless-faqs.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-release-notes.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/release-notes-2024.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/release-notes-2023.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-faq.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/serverless-faqs.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-release-notes.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/release-notes-2024.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/release-notes-2023.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Cluster Management - Part 45 (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/release-notes-2022.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/release-notes-2021.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/release-notes-2020.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/release-notes-2022.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/release-notes-2021.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/release-notes-2020.md"]' \
  -F 'metadata={"topic_name": "cluster_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Connection Management (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/connect-to-tidb-cluster-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/set-up-private-endpoint-connections-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/connect-to-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/set-up-private-endpoint-connections.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/connect-to-tidb-cluster-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/set-up-private-endpoint-connections-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/connect-to-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/set-up-private-endpoint-connections.md"]' \
  -F 'metadata={"topic_name": "connection_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Backup Restore (2 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/backup-and-restore-serverless.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/backup-and-restore.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/backup-and-restore-serverless.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/backup-and-restore.md"]' \
  -F 'metadata={"topic_name": "backup_restore", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Monitoring Alerts (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-tidb-cluster.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/built-in-monitoring.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/monitor-built-in-alerting.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-tidb-cluster.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/built-in-monitoring.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/monitor-built-in-alerting.md"]' \
  -F 'metadata={"topic_name": "monitoring_alerts", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Performance Tuning (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-tune-performance-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tune-performance.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/index-insight.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-tune-performance-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tune-performance.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/index-insight.md"]' \
  -F 'metadata={"topic_name": "performance_tuning", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Data Migration (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-migration-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-from-mysql-using-data-migration.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/migrate-from-op-tidb.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-migration-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-from-mysql-using-data-migration.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/migrate-from-op-tidb.md"]' \
  -F 'metadata={"topic_name": "data_migration", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Data Import (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-sample-data.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-csv-files.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-parquet-files.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/import-with-mysql-cli.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/import-sample-data.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-csv-files.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-parquet-files.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/import-with-mysql-cli.md"]' \
  -F 'metadata={"topic_name": "data_import", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tiftlash Analytics (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-overview.md" \
  -F "files=@$DOCS_PATH/tiflash/create-tiflash-replicas.md" \
  -F "files=@$DOCS_PATH/tiflash/use-tidb-to-read-tiflash.md" \
  -F "files=@$DOCS_PATH/tiflash/use-tiflash-mpp-mode.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tiflash/tiflash-overview.md", "https://github.com/pingcap/docs/blob/master/tiflash/create-tiflash-replicas.md", "https://github.com/pingcap/docs/blob/master/tiflash/use-tidb-to-read-tiflash.md", "https://github.com/pingcap/docs/blob/master/tiflash/use-tiflash-mpp-mode.md"]' \
  -F 'metadata={"topic_name": "tiftlash_analytics", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Vector Search Guide (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-get-started-using-sql.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-functions-and-operators.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-index.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/vector-search/vector-search-get-started-using-sql.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-functions-and-operators.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-index.md"]' \
  -F 'metadata={"topic_name": "vector_search_guide", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Vector Search Integrations (2 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-llamaindex.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-langchain.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-llamaindex.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-langchain.md"]' \
  -F 'metadata={"topic_name": "vector_search_integrations", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Data Service Api (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/data-service-get-started.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/use-chat2query-api.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/data-service-get-started.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/use-chat2query-api.md"]' \
  -F 'metadata={"topic_name": "data_service_api", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Changefeed Streaming (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-overview.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-sink-to-mysql.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/changefeed-sink-to-apache-kafka.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-overview.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-sink-to-mysql.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/changefeed-sink-to-apache-kafka.md"]' \
  -F 'metadata={"topic_name": "changefeed_streaming", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Security Authentication (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-password-authentication.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/tidb-cloud-sso-authentication.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/manage-user-access.md" \
  -F "files=@$DOCS_PATH/tidb-cloud/configure-security-settings.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-password-authentication.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/tidb-cloud-sso-authentication.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/manage-user-access.md", "https://github.com/pingcap/docs/blob/master/tidb-cloud/configure-security-settings.md"]' \
  -F 'metadata={"topic_name": "security_authentication", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-overview.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-alter-ddl.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-cancel-ddl.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-checksum-table.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-overview.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-alter-ddl.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-cancel-ddl.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-checksum-table.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-cleanup.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-pause-ddl.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-recover.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-admin-resume-ddl.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-database.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-cleanup.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-pause-ddl.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-recover.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-admin-resume-ddl.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-database.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-instance.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-placement-policy.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-range.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-resource-group.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-sequence.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-instance.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-placement-policy.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-range.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-resource-group.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-sequence.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-add-column.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-add-index.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-index.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-change-column.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-add-column.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-add-index.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-index.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-change-column.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-table-compact.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-column.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-index.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-modify-column.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-rename-index.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-table-compact.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-column.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-index.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-modify-column.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-rename-index.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-alter-user.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-analyze-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-backup.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-batch.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-begin.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-alter-user.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-analyze-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-backup.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-batch.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-begin.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 7 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-cancel-import-job.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-commit.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-database.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-index.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-placement-policy.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-cancel-import-job.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-commit.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-database.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-index.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-placement-policy.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 8 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-resource-group.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-sequence.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-table-like.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-table.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-resource-group.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-sequence.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-table-like.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-table.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 9 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-user.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-create-view.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-deallocate.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-delete.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-desc.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-user.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-create-view.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-deallocate.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-delete.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-desc.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 10 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-describe.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-do.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-database.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-placement-policy.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-resource-group.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-describe.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-do.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-database.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-placement-policy.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-resource-group.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 11 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-sequence.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-stats.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-user.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-sequence.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-stats.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-user.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 12 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-drop-view.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-execute.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-explain-analyze.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-explain.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flashback-cluster.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-drop-view.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-execute.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-explain-analyze.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-explain.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flashback-cluster.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 13 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flashback-database.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flashback-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flush-privileges.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flush-status.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-flush-tables.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flashback-database.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flashback-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flush-privileges.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flush-status.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-flush-tables.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 14 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-grant-privileges.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-grant-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-import-into.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-insert.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-load-data.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-grant-privileges.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-grant-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-import-into.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-insert.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-load-data.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 15 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-load-stats.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-lock-stats.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-lock-tables-and-unlock-tables.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-prepare.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-query-watch.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-load-stats.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-lock-stats.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-lock-tables-and-unlock-tables.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-prepare.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-query-watch.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 16 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-recover-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-rename-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-rename-user.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-replace.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-restore.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-recover-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-rename-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-rename-user.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-replace.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-restore.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 17 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-revoke-privileges.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-revoke-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-rollback.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-savepoint.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-select.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-revoke-privileges.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-revoke-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-rollback.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-savepoint.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-select.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 18 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-set-default-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-set-password.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-set-resource-group.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-set-role.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-set-transaction.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-set-default-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-set-password.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-set-resource-group.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-set-role.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-set-transaction.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 19 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-analyze-status.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-builtins.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-character-set.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-collation.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-column-stats-usage.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-analyze-status.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-builtins.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-character-set.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-collation.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-column-stats-usage.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 20 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-columns-from.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-database.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-placement-policy.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-resource-group.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-sequence.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-columns-from.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-database.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-placement-policy.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-resource-group.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-sequence.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 21 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-create-user.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-databases.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-engines.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-errors.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-create-user.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-databases.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-engines.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-errors.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 22 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-fields-from.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-grants.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-import-job.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-master-status.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-placement.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-fields-from.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-grants.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-import-job.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-master-status.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-placement.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 23 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-placement-for.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-placement-labels.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-plugins.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-privileges.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-processlist.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-placement-for.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-placement-labels.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-plugins.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-privileges.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-processlist.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 24 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-profiles.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-schemas.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-buckets.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-healthy.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-histograms.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-profiles.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-schemas.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-buckets.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-healthy.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-histograms.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 25 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-locked.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-meta.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-stats-topn.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-status.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-table-next-rowid.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-locked.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-meta.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-stats-topn.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-status.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-table-next-rowid.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 26 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-table-regions.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-table-status.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-tables.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-show-warnings.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-split-region.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-table-regions.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-table-status.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-tables.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-show-warnings.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-split-region.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 27 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-start-transaction.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-table.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-trace.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-truncate.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-unlock-stats.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-start-transaction.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-table.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-trace.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-truncate.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-unlock-stats.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Statements Reference - Part 28 (3 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-update.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-use.md" \
  -F "files=@$DOCS_PATH/sql-statements/sql-statement-with.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-update.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-use.md", "https://github.com/pingcap/docs/blob/master/sql-statements/sql-statement-with.md"]' \
  -F 'metadata={"topic_name": "sql_statements_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Data Types - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/data-type-overview.md" \
  -F "files=@$DOCS_PATH/data-type-default-values.md" \
  -F "files=@$DOCS_PATH/data-type-numeric.md" \
  -F "files=@$DOCS_PATH/data-type-date-and-time.md" \
  -F "files=@$DOCS_PATH/data-type-string.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/data-type-overview.md", "https://github.com/pingcap/docs/blob/master/data-type-default-values.md", "https://github.com/pingcap/docs/blob/master/data-type-numeric.md", "https://github.com/pingcap/docs/blob/master/data-type-date-and-time.md", "https://github.com/pingcap/docs/blob/master/data-type-string.md"]' \
  -F 'metadata={"topic_name": "sql_data_types", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Data Types - Part 2 (1 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/data-type-json.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/data-type-json.md"]' \
  -F 'metadata={"topic_name": "sql_data_types", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/functions-and-operators-overview.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/type-conversion-in-expression-evaluation.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/operators.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/control-flow-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/string-functions.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/functions-and-operators-overview.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/type-conversion-in-expression-evaluation.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/operators.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/control-flow-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/string-functions.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/numeric-functions-and-operators.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/date-and-time-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/bit-functions-and-operators.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/cast-functions-and-operators.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/encryption-and-compression-functions.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/numeric-functions-and-operators.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/date-and-time-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/bit-functions-and-operators.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/cast-functions-and-operators.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/encryption-and-compression-functions.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/locking-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/information-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-create.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-search.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/locking-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/information-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-create.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-search.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-modify.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-return.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-utility.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-aggregate.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/json-functions/json-functions-validate.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-modify.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-return.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-utility.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-aggregate.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/json-functions/json-functions-validate.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/aggregate-group-by-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/group-by-modifier.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/window-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/sequence-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/utility-functions.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/aggregate-group-by-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/group-by-modifier.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/window-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/sequence-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/utility-functions.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Sql Functions Operators - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/functions-and-operators/miscellaneous-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/tidb-functions.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/precision-math.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/set-operators.md" \
  -F "files=@$DOCS_PATH/functions-and-operators/expressions-pushed-down.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/functions-and-operators/miscellaneous-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/tidb-functions.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/precision-math.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/set-operators.md", "https://github.com/pingcap/docs/blob/master/functions-and-operators/expressions-pushed-down.md"]' \
  -F 'metadata={"topic_name": "sql_functions_operators", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/time-to-live.md" \
  -F "files=@$DOCS_PATH/tiflash/use-fastscan.md" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-supported-pushdown-calculations.md" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-results-materialization.md" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-late-materialization.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/time-to-live.md", "https://github.com/pingcap/docs/blob/master/tiflash/use-fastscan.md", "https://github.com/pingcap/docs/blob/master/tiflash/tiflash-supported-pushdown-calculations.md", "https://github.com/pingcap/docs/blob/master/tiflash/tiflash-results-materialization.md", "https://github.com/pingcap/docs/blob/master/tiflash/tiflash-late-materialization.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-compatibility.md" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-pipeline-model.md" \
  -F "files=@$DOCS_PATH/statement-summary-tables.md" \
  -F "files=@$DOCS_PATH/explain-overview.md" \
  -F "files=@$DOCS_PATH/explain-walkthrough.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tiflash/tiflash-compatibility.md", "https://github.com/pingcap/docs/blob/master/tiflash/tiflash-pipeline-model.md", "https://github.com/pingcap/docs/blob/master/statement-summary-tables.md", "https://github.com/pingcap/docs/blob/master/explain-overview.md", "https://github.com/pingcap/docs/blob/master/explain-walkthrough.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/explain-indexes.md" \
  -F "files=@$DOCS_PATH/explain-joins.md" \
  -F "files=@$DOCS_PATH/explain-mpp.md" \
  -F "files=@$DOCS_PATH/explain-subqueries.md" \
  -F "files=@$DOCS_PATH/explain-aggregation.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/explain-indexes.md", "https://github.com/pingcap/docs/blob/master/explain-joins.md", "https://github.com/pingcap/docs/blob/master/explain-mpp.md", "https://github.com/pingcap/docs/blob/master/explain-subqueries.md", "https://github.com/pingcap/docs/blob/master/explain-aggregation.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/explain-views.md" \
  -F "files=@$DOCS_PATH/explain-partitions.md" \
  -F "files=@$DOCS_PATH/explain-index-merge.md" \
  -F "files=@$DOCS_PATH/sql-optimization-concepts.md" \
  -F "files=@$DOCS_PATH/sql-logical-optimization.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/explain-views.md", "https://github.com/pingcap/docs/blob/master/explain-partitions.md", "https://github.com/pingcap/docs/blob/master/explain-index-merge.md", "https://github.com/pingcap/docs/blob/master/sql-optimization-concepts.md", "https://github.com/pingcap/docs/blob/master/sql-logical-optimization.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/subquery-optimization.md" \
  -F "files=@$DOCS_PATH/column-pruning.md" \
  -F "files=@$DOCS_PATH/correlated-subquery-optimization.md" \
  -F "files=@$DOCS_PATH/max-min-eliminate.md" \
  -F "files=@$DOCS_PATH/predicate-push-down.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/subquery-optimization.md", "https://github.com/pingcap/docs/blob/master/column-pruning.md", "https://github.com/pingcap/docs/blob/master/correlated-subquery-optimization.md", "https://github.com/pingcap/docs/blob/master/max-min-eliminate.md", "https://github.com/pingcap/docs/blob/master/predicate-push-down.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/partition-pruning.md" \
  -F "files=@$DOCS_PATH/topn-limit-push-down.md" \
  -F "files=@$DOCS_PATH/join-reorder.md" \
  -F "files=@$DOCS_PATH/derive-topn-from-window.md" \
  -F "files=@$DOCS_PATH/sql-physical-optimization.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/partition-pruning.md", "https://github.com/pingcap/docs/blob/master/topn-limit-push-down.md", "https://github.com/pingcap/docs/blob/master/join-reorder.md", "https://github.com/pingcap/docs/blob/master/derive-topn-from-window.md", "https://github.com/pingcap/docs/blob/master/sql-physical-optimization.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 7 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/choose-index.md" \
  -F "files=@$DOCS_PATH/statistics.md" \
  -F "files=@$DOCS_PATH/extended-statistics.md" \
  -F "files=@$DOCS_PATH/wrong-index-solution.md" \
  -F "files=@$DOCS_PATH/agg-distinct-optimization.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/choose-index.md", "https://github.com/pingcap/docs/blob/master/statistics.md", "https://github.com/pingcap/docs/blob/master/extended-statistics.md", "https://github.com/pingcap/docs/blob/master/wrong-index-solution.md", "https://github.com/pingcap/docs/blob/master/agg-distinct-optimization.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 8 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/cost-model.md" \
  -F "files=@$DOCS_PATH/runtime-filter.md" \
  -F "files=@$DOCS_PATH/sql-prepared-plan-cache.md" \
  -F "files=@$DOCS_PATH/sql-non-prepared-plan-cache.md" \
  -F "files=@$DOCS_PATH/control-execution-plan.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/cost-model.md", "https://github.com/pingcap/docs/blob/master/runtime-filter.md", "https://github.com/pingcap/docs/blob/master/sql-prepared-plan-cache.md", "https://github.com/pingcap/docs/blob/master/sql-non-prepared-plan-cache.md", "https://github.com/pingcap/docs/blob/master/control-execution-plan.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 9 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/optimizer-hints.md" \
  -F "files=@$DOCS_PATH/sql-plan-management.md" \
  -F "files=@$DOCS_PATH/blocklist-control-plan.md" \
  -F "files=@$DOCS_PATH/optimizer-fix-controls.md" \
  -F "files=@$DOCS_PATH/index-advisor.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/optimizer-hints.md", "https://github.com/pingcap/docs/blob/master/sql-plan-management.md", "https://github.com/pingcap/docs/blob/master/blocklist-control-plan.md", "https://github.com/pingcap/docs/blob/master/optimizer-fix-controls.md", "https://github.com/pingcap/docs/blob/master/index-advisor.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 10 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/follower-read.md" \
  -F "files=@$DOCS_PATH/coprocessor-cache.md" \
  -F "files=@$DOCS_PATH/garbage-collection-overview.md" \
  -F "files=@$DOCS_PATH/garbage-collection-configuration.md" \
  -F "files=@$DOCS_PATH/tiflash/tune-tiflash-performance.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/follower-read.md", "https://github.com/pingcap/docs/blob/master/coprocessor-cache.md", "https://github.com/pingcap/docs/blob/master/garbage-collection-overview.md", "https://github.com/pingcap/docs/blob/master/garbage-collection-configuration.md", "https://github.com/pingcap/docs/blob/master/tiflash/tune-tiflash-performance.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 11 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-resource-control-ru-groups.md" \
  -F "files=@$DOCS_PATH/tidb-resource-control-runaway-queries.md" \
  -F "files=@$DOCS_PATH/tidb-resource-control-background-tasks.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integration-overview.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-jinaai-embedding.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-resource-control-ru-groups.md", "https://github.com/pingcap/docs/blob/master/tidb-resource-control-runaway-queries.md", "https://github.com/pingcap/docs/blob/master/tidb-resource-control-background-tasks.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integration-overview.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-jinaai-embedding.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 12 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-sqlalchemy.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-peewee.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-integrate-with-django-orm.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-data-types.md" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-improve-performance.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-sqlalchemy.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-peewee.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-integrate-with-django-orm.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-data-types.md", "https://github.com/pingcap/docs/blob/master/vector-search/vector-search-improve-performance.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 13 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/vector-search/vector-search-limitations.md" \
  -F "files=@$DOCS_PATH/basic-sql-operations.md" \
  -F "files=@$DOCS_PATH/auto-increment.md" \
  -F "files=@$DOCS_PATH/auto-random.md" \
  -F "files=@$DOCS_PATH/shard-row-id-bits.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/vector-search/vector-search-limitations.md", "https://github.com/pingcap/docs/blob/master/basic-sql-operations.md", "https://github.com/pingcap/docs/blob/master/auto-increment.md", "https://github.com/pingcap/docs/blob/master/auto-random.md", "https://github.com/pingcap/docs/blob/master/shard-row-id-bits.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 14 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/literal-values.md" \
  -F "files=@$DOCS_PATH/schema-object-names.md" \
  -F "files=@$DOCS_PATH/keywords.md" \
  -F "files=@$DOCS_PATH/user-defined-variables.md" \
  -F "files=@$DOCS_PATH/expression-syntax.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/literal-values.md", "https://github.com/pingcap/docs/blob/master/schema-object-names.md", "https://github.com/pingcap/docs/blob/master/keywords.md", "https://github.com/pingcap/docs/blob/master/user-defined-variables.md", "https://github.com/pingcap/docs/blob/master/expression-syntax.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 15 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/comment-syntax.md" \
  -F "files=@$DOCS_PATH/clustered-indexes.md" \
  -F "files=@$DOCS_PATH/constraints.md" \
  -F "files=@$DOCS_PATH/generated-columns.md" \
  -F "files=@$DOCS_PATH/sql-mode.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/comment-syntax.md", "https://github.com/pingcap/docs/blob/master/clustered-indexes.md", "https://github.com/pingcap/docs/blob/master/constraints.md", "https://github.com/pingcap/docs/blob/master/generated-columns.md", "https://github.com/pingcap/docs/blob/master/sql-mode.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 16 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/table-attributes.md" \
  -F "files=@$DOCS_PATH/pipelined-dml.md" \
  -F "files=@$DOCS_PATH/views.md" \
  -F "files=@$DOCS_PATH/partitioned-table.md" \
  -F "files=@$DOCS_PATH/temporary-tables.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/table-attributes.md", "https://github.com/pingcap/docs/blob/master/pipelined-dml.md", "https://github.com/pingcap/docs/blob/master/views.md", "https://github.com/pingcap/docs/blob/master/partitioned-table.md", "https://github.com/pingcap/docs/blob/master/temporary-tables.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 17 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/cached-tables.md" \
  -F "files=@$DOCS_PATH/foreign-key.md" \
  -F "files=@$DOCS_PATH/character-set-and-collation.md" \
  -F "files=@$DOCS_PATH/character-set-gbk.md" \
  -F "files=@$DOCS_PATH/stale-read.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/cached-tables.md", "https://github.com/pingcap/docs/blob/master/foreign-key.md", "https://github.com/pingcap/docs/blob/master/character-set-and-collation.md", "https://github.com/pingcap/docs/blob/master/character-set-gbk.md", "https://github.com/pingcap/docs/blob/master/stale-read.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 18 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/as-of-timestamp.md" \
  -F "files=@$DOCS_PATH/tidb-read-staleness.md" \
  -F "files=@$DOCS_PATH/tidb-external-ts.md" \
  -F "files=@$DOCS_PATH/read-historical-data.md" \
  -F "files=@$DOCS_PATH/placement-rules-in-sql.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/as-of-timestamp.md", "https://github.com/pingcap/docs/blob/master/tidb-read-staleness.md", "https://github.com/pingcap/docs/blob/master/tidb-external-ts.md", "https://github.com/pingcap/docs/blob/master/read-historical-data.md", "https://github.com/pingcap/docs/blob/master/placement-rules-in-sql.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 19 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/sys-schema/sys-schema.md" \
  -F "files=@$DOCS_PATH/sys-schema/sys-schema-unused-indexes.md" \
  -F "files=@$DOCS_PATH/metadata-lock.md" \
  -F "files=@$DOCS_PATH/best-practices/uuid.md" \
  -F "files=@$DOCS_PATH/accelerated-table-creation.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/sys-schema/sys-schema.md", "https://github.com/pingcap/docs/blob/master/sys-schema/sys-schema-unused-indexes.md", "https://github.com/pingcap/docs/blob/master/metadata-lock.md", "https://github.com/pingcap/docs/blob/master/best-practices/uuid.md", "https://github.com/pingcap/docs/blob/master/accelerated-table-creation.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 20 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/schema-cache.md" \
  -F "files=@$DOCS_PATH/tidb-architecture.md" \
  -F "files=@$DOCS_PATH/tidb-storage.md" \
  -F "files=@$DOCS_PATH/tidb-computing.md" \
  -F "files=@$DOCS_PATH/tidb-scheduling.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/schema-cache.md", "https://github.com/pingcap/docs/blob/master/tidb-architecture.md", "https://github.com/pingcap/docs/blob/master/tidb-storage.md", "https://github.com/pingcap/docs/blob/master/tidb-computing.md", "https://github.com/pingcap/docs/blob/master/tidb-scheduling.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 21 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tso.md" \
  -F "files=@$DOCS_PATH/tikv-overview.md" \
  -F "files=@$DOCS_PATH/storage-engine/rocksdb-overview.md" \
  -F "files=@$DOCS_PATH/tiflash/tiflash-spill-disk.md" \
  -F "files=@$DOCS_PATH/tidb-distributed-execution-framework.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tso.md", "https://github.com/pingcap/docs/blob/master/tikv-overview.md", "https://github.com/pingcap/docs/blob/master/storage-engine/rocksdb-overview.md", "https://github.com/pingcap/docs/blob/master/tiflash/tiflash-spill-disk.md", "https://github.com/pingcap/docs/blob/master/tidb-distributed-execution-framework.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 22 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/tidb-global-sort.md" \
  -F "files=@$DOCS_PATH/tidb-limitations.md" \
  -F "files=@$DOCS_PATH/system-variables.md" \
  -F "files=@$DOCS_PATH/status-variables.md" \
  -F "files=@$DOCS_PATH/table-filter.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/tidb-global-sort.md", "https://github.com/pingcap/docs/blob/master/tidb-limitations.md", "https://github.com/pingcap/docs/blob/master/system-variables.md", "https://github.com/pingcap/docs/blob/master/status-variables.md", "https://github.com/pingcap/docs/blob/master/table-filter.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Tidb Architecture - Part 23 (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/external-storage-uri.md" \
  -F "files=@$DOCS_PATH/ddl-introduction.md" \
  -F "files=@$DOCS_PATH/batch-processing.md" \
  -F "files=@$DOCS_PATH/troubleshoot-data-inconsistency-errors.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/external-storage-uri.md", "https://github.com/pingcap/docs/blob/master/ddl-introduction.md", "https://github.com/pingcap/docs/blob/master/batch-processing.md", "https://github.com/pingcap/docs/blob/master/troubleshoot-data-inconsistency-errors.md"]' \
  -F 'metadata={"topic_name": "tidb_architecture", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Transaction Management - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/develop/dev-guide-transaction-overview.md" \
  -F "files=@$DOCS_PATH/develop/dev-guide-optimistic-and-pessimistic-transaction.md" \
  -F "files=@$DOCS_PATH/transaction-overview.md" \
  -F "files=@$DOCS_PATH/transaction-isolation-levels.md" \
  -F "files=@$DOCS_PATH/optimistic-transaction.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/develop/dev-guide-transaction-overview.md", "https://github.com/pingcap/docs/blob/master/develop/dev-guide-optimistic-and-pessimistic-transaction.md", "https://github.com/pingcap/docs/blob/master/transaction-overview.md", "https://github.com/pingcap/docs/blob/master/transaction-isolation-levels.md", "https://github.com/pingcap/docs/blob/master/optimistic-transaction.md"]' \
  -F 'metadata={"topic_name": "transaction_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# Transaction Management - Part 2 (2 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/pessimistic-transaction.md" \
  -F "files=@$DOCS_PATH/non-transactional-dml.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/pessimistic-transaction.md", "https://github.com/pingcap/docs/blob/master/non-transactional-dml.md"]' \
  -F 'metadata={"topic_name": "transaction_management", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 1 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/mysql-schema/mysql-schema.md" \
  -F "files=@$DOCS_PATH/mysql-schema/mysql-schema-tidb-mdl-view.md" \
  -F "files=@$DOCS_PATH/mysql-schema/mysql-schema-user.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-analyze-status.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/mysql-schema/mysql-schema.md", "https://github.com/pingcap/docs/blob/master/mysql-schema/mysql-schema-tidb-mdl-view.md", "https://github.com/pingcap/docs/blob/master/mysql-schema/mysql-schema-user.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-analyze-status.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 2 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-check-constraints.md" \
  -F "files=@$DOCS_PATH/information-schema/client-errors-summary-by-host.md" \
  -F "files=@$DOCS_PATH/information-schema/client-errors-summary-by-user.md" \
  -F "files=@$DOCS_PATH/information-schema/client-errors-summary-global.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-character-sets.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-check-constraints.md", "https://github.com/pingcap/docs/blob/master/information-schema/client-errors-summary-by-host.md", "https://github.com/pingcap/docs/blob/master/information-schema/client-errors-summary-by-user.md", "https://github.com/pingcap/docs/blob/master/information-schema/client-errors-summary-global.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-character-sets.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 3 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-cluster-info.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-collations.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-collation-character-set-applicability.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-columns.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-data-lock-waits.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-cluster-info.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-collations.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-collation-character-set-applicability.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-columns.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-data-lock-waits.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 4 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-ddl-jobs.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-deadlocks.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-engines.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-keywords.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-key-column-usage.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-ddl-jobs.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-deadlocks.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-engines.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-keywords.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-key-column-usage.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 5 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-memory-usage.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-memory-usage-ops-history.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-partitions.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-placement-policies.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-processlist.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-memory-usage.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-memory-usage-ops-history.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-partitions.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-placement-policies.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-processlist.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 6 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-referential-constraints.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-resource-groups.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-runaway-watches.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-schemata.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-sequences.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-referential-constraints.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-resource-groups.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-runaway-watches.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-schemata.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-sequences.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 7 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-session-variables.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-slow-query.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-statistics.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tables.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-table-constraints.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-session-variables.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-slow-query.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-statistics.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tables.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-table-constraints.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 8 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-table-storage-stats.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-check-constraints.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-hot-regions-history.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-indexes.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-index-usage.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-table-storage-stats.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-check-constraints.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-hot-regions-history.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-indexes.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-index-usage.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 9 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-servers-info.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tidb-trx.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tiflash-replica.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tiflash-segments.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tiflash-tables.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-servers-info.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tidb-trx.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tiflash-replica.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tiflash-segments.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tiflash-tables.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 10 (5 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tikv-region-peers.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tikv-region-status.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-tikv-store-status.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-user-attributes.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-user-privileges.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tikv-region-peers.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tikv-region-status.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-tikv-store-status.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-user-attributes.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-user-privileges.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"

# System Tables Reference - Part 11 (4 files)
curl -X POST "http://localhost:8000/api/v1/save" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-variables-info.md" \
  -F "files=@$DOCS_PATH/information-schema/information-schema-views.md" \
  -F "files=@$DOCS_PATH/performance-schema/performance-schema.md" \
  -F "files=@$DOCS_PATH/performance-schema/performance-schema-session-connect-attrs.md" \
  -F 'links=["https://github.com/pingcap/docs/blob/master/information-schema/information-schema-variables-info.md", "https://github.com/pingcap/docs/blob/master/information-schema/information-schema-views.md", "https://github.com/pingcap/docs/blob/master/performance-schema/performance-schema.md", "https://github.com/pingcap/docs/blob/master/performance-schema/performance-schema-session-connect-attrs.md"]' \
  -F 'metadata={"topic_name": "system_tables_reference", "force_regenerate": "False"}' \
  -F "target_type=knowledge_graph"


# Usage:
# ./upload_tidb_cloud_docs.sh
# ./upload_tidb_cloud_docs.sh /custom/path/to/docs
