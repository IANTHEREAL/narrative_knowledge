"""
Background daemon for processing pending graph build tasks.
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from setting.db import SessionLocal, db_manager
from knowledge_graph.models import GraphBuild, SourceData
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from etl.knowledge import KnowledgeBuilder
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding

logger = logging.getLogger(__name__)


class GraphBuildDaemon:
    """
    Background daemon for processing pending graph build tasks.
    """

    def __init__(
        self,
        llm_client: Optional[LLMInterface] = None,
        embedding_func=None,
        check_interval: int = 60,
        worker_count: int = 5,
    ):
        """
        Initialize the graph build daemon.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            check_interval: Interval in seconds to check for pending tasks
        """
        self.llm_client = llm_client or LLMInterface("openai_like", "qwen3-32b")
        self.embedding_func = embedding_func or get_text_embedding
        self.check_interval = check_interval
        self.is_running = False
        self.worker_count = worker_count

    def start(self):
        """Start the daemon."""
        self.is_running = True
        logger.info("Graph build daemon started")

        while self.is_running:
            try:
                self._process_pending_tasks()
            except Exception as e:
                logger.error(f"Error in daemon main loop: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def stop(self):
        """Stop the daemon."""
        self.is_running = False
        logger.info("Graph build daemon stopped")

    def _process_pending_tasks(self):
        """Process the earliest pending graph build topic."""
        # Step 1: Find earliest task and prepare data (inside db session)
        task_info = self._prepare_task_build()
        if not task_info:
            return

        # Step 2: Process the build graph
        logger.info(f"Start to process the build graph: {task_info["topic_name"]}")
        topic_name = task_info["topic_name"]
        external_database_uri = task_info["external_database_uri"]
        task_data = task_info["task_data"]
        build_ids = [task["build_id"] for task in task_data]
        logger.info(
            f"Processing earliest topic: {topic_name} with {len(build_ids)} sources (database: {'external' if external_database_uri else 'local'})"
        )
        self._process_task(task_info, external_database_uri)

    def _prepare_task_build(self) -> Optional[Dict]:
        """
        Prepare task build by finding earliest task, updating status, and loading document data from storage.

        Returns:
            Dict with task info or None if no pending tasks
        """
        with SessionLocal() as db:
            # Find the earliest pending task
            earliest_task = self._get_earliest_pending_task(db)

            if not earliest_task:
                return None

            logger.info(f"Start to collect earliest task: {earliest_task}")

            topic_name = earliest_task.topic_name
            external_database_uri = earliest_task.external_database_uri

            # Get all pending tasks for this topic and database combination
            topic_tasks = self._get_pending_tasks_for_topic_and_db(
                db, topic_name, external_database_uri
            )

            # Update all tasks to processing status in local database
            try:
                task_data = []
                for task in topic_tasks:
                    if task.storage_directory.startswith("memory://"):
                        task_data.append(
                            {
                                "build_id": task.build_id,
                                "storage_directory": "",
                                "document_file": task.doc_link,
                                "metadata": {
                                    "doc_link": task.doc_link,
                                    "topic_name": task.topic_name,
                                }
                            }
                        )
                        continue

                    if not task.storage_directory:
                        logger.error(
                            f"Task {task.build_id} has no storage directory, marking as failed"
                        )
                        raise Exception(
                            f"Task {task.build_id} has no storage directory {task.storage_directory}"
                        )

                    # Verify storage directory exists and contains required files
                    storage_path = Path(task.storage_directory)
                    if not storage_path.exists():
                        logger.error(f"Storage directory not found: {storage_path}")
                        raise Exception(f"Storage directory not found: {storage_path}")

                    # Load document metadata
                    metadata_file = storage_path / "document_metadata.json"
                    if not metadata_file.exists():
                        logger.error(f"Metadata file not found: {metadata_file}")
                        raise Exception(f"Metadata file not found: {metadata_file}")

                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading metadata file: {e}")
                        raise Exception(f"Error loading metadata file: {e}")

                    # Get the expected document filename from metadata
                    expected_filename = metadata.get("file_name")
                    if not expected_filename:
                        logger.error(f"No file_name found in metadata: {metadata_file}")
                        raise Exception(
                            f"No file_name found in metadata: {metadata_file}"
                        )

                    # Find the specific document file by name
                    doc_file = storage_path / expected_filename
                    if not doc_file.exists():
                        logger.error(f"Expected document file not found: {doc_file}")
                        raise Exception(f"Expected document file not found: {doc_file}")

                    task_data.append(
                        {
                            "build_id": task.build_id,
                            "storage_directory": str(storage_path),
                            "document_file": str(doc_file),
                            "metadata": metadata,
                        }
                    )

                if not task_data:
                    logger.warning(f"No valid tasks found for topic: {topic_name}")
                    return None

                # Update remaining valid tasks to processing status
                valid_build_ids = [task["build_id"] for task in task_data]
                self._update_task_status(
                    db,
                    topic_name,
                    valid_build_ids,
                    external_database_uri,
                    "processing",
                )
            except Exception as e:
                logger.error(f"Error in _prepare_task_build: {e}", exc_info=True)
                valid_build_ids = [task.build_id for task in topic_tasks]
                self._update_final_status(
                    topic_name,
                    valid_build_ids,
                    external_database_uri,
                    "failed",
                    str(e),
                )
                raise

            return {
                "topic_name": topic_name,
                "external_database_uri": external_database_uri,
                "task_data": task_data,
            }

    def _get_earliest_pending_task(self, db: Session) -> Optional[GraphBuild]:
        """
        Get the earliest pending task across all databases.

        Args:
            db: Database session

        Returns:
            Earliest pending GraphBuild record, or None if no pending tasks
        """
        earliest_task = (
            db.query(GraphBuild)
            .filter(GraphBuild.status == "pending")
            .order_by(GraphBuild.scheduled_at.asc())
            .first()
        )

        return earliest_task

    def _get_pending_tasks_for_topic_and_db(
        self, db: Session, topic_name: str, external_database_uri: str
    ) -> List[GraphBuild]:
        """
        Get all pending tasks for a specific topic and database combination.

        Args:
            db: Database session
            topic_name: Name of the topic
            external_database_uri: External database URI

        Returns:
            List of pending GraphBuild records for the topic and database
        """
        return (
            db.query(GraphBuild)
            .filter(
                and_(
                    GraphBuild.status == "pending",
                    GraphBuild.topic_name == topic_name,
                    GraphBuild.external_database_uri == external_database_uri,
                )
            )
            .order_by(GraphBuild.scheduled_at.asc())
            .all()
        )

    def _update_task_status(
        self,
        db: Session,
        topic_name: str,
        build_ids: List[str],
        external_database_uri: str,
        status: str,
        error_message: Optional[str] = None,
    ):
        """
        Update the status of graph build tasks in local database.

        Args:
            db: Database session
            topic_name: Name of the topic
            build_ids: List of temp token IDs to update
            external_database_uri: External database URI
            status: New status to set
            error_message: Error message if status is 'failed'
        """
        try:
            update_data = {"status": status, "updated_at": func.current_timestamp()}

            if error_message:
                update_data["error_message"] = error_message

            db.query(GraphBuild).filter(
                and_(
                    GraphBuild.topic_name == topic_name,
                    GraphBuild.build_id.in_(build_ids),
                    GraphBuild.external_database_uri == external_database_uri,
                )
            ).update(update_data, synchronize_session=False)

            db.commit()
            logger.info(
                f"Updated {len(build_ids)} local database tasks to status: {status}"
            )

        except Exception as e:
            db.rollback()
            logger.error(
                f"Failed to update local database task status: {e}", exc_info=True
            )
            raise

    def _update_final_status(
        self,
        topic_name: str,
        build_ids: List[str],
        external_database_uri: str,
        status: str,
        error_message: Optional[str] = None,
    ):
        """
        Update the final status of graph build tasks in local database.
        All task scheduling is centralized in local database.

        Args:
            topic_name: Name of the topic
            build_ids: List of temp token IDs to update
            external_database_uri: External database URI (for filtering)
            status: New status to set
            error_message: Error message if status is 'failed'
        """
        try:
            # Update local database only (all tasks are stored here)
            with SessionLocal() as db:
                update_data = {"status": status, "updated_at": func.current_timestamp()}

                if error_message:
                    update_data["error_message"] = error_message

                db.query(GraphBuild).filter(
                    and_(
                        GraphBuild.topic_name == topic_name,
                        GraphBuild.build_id.in_(build_ids),
                        GraphBuild.external_database_uri == external_database_uri,
                    )
                ).update(update_data, synchronize_session=False)

                db.commit()
                logger.info(
                    f"Updated {len(build_ids)} tasks to final status: {status} (topic: {topic_name})"
                )

        except Exception as e:
            logger.error(f"Failed to update final task status: {e}", exc_info=True)
            # Don't raise here as the graph building itself might have succeeded

    def get_daemon_status(self) -> Dict:
        """
        Get current daemon status and statistics.

        Returns:
            Dictionary with daemon status information
        """
        with SessionLocal() as db:
            uploaded_count = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "uploaded")
                .count()
            )

            pending_count = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "pending")
                .count()
            )

            processing_count = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "processing")
                .count()
            )

            completed_count = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "completed")
                .count()
            )

            failed_count = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "failed")
                .count()
            )

        return {
            "is_running": self.is_running,
            "check_interval": self.check_interval,
            "uploaded_tasks": uploaded_count,
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "total_tasks": uploaded_count
            + pending_count
            + processing_count
            + completed_count
            + failed_count,
            "note": "Daemon only processes tasks with 'pending' status. Use trigger-processing API to move 'uploaded' tasks to 'pending'.",
        }

    def _process_task(self, task_info: Dict, external_database_uri: str):
        """
        Process tasks from storage directories.
        First performs knowledge extraction, then graph building.
        """
        topic_name = task_info["topic_name"]
        external_database_uri = task_info["external_database_uri"]
        task_data = task_info["task_data"]

        if db_manager.is_local_mode(external_database_uri):
            session_factory = SessionLocal
            logger.info(
                f"Starting local knowledge extraction and graph build for topic: {topic_name}"
            )
        else:
            session_factory = db_manager.get_session_factory(external_database_uri)
            logger.info(
                f"Starting external knowledge extraction and graph build for topic: {topic_name}"
            )

        # Step 1: Extract knowledge from documents
        extracted_sources = []
        failed_extractions = []

        kb_builder = KnowledgeBuilder(
            self.llm_client, self.embedding_func, session_factory=session_factory
        )

        for task in task_data:
            try:
                logger.info(f"Extracting knowledge from: {task['document_file']}")

                # Prepare attributes for knowledge extraction
                metadata = task["metadata"]
                attributes = {
                    "doc_link": metadata.get("doc_link", ""),
                    "topic_name": metadata.get("topic_name", topic_name),
                }

                # Add custom metadata to attributes if present
                custom_metadata = metadata.get("custom_metadata")
                if custom_metadata and isinstance(custom_metadata, dict):
                    attributes.update(custom_metadata)

                result = kb_builder.extract_knowledge(task["document_file"], attributes)

                if result["status"] != "success":
                    error_msg = f"Knowledge extraction failed: {result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    failed_extractions.append(
                        {"build_id": task["build_id"], "error": error_msg}
                    )
                    continue

                # Add to successful extractions using the consistent build_id
                extracted_sources.append(
                    {
                        "build_id": task[
                            "build_id"
                        ],  # Keep build_id for status update
                        "source_id": result["source_id"],
                        "source_name": result["source_name"],
                        "source_content": result["source_content"],
                        "source_link": result["source_link"],
                        "source_attributes": result["source_attributes"],
                    }
                )

                logger.info(
                    f"Successfully extracted knowledge from {task['document_file']} with build_id: {task['build_id']}"
                )

            except Exception as e:
                error_msg = f"Failed to extract knowledge from {task['document_file']}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                failed_extractions.append(
                    {"build_id": task["build_id"], "error": error_msg}
                )

        # Handle failed extractions
        if failed_extractions:
            error_messages = "; ".join([item["error"] for item in failed_extractions])
            raise Exception(f"Knowledge extraction failed: {error_messages}")

        # Continue with graph building if we have successful extractions
        if not extracted_sources:
            error_msg = (
                "No documents were successfully processed for knowledge extraction"
            )
            raise Exception(error_msg)

        # Step 2: Build knowledge graph from extracted sources
        try:
            logger.info(
                f"Building knowledge graph for topic: {topic_name} with {len(extracted_sources)} sources"
            )

            graph_builder = KnowledgeGraphBuilder(
                self.llm_client, self.embedding_func, session_factory, self.worker_count
            )
            result = graph_builder.build_knowledge_graph(topic_name, extracted_sources)

            # Update successful tasks to completed status
            # Use the consistent build_ids from successful extractions
            completed_build_ids = [
                source["build_id"] for source in extracted_sources
            ]

            self._update_final_status(
                topic_name, completed_build_ids, external_database_uri, "completed"
            )

            logger.info(
                f"Successfully completed knowledge extraction and graph build for topic: {topic_name}"
            )
            logger.info(f"Build results: {result}")

        except Exception as e:
            error_msg = f"Graph building failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            failed_build_ids = [
                source["build_id"] for source in extracted_sources
            ]
            self._update_final_status(
                topic_name,
                failed_build_ids,
                external_database_uri,
                "failed",
                error_msg,
            )
