"""
Background daemon for processing pending graph build tasks.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from setting.db import SessionLocal, db_manager
from knowledge_graph.models import GraphBuildStatus, SourceData
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
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
        max_retries: int = 3,
    ):
        """
        Initialize the graph build daemon.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            check_interval: Interval in seconds to check for pending tasks
            max_retries: Maximum number of retries for failed tasks
        """
        self.llm_client = llm_client or LLMInterface("openai_like", "qwen3-32b")
        self.embedding_func = embedding_func or get_text_embedding
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.is_running = False

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

        topic_name = task_info["topic_name"]
        source_ids = task_info["source_ids"]
        external_database_uri = task_info["external_database_uri"]

        logger.info(
            f"Processing earliest topic: {topic_name} with {len(source_ids)} sources (database: {"external" if external_database_uri else 'local'})"
        )

        try:
            self._process_task(task_info, external_database_uri)
        except Exception as e:
            error_message = f"Graph build failed: {str(e)}"
            logger.error(
                f"Failed to build graph for topic {topic_name}: {e}", exc_info=True
            )

            # Update tasks to failed status
            self._update_final_status(
                topic_name, source_ids, external_database_uri, "failed", error_message
            )

    def _prepare_task_build(self) -> Optional[Dict]:
        """
        Prepare task build by finding earliest task, updating status, and getting source data.

        Returns:
            Dict with task info or None if no pending tasks
        """
        with SessionLocal() as db:
            # Find the earliest pending task
            earliest_task = self._get_earliest_pending_task(db)

            if not earliest_task:
                return None

            topic_name = earliest_task.topic_name
            external_database_uri = earliest_task.external_database_uri

            # Get all pending tasks for this topic and database combination
            topic_tasks = self._get_pending_tasks_for_topic_and_db(
                db, topic_name, external_database_uri
            )
            source_ids = [task.source_id for task in topic_tasks]

            # Update all tasks to processing status in local database
            self._update_task_status(
                db, topic_name, source_ids, external_database_uri, "processing"
            )

            # Fetch source data from appropriate database
            if db_manager.is_local_mode(external_database_uri):
                # Get data from local database
                topic_docs = self._create_source_list(db, source_ids)
            else:
                # Get data from user database
                user_session_factory = db_manager.get_session_factory(
                    external_database_uri
                )
                with user_session_factory() as user_db:
                    topic_docs = self._create_source_list(user_db, source_ids)

            if not topic_docs:
                logger.warning(f"No valid sources found for topic: {topic_name}")
                # Use _update_final_status to update both local and external databases
                self._update_final_status(
                    topic_name,
                    source_ids,
                    external_database_uri,
                    "failed",
                    "No valid sources found",
                )
                return None

            return {
                "topic_name": topic_name,
                "source_ids": source_ids,
                "external_database_uri": external_database_uri,
                "topic_docs": topic_docs,
            }

    def _get_earliest_pending_task(self, db: Session) -> Optional[GraphBuildStatus]:
        """
        Get the earliest pending task across all databases.

        Args:
            db: Database session

        Returns:
            Earliest pending GraphBuildStatus record, or None if no pending tasks
        """
        earliest_task = (
            db.query(GraphBuildStatus)
            .filter(
                or_(
                    GraphBuildStatus.status == "pending",
                    GraphBuildStatus.status == "processing",
                )
            )
            .order_by(GraphBuildStatus.scheduled_at.asc())
            .first()
        )

        return earliest_task

    def _get_pending_tasks_for_topic_and_db(
        self, db: Session, topic_name: str, external_database_uri: str
    ) -> List[GraphBuildStatus]:
        """
        Get all pending tasks for a specific topic and database combination.

        Args:
            db: Database session
            topic_name: Name of the topic
            external_database_uri: External database URI

        Returns:
            List of pending GraphBuildStatus records for the topic and database
        """
        return (
            db.query(GraphBuildStatus)
            .filter(
                and_(
                    or_(
                        GraphBuildStatus.status == "pending",
                        GraphBuildStatus.status == "processing",
                    ),
                    GraphBuildStatus.topic_name == topic_name,
                    GraphBuildStatus.external_database_uri == external_database_uri,
                )
            )
            .order_by(GraphBuildStatus.scheduled_at.asc())
            .all()
        )

    def _update_task_status(
        self,
        db: Session,
        topic_name: str,
        source_ids: List[str],
        external_database_uri: str,
        status: str,
        error_message: Optional[str] = None,
    ):
        """
        Update the status of graph build tasks in local database.

        Args:
            db: Database session
            topic_name: Name of the topic
            source_ids: List of source IDs to update
            external_database_uri: External database URI
            status: New status to set
            error_message: Error message if status is 'failed'
        """
        try:
            update_data = {"status": status, "updated_at": func.current_timestamp()}

            if error_message:
                update_data["error_message"] = error_message

            db.query(GraphBuildStatus).filter(
                and_(
                    GraphBuildStatus.topic_name == topic_name,
                    GraphBuildStatus.source_id.in_(source_ids),
                    GraphBuildStatus.external_database_uri == external_database_uri,
                )
            ).update(update_data, synchronize_session=False)

            db.commit()
            logger.info(
                f"Updated {len(source_ids)} local database tasks to status: {status}"
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
        source_ids: List[str],
        external_database_uri: str,
        status: str,
        error_message: Optional[str] = None,
    ):
        """
        Update the final status of graph build tasks using a new database session.
        This is used after the long-running build_knowledge_graph operation.

        Args:
            topic_name: Name of the topic
            source_ids: List of source IDs to update
            external_database_uri: External database URI
            status: New status to set
            error_message: Error message if status is 'failed'
        """
        try:
            # Update local database
            with SessionLocal() as db:
                update_data = {"status": status, "updated_at": func.current_timestamp()}

                if error_message:
                    update_data["error_message"] = error_message

                db.query(GraphBuildStatus).filter(
                    and_(
                        GraphBuildStatus.topic_name == topic_name,
                        GraphBuildStatus.source_id.in_(source_ids),
                        GraphBuildStatus.external_database_uri == external_database_uri,
                    )
                ).update(update_data, synchronize_session=False)

                db.commit()
                logger.info(
                    f"Updated {len(source_ids)} local database tasks to final status: {status} (local database)"
                )

            if db_manager.is_local_mode(external_database_uri):
                return

            user_session_factory = db_manager.get_session_factory(external_database_uri)
            with user_session_factory() as db:
                update_data = {"status": status, "updated_at": func.current_timestamp()}

                if error_message:
                    update_data["error_message"] = error_message

                # In external database, records have empty external_database_uri
                db.query(GraphBuildStatus).filter(
                    and_(
                        GraphBuildStatus.topic_name == topic_name,
                        GraphBuildStatus.source_id.in_(source_ids),
                        GraphBuildStatus.external_database_uri
                        == "",  # External DB records have empty URI
                    )
                ).update(update_data, synchronize_session=False)

                db.commit()
                logger.info(
                    f"Updated {len(source_ids)} external database tasks to final status: {status}"
                )

        except Exception as e:
            logger.error(f"Failed to update final task status: {e}", exc_info=True)
            # Don't raise here as the graph building itself might have succeeded

    def _create_source_list(self, db: Session, source_ids: List[str]) -> List[Dict]:
        """
        Create a list of source documents for graph building.

        Args:
            db: Database session
            source_ids: List of source IDs to fetch

        Returns:
            List of source document dictionaries
        """
        sources = db.query(SourceData).filter(SourceData.id.in_(source_ids)).all()

        topic_docs = []
        for source in sources:
            if source.content:  # Only include sources with content
                topic_docs.append(
                    {
                        "source_id": source.id,
                        "source_name": source.name,
                        "source_content": source.content,
                        "source_link": source.link,
                    }
                )
            else:
                logger.warning(
                    f"Source {source.id} ({source.name}) has no content, skipping"
                )

        return topic_docs

    def get_daemon_status(self) -> Dict:
        """
        Get current daemon status and statistics.

        Returns:
            Dictionary with daemon status information
        """
        with SessionLocal() as db:
            pending_count = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "pending")
                .count()
            )

            processing_count = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "processing")
                .count()
            )

            completed_count = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "completed")
                .count()
            )

            failed_count = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "failed")
                .count()
            )

        return {
            "is_running": self.is_running,
            "check_interval": self.check_interval,
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "total_tasks": pending_count
            + processing_count
            + completed_count
            + failed_count,
        }

    def _process_task(self, task_info: Dict, external_database_uri: str):
        """Process a local database task."""
        topic_name = task_info["topic_name"]
        source_ids = task_info["source_ids"]
        external_database_uri = task_info["external_database_uri"]
        topic_docs = task_info["topic_docs"]

        if db_manager.is_local_mode(external_database_uri):
            session_factory = SessionLocal
            logger.info(f"Starting local graph build for topic: {topic_name}")
        else:
            session_factory = db_manager.get_session_factory(external_database_uri)
            logger.info(f"Starting external graph build for topic: {topic_name}")

        graph_builder = KnowledgeGraphBuilder(
            self.llm_client, self.embedding_func, session_factory
        )
        result = graph_builder.build_knowledge_graph(topic_name, topic_docs)

        # Update final status
        self._update_final_status(
            topic_name, source_ids, external_database_uri, "completed"
        )

        logger.info(f"Successfully completed graph build for topic: {topic_name}")
        logger.info(f"Build results: {result}")
