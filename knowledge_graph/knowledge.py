from pathlib import Path
import logging
import hashlib
from typing import Dict, Any

from knowledge_graph.models import SourceData, GraphBuildStatus
from setting.db import SessionLocal
from etl.extract import extract_source_data

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """
    A builder class for constructing knowledge graphs from documents.
    """

    def __init__(self, session_factory=None):
        """
        Initialize the builder with a graph instance and specifications.

        Args:
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.SessionLocal = session_factory or SessionLocal

    def extract_knowledge(
        self,
        source_path: str,
        attributes: Dict[str, Any],
        source_id: str = None,
        **kwargs,
    ):
        # Extract basic info of source
        doc_link = attributes.get("doc_link", None)
        if doc_link is None or doc_link == "":
            doc_link = source_path

        with self.SessionLocal() as db:
            # Check if source data already exists by hash or doc_link
            existing_source = (
                db.query(SourceData).filter((SourceData.link == doc_link)).first()
            )

            if existing_source:
                logger.info(
                    f"Source data already exists for {source_path} (matched by link), id: {existing_source.id}"
                )

                return {
                    "status": "success",
                    "source_id": existing_source.id,
                    "source_type": existing_source.source_type,
                    "source_path": source_path,
                    "source_content": existing_source.content,
                    "source_link": existing_source.link,
                    "source_name": existing_source.name,
                }

        try:
            source_info = extract_source_data(source_path)
        except Exception as e:
            logger.error(f"Failed to process {source_path}: {e}")
            raise RuntimeError(f"Failed to process{source_path}: {e}")

        full_content = source_info.get("content", None)
        source_type = source_info.get("file_type", "document")

        name = Path(source_path).stem
        source_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()

        with self.SessionLocal() as db:
            # Check if source data already exists by hash
            existing_source = (
                db.query(SourceData).filter(SourceData.hash == source_hash).first()
            )

            if existing_source:
                logger.info(
                    f"Source data already exists for {source_path}, id: {existing_source.id}"
                )

                return {
                    "status": "success",
                    "source_id": existing_source.id,
                    "source_type": existing_source.source_type,
                    "source_path": source_path,
                    "source_content": existing_source.content,
                    "source_link": existing_source.link,
                    "source_name": existing_source.name,
                }
            else:
                # Create SourceData with pre-set ID if provided
                source_data_kwargs = {
                    "name": name,
                    "content": full_content,
                    "link": doc_link,
                    "source_type": source_type,
                    "hash": source_hash,
                    "attributes": attributes,
                }

                # Use pre-set source_id if provided (for consistency with task tracking)
                if source_id:
                    source_data_kwargs["id"] = source_id

                source_data = SourceData(**source_data_kwargs)

                db.add(source_data)
                db.commit()
                db.refresh(source_data)
                logger.info(
                    f"Source data created for {source_path}, id: {source_data.id}"
                )

                return {
                    "status": "success",
                    "source_id": source_data.id,
                    "source_path": source_path,
                    "source_content": source_data.content,
                    "source_link": source_data.link,
                    "source_name": source_data.name,
                    "source_type": source_type,
                }

    def create_build_status_record(
        self, source_id: str, topic_name: str, external_database_uri: str = ""
    ) -> None:
        """
        Create a GraphBuildStatus record for the uploaded document.

        Args:
            source_id: The source document ID
            topic_name: The topic name for graph building
            external_database_uri: External database URI for multi-database mode

        Raises:
            Exception: If database operation fails
        """
        try:
            with self.SessionLocal() as db:
                # Check if record already exists
                existing_status = (
                    db.query(GraphBuildStatus)
                    .filter(
                        GraphBuildStatus.topic_name == topic_name,
                        GraphBuildStatus.source_id == source_id,
                        GraphBuildStatus.external_database_uri == external_database_uri,
                    )
                    .first()
                )

                if not existing_status:
                    # Create new build status record
                    build_status = GraphBuildStatus(
                        topic_name=topic_name,
                        source_id=source_id,
                        external_database_uri=external_database_uri,
                        status="pending",
                    )
                    db.add(build_status)
                    db.commit()
                    logger.info(
                        f"Created build status record for source {source_id} in topic {topic_name} (external_db: {'external' if external_database_uri else 'local'})"
                    )
                else:
                    logger.info(
                        f"Build status record already exists for source {source_id} in topic {topic_name} (external_db: {'external' if external_database_uri else 'local'})"
                    )

        except Exception as e:
            logger.error(f"Failed to create build status record: {e}")
            raise
