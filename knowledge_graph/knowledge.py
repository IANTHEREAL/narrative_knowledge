from pathlib import Path
import logging
import hashlib
from typing import Dict, Any

from knowledge_graph.models import SourceData, GraphBuildStatus
from setting.db import SessionLocal
from sqlalchemy import and_
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
    ):
        # Extract basic info of source
        doc_link = attributes.get("doc_link", None)
        if doc_link is None or doc_link == "":
            doc_link = source_path

        with self.SessionLocal() as db:
            # Check if source data already exists by doc_link
            existing_source = (
                db.query(SourceData).filter(SourceData.link == doc_link).first()
            )

            if existing_source:
                logger.info(
                    f"Source data already exists for {source_path} (matched by link), reusing existing id: {existing_source.id}"
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
            # Create SourceData with pre-set ID if provided
            source_data_kwargs = {
                "name": name,
                "content": full_content,
                "link": doc_link,
                "source_type": source_type,
                "hash": source_hash,
                "attributes": attributes,
            }

            source_data = SourceData(**source_data_kwargs)

            db.add(source_data)
            db.commit()
            db.refresh(source_data)
            logger.info(f"Source data created for {source_path}, id: {source_data.id}")

            return {
                "status": "success",
                "source_id": source_data.id,
                "source_path": source_path,
                "source_content": source_data.content,
                "source_link": source_data.link,
                "source_name": source_data.name,
                "source_type": source_type,
            }
