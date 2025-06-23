from pathlib import Path
import logging
import hashlib
from typing import Dict, Any

from knowledge_graph.models import SourceData, ContentStore, GraphBuildStatus
from setting.db import SessionLocal
from sqlalchemy import and_
from etl.extract import extract_source_data

logger = logging.getLogger(__name__)


def _get_content_type_from_path(source_path: str) -> str:
    """Get content MIME type from file extension"""
    extension = Path(source_path).suffix.lower()
    type_mapping = {
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".sql": "text/sql",
        ".py": "text/plain",
        ".txt": "text/plain",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }
    return type_mapping.get(extension, "application/octet-stream")


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
                    "source_content": existing_source.effective_content,
                    "source_link": existing_source.link,
                    "source_name": existing_source.name,
                    "source_attributes": existing_source.attributes,
                }

        # Read raw file content first for hash calculation
        with open(source_path, "rb") as f:
            raw_content = f.read()
            content_hash = hashlib.sha256(raw_content).hexdigest()

        # Initialize variables
        extracted_content = None
        content_type = _get_content_type_from_path(source_path)

        with self.SessionLocal() as db:
            # Check if content already exists
            content_store = (
                db.query(ContentStore).filter_by(content_hash=content_hash).first()
            )

            if not content_store:
                # New content - need to extract
                try:
                    source_info = extract_source_data(source_path)
                except Exception as e:
                    logger.error(f"Failed to process {source_path}: {e}")
                    raise RuntimeError(f"Failed to process {source_path}: {e}")

                extracted_content = source_info.get("content", None)

                content_store = ContentStore(
                    content_hash=content_hash,
                    content=extracted_content,
                    content_size=len(raw_content),
                    content_type=content_type,
                    name=Path(source_path).stem,
                    link=doc_link,
                )
                db.add(content_store)
                logger.info(
                    f"Created new content store entry with hash: {content_hash[:8]}..."
                )
            else:
                # Content already exists, get the extracted content from content_store
                extracted_content = content_store.content
                logger.info(
                    f"Reusing existing content store entry with hash: {content_hash[:8]}..."
                )

            source_data = SourceData(
                name=Path(source_path).stem,
                content=extracted_content,  # Keep for backward compatibility
                link=doc_link,
                source_type=content_type,
                content_hash=content_store.content_hash,
                attributes=attributes,
            )

            db.add(source_data)
            db.commit()
            db.refresh(source_data)
            logger.info(f"Source data created for {source_path}, id: {source_data.id}")

            return {
                "status": "success",
                "source_id": source_data.id,
                "source_path": source_path,
                "source_content": source_data.effective_content,
                "source_link": source_data.link,
                "source_name": source_data.name,
                "source_type": content_type,
                "source_attributes": attributes,
            }
