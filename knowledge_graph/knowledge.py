from pathlib import Path
import logging
import hashlib
from typing import Dict, Any, List

from knowledge_graph.models import (
    SourceData,
    ContentStore,
    GraphBuildStatus,
    KnowledgeBlock,
    BlockSourceMapping,
)
from knowledge_graph.parser.factory import get_parser_by_content_type
from knowledge_graph.parser.base import Block
from knowledge_graph.situate_context import gen_situate_context
from setting.db import SessionLocal
from etl.extract import extract_source_data
from utils.token import encode_text, decode_tokens
from llm.factory import LLMInterface

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

    def __init__(
        self, llm_client: LLMInterface = None, embedding_func=None, session_factory=None
    ):
        """
        Initialize the builder with a graph instance and specifications.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.embedding_func = embedding_func
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

    def split_knowledge_blocks(self, source_id: str, **kwargs) -> List[KnowledgeBlock]:
        """
        Extract knowledge blocks from existing source data.

        Args:
            source_id: ID of the SourceData to extract blocks from
            **kwargs: Additional parameters for parser

        Returns:
            List of created KnowledgeBlock instances
        """
        with self.SessionLocal() as db:
            # Get source data by ID
            source_data = (
                db.query(SourceData).filter(SourceData.id == source_id).first()
            )
            if not source_data:
                raise ValueError(f"SourceData with id {source_id} not found")

            # Check if knowledge blocks already exist for this source
            existing_blocks = (
                db.query(KnowledgeBlock)
                .join(BlockSourceMapping)
                .filter(BlockSourceMapping.source_id == source_id)
                .all()
            )

            if existing_blocks:
                logger.info(
                    f"Found {len(existing_blocks)} existing knowledge blocks for source {source_id}"
                )
                return existing_blocks

            full_content = source_data.effective_content
            # Get full content from source
            if not full_content:
                logger.warning(f"No content found for source {source_id}")
                return []

        # Get suitable parser based on source_type
        if not self.llm_client:
            raise ValueError("LLM client is required for knowledge block extraction")

        try:
            # Select parser based on source_type
            parser = get_parser_by_content_type(
                source_data.source_type, self.llm_client
            )
            # Use the new parse_content method directly
            doc_knowledge = parser.parse_content(
                content=full_content, name=source_data.name, **kwargs
            )
            blocks = doc_knowledge.blocks
            name = doc_knowledge.name

        except Exception as e:
            logger.error(f"Failed to parse content for source {source_id}: {e}")
            # Fallback: create a single block with all content
            blocks = [Block(name=source_data.name, content=full_content, position=1)]
            name = source_data.name

        logger.info(f"Split {len(blocks)} blocks for source {source_id}")

        if not blocks or len(blocks) == 0:
            blocks = [Block(name=name, content=full_content, position=1)]

        # Check token limits for each block
        for block in blocks:
            content = block.content
            tokens = encode_text(content)
            if len(tokens) > 4096:
                logger.warning(
                    f"Section '{block.name}' has {len(tokens)} tokens, exceeding 4096. Consider restructuring."
                )

        # Generate situated context for each section
        section_context = {}
        for block in blocks:
            logger.info(f"Generating context for block {block.name}")
            try:
                section_context[block.name] = gen_situate_context(
                    self.llm_client, full_content, block.content
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate context for block {block.name}: {e}"
                )
                section_context[block.name] = None

        # Process each knowledge block with hash-based deduplication
        created_blocks = []

        with self.SessionLocal() as db:
            # Re-fetch source data in this transaction
            source_data = (
                db.query(SourceData).filter(SourceData.id == source_id).first()
            )

            for block in blocks:
                context = section_context.get(block.name, None)
                content_str = block.content

                # Generate hash for knowledge block
                kb_hash_input = f"{block.name}|{content_str}|{context or ''}"
                kb_hash = hashlib.sha256(kb_hash_input.encode("utf-8")).hexdigest()

                # Check if knowledge block already exists by hash
                existing_kb = (
                    db.query(KnowledgeBlock)
                    .filter(KnowledgeBlock.hash == kb_hash)
                    .first()
                )

                if existing_kb:
                    logger.info(f"Knowledge block already exists: {block.name}")
                    knowledge_block = existing_kb
                else:
                    # Generate embedding based on context + block content
                    if context:
                        embedding_input = (
                            f"<context>\n{context}</context>\n\n{content_str}"
                        )
                    else:
                        embedding_input = content_str

                    # Generate embedding if embedding function is available
                    content_vec = None
                    if self.embedding_func:
                        content_vec = self.embedding_func(embedding_input)

                    # Create new knowledge block
                    knowledge_block = KnowledgeBlock(
                        name=block.name,
                        context=context,
                        content=content_str,
                        knowledge_type="paragraph",
                        content_vec=content_vec,
                        hash=kb_hash,
                        attributes={"position": block.position},
                    )
                    db.add(knowledge_block)
                    db.flush()  # Flush to get the ID
                    logger.info(f"Knowledge block created: {block.name}")

                created_blocks.append(
                    {
                        "id": knowledge_block.id,
                        "name": knowledge_block.name,
                        "content": knowledge_block.content,
                        "context": knowledge_block.context,
                        "hash": knowledge_block.hash,
                        "attributes": knowledge_block.attributes,
                    }
                )

                # Always establish mapping between knowledge block and source data
                existing_mapping = (
                    db.query(BlockSourceMapping)
                    .filter(
                        BlockSourceMapping.block_id == knowledge_block.id,
                        BlockSourceMapping.source_id == source_data.id,
                    )
                    .first()
                )

                if not existing_mapping:
                    mapping = BlockSourceMapping(
                        block_id=knowledge_block.id,
                        source_id=source_data.id,
                        position_in_source=block.position,
                    )
                    db.add(mapping)
                    logger.info(
                        f"Created mapping: knowledge_block({knowledge_block.id}) -> source_data({source_data.id})"
                    )
                else:
                    logger.info(
                        f"Mapping already exists: knowledge_block({knowledge_block.id}) -> source_data({source_data.id})"
                    )

            db.commit()
            logger.info(f"Processing completed for source {source_id}")

        return created_blocks
