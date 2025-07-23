import uuid
import hashlib
from sqlalchemy import (
    BigInteger,
    Column,
    String,
    Text,
    ForeignKey,
    DateTime,
    Enum,
    Index,
    JSON,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from tidb_vector.sqlalchemy import VectorType

Base = declarative_base()


# ============================================================================
# Core Storage Tables (Keep from existing design)
# ============================================================================

class ContentStore(Base):
    """Content storage with hash-based deduplication"""

    __tablename__ = "content_store"

    content_hash = Column(String(64), primary_key=True)
    content = Column(LONGTEXT, nullable=False)
    content_size = Column(BigInteger, nullable=False)
    content_type = Column(String(50), nullable=False, default="text/plain")
    name = Column(String(255), nullable=False)
    link = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    source_data_entries = relationship("SourceData", back_populates="content_store")

    __table_args__ = (
        Index("idx_content_store_size", "content_size"),
        Index("idx_content_store_type", "content_type"),
    )

    def __repr__(self):
        return f"<ContentStore(hash={self.content_hash[:8]}...)>"


class SourceData(Base):
    """Source document entity - serves as data source for knowledge extraction"""

    __tablename__ = "source_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)

    # Reference to deduplicated content
    content_hash = Column(
        String(64), ForeignKey("content_store.content_hash"), nullable=True
    )
    link = Column(String(512), nullable=True)

    source_type = Column(String(50), nullable=False, default="text/plain")
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    content_store = relationship("ContentStore", back_populates="source_data_entries")
    block_mappings = relationship(
        "BlockSourceMapping", back_populates="source", cascade="all, delete-orphan"
    )
    graph_mappings = relationship(
        "SourceGraphMapping", back_populates="source", cascade="all, delete-orphan"
    )

    # Content access properties
    @property
    def effective_content(self):
        """Get content from content_store"""
        if self.content_store:
            return self.content_store.content
        return None

    @property
    def effective_hash(self):
        """Get content hash from content_hash field"""
        return self.content_hash

    __table_args__ = (
        Index("uq_source_data_link", "link", unique=True),
        Index("idx_source_data_name", "name"),
        Index("idx_source_data_type", "source_type"),
        Index("idx_source_data_content_hash", "content_hash"),
        Index("idx_source_data_link_time", "link", "created_at"),
    )

    def __repr__(self):
        return f"<SourceData(id={self.id}, name={self.name}, link={self.link})>"


class BlockSourceMapping(Base):
    """Mapping table for KnowledgeBlock ↔ SourceData relationships"""

    __tablename__ = "block_source_mapping"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    block_id = Column(String(36), ForeignKey("knowledge_blocks.id"), nullable=False)
    source_id = Column(String(36), ForeignKey("source_data.id"), nullable=False)
    position_in_source = Column(BigInteger, default=0)  # Position within source
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    block = relationship("KnowledgeBlock", back_populates="source_mappings")
    source = relationship("SourceData", back_populates="block_mappings")

    __table_args__ = (
        Index("idx_block_source_mapping_block_id", "block_id"),
        Index("idx_block_source_mapping_source_id", "source_id"),
        Index(
            "uq_block_source_mapping_block_source", "block_id", "source_id", unique=True
        ),
    )

    def __repr__(self):
        return f"<BlockSourceMapping(block={self.block_id}, source={self.source_id})>"


class SourceGraphMapping(Base):
    """Mapping table for SourceData ↔ (Entity/Relationship) relationships"""

    __tablename__ = "source_graph_mapping"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String(36), ForeignKey("source_data.id"), nullable=False)
    graph_element_id = Column(
        String(36), nullable=False
    )  # entity_id or relationship_id
    graph_element_type = Column(Enum("entity", "relationship"), nullable=False)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source = relationship("SourceData", back_populates="graph_mappings")

    __table_args__ = (
        Index("idx_source_graph_mapping_source_id", "source_id"),
        Index(
            "idx_source_graph_mapping_element", "graph_element_type", "graph_element_id"
        ),
    )

    def __repr__(self):
        return f"<SourceGraphMapping(source={self.source_id}, element={self.graph_element_type}:{self.graph_element_id})>"


class KnowledgeBlock(Base):
    """Core graph node - represents atomic knowledge units"""

    __tablename__ = "knowledge_blocks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(512), nullable=False)
    knowledge_type = Column(
        Enum(
            "qa",
            "paragraph",
            "synopsis",
            "image",
            "video",
            "code",
            "chat_summary",
            "chat_content",
        ),
        nullable=False,
    )
    content = Column(LONGTEXT, nullable=True)
    context = Column(Text, nullable=True)
    # fixed 4096 dimension embedding for knowledge graph, fix it later
    content_vec = Column(VectorType(4096), nullable=True)
    hash = Column(
        String(64),
        nullable=False,
        unique=True,
        comment="SHA-256 hash for deduplication",
    )
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_mappings = relationship(
        "BlockSourceMapping", back_populates="block", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_knowledge_blocks_knowledge_type", "knowledge_type"),
        Index("idx_knowledge_blocks_name", "name"),
    )

    def __repr__(self):
        return f"<KnowledgeBlock(id={self.id}, name={self.name}, type={self.knowledge_type})>"


class Entity(Base):
    """High-level entity node - represents abstract knowledge entities"""

    __tablename__ = "entities"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    # fixed 4096 dimension embedding for knowledge graph, fix it later
    description_vec = Column(VectorType(4096), nullable=True)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    __table_args__ = (Index("idx_entities_name", "name"),)

    def __repr__(self):
        return f"<Entity(id={self.id}, name={self.name})>"


class Relationship(Base):
    """Graph edge - represents relationships between entities"""

    __tablename__ = "relationships"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_entity_id = Column(String(36), ForeignKey("entities.id"), nullable=False)
    target_entity_id = Column(String(36), ForeignKey("entities.id"), nullable=False)
    relationship_desc = Column(Text, nullable=True)
    # fixed 4096 dimension embedding for knowledge graph, fix it later
    relationship_desc_vec = Column(VectorType(4096), nullable=True)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_entity = relationship("Entity", foreign_keys=[source_entity_id])
    target_entity = relationship("Entity", foreign_keys=[target_entity_id])

    __table_args__ = (
        Index("idx_relationships_source", "source_entity_id"),
        Index("idx_relationships_target", "target_entity_id"),
        Index("idx_relationships_pair", "source_entity_id", "target_entity_id"),
    )

    def __repr__(self):
        return f"<Relationship(id={self.id}, source={self.source_entity_id}, target={self.target_entity_id})>"


class KnowledgeBlock(Base):
    """Core graph node - represents atomic knowledge units"""

    __tablename__ = "knowledge_blocks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(512), nullable=False)
    knowledge_type = Column(
        Enum(
            "qa",
            "paragraph",
            "synopsis",
            "image",
            "video",
            "code",
            "chat_summary",
            "chat_content",
        ),
        nullable=False,
    )
    content = Column(LONGTEXT, nullable=True)
    context = Column(Text, nullable=True)
    content_vec = Column(VectorType(4096), nullable=True)
    hash = Column(
        String(64),
        nullable=False,
        unique=True,
        comment="SHA-256 hash for deduplication",
    )
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_mappings = relationship(
        "BlockSourceMapping", back_populates="block", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_knowledge_blocks_hash", "hash"),
        Index("idx_knowledge_blocks_type", "knowledge_type"),
        Index("idx_knowledge_blocks_name", "name"),
    )

    def __repr__(self):
        return f"<KnowledgeBlock(id={self.id}, name={self.name}, type={self.knowledge_type})>"


# ============================================================================
# Core State Objects (New Architecture)
# ============================================================================

class RawDataSource(Base):
    """Raw data source - represents uploaded files before ETL processing"""

    __tablename__ = "raw_data_sources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=False)
    topic_name = Column(String(255), nullable=False)  # Key attribute for grouping
    file_size = Column(BigInteger, nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 hash of file content
    metadata = Column(JSON, nullable=True)  # Custom metadata from upload
    
    # Status tracking for ETL processing
    status = Column(
        Enum("uploaded", "etl_pending", "etl_processing", "etl_completed", "etl_failed"),
        nullable=False,
        default="uploaded",
    )
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_data_entries = relationship("SourceData", back_populates="raw_data_source")

    __table_args__ = (
        Index("idx_raw_data_topic", "topic_name"),
        Index("idx_raw_data_status", "status"),
        Index("idx_raw_data_hash", "file_hash"),
        Index("idx_raw_data_topic_status", "topic_name", "status"),
    )

    def __repr__(self):
        return f"<RawDataSource(id={self.id}, filename={self.original_filename}, topic={self.topic_name}, status={self.status})>"


class SourceData(Base):
    """Source document entity - ETL processed data ready for graph building"""

    __tablename__ = "source_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    topic_name = Column(String(255), nullable=False)  # Key attribute for grouping

    # Reference to raw data source
    raw_data_source_id = Column(
        String(36), ForeignKey("raw_data_sources.id"), nullable=True
    )
    
    # Reference to deduplicated content
    content_hash = Column(
        String(64), ForeignKey("content_store.content_hash"), nullable=True
    )
    link = Column(String(512), nullable=True)

    source_type = Column(String(50), nullable=False, default="text/plain")
    attributes = Column(JSON, nullable=True)
    
    # Status tracking for graph building
    status = Column(
        Enum("created", "updated", "graph_pending", "graph_processing", "graph_completed", "graph_failed"),
        nullable=False,
        default="created",
    )
    
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    raw_data_source = relationship("RawDataSource", back_populates="source_data_entries")
    content_store = relationship("ContentStore", back_populates="source_data_entries")
    block_mappings = relationship(
        "BlockSourceMapping", back_populates="source", cascade="all, delete-orphan"
    )
    graph_mappings = relationship(
        "SourceGraphMapping", back_populates="source", cascade="all, delete-orphan"
    )

    # Content access properties
    @property
    def effective_content(self):
        """Get content from content_store"""
        if self.content_store:
            return self.content_store.content
        return None

    @property
    def effective_hash(self):
        """Get content hash from content_hash field"""
        return self.content_hash

    __table_args__ = (
        Index("uq_source_data_link", "link", unique=True),
        Index("idx_source_data_name", "name"),
        Index("idx_source_data_topic", "topic_name"),
        Index("idx_source_data_status", "status"),
        Index("idx_source_data_content_hash", "content_hash"),
        Index("idx_source_data_topic_status", "topic_name", "status"),
        Index("idx_source_data_raw_id", "raw_data_source_id"),
    )

    def __repr__(self):
        return f"<SourceData(id={self.id}, name={self.name}, topic={self.topic_name}, status={self.status})>"


class AnalysisBlueprint(Base):
    """
    Analysis blueprint for a specific topic - generated from multiple source documents
    """

    __tablename__ = "analysis_blueprints"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    topic_name = Column(String(255), nullable=False)

    processing_instructions = Column(
        Text, nullable=True
    )  # Human-readable processing guidance

    # Status tracking for blueprint generation
    status = Column(
        Enum("outdated", "generating", "ready", "failed"),
        nullable=False,
        default="outdated",
    )
    # Track which source data contributed to this blueprint
    contributing_source_data_ids = Column(JSON, nullable=False)  # List of source_data IDs
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_analysis_blueprints_topic_name", "topic_name"),
        Index("idx_analysis_blueprints_status", "status"),
        Index("idx_analysis_blueprints_topic_status", "topic_name", "status"),
        Index("uq_analysis_blueprints_topic", "topic_name", unique=True),
    )

    def __repr__(self):
        return f"<AnalysisBlueprint(id={self.id}, topic={self.topic_name}, status={self.status})>"


# ============================================================================
# Mapping Tables (Keep from existing design)
# ============================================================================

class BlockSourceMapping(Base):
    """Mapping table for KnowledgeBlock ↔ SourceData relationships"""

    __tablename__ = "block_source_mapping"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    block_id = Column(String(36), ForeignKey("knowledge_blocks.id"), nullable=False)
    source_id = Column(String(36), ForeignKey("source_data.id"), nullable=False)
    position_in_source = Column(BigInteger, default=0)  # Position within source
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    block = relationship("KnowledgeBlock", back_populates="source_mappings")
    source = relationship("SourceData", back_populates="block_mappings")

    __table_args__ = (
        Index("uq_block_source_mapping", "block_id", "source_id", unique=True),
        Index("idx_block_source_mapping_block", "block_id"),
        Index("idx_block_source_mapping_source", "source_id"),
    )

    def __repr__(self):
        return f"<BlockSourceMapping(block_id={self.block_id}, source_id={self.source_id})>"


class SourceGraphMapping(Base):
    """Mapping table for SourceData ↔ (Entity/Relationship) relationships"""

    __tablename__ = "source_graph_mapping"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String(36), ForeignKey("source_data.id"), nullable=False)
    graph_element_id = Column(
        String(36), nullable=False
    )  # entity_id or relationship_id
    graph_element_type = Column(Enum("entity", "relationship"), nullable=False)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source = relationship("SourceData", back_populates="graph_mappings")

    __table_args__ = (
        Index("idx_source_graph_mapping_source", "source_id"),
        Index("idx_source_graph_mapping_element", "graph_element_id"),
        Index("idx_source_graph_mapping_type", "graph_element_type"),
        Index(
            "uq_source_graph_mapping",
            "source_id",
            "graph_element_id",
            "graph_element_type",
            unique=True,
        ),
    )

    def __repr__(self):
        return f"<SourceGraphMapping(source_id={self.source_id}, element_id={self.graph_element_id}, type={self.graph_element_type})>"
    
class DocumentSummary(Base):
    """Document summary focused on specific topics for efficient blueprint generation"""

    __tablename__ = "document_summaries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("source_data.id"), nullable=False)
    topic_name = Column(String(255), nullable=False)
    summary_content = Column(LONGTEXT, nullable=False)
    key_entities = Column(JSON, nullable=True)  # List of key entities mentioned
    main_themes = Column(JSON, nullable=True)  # Main topics and themes
    business_context = Column(Text, nullable=True)  # Business context and importance
    document_type = Column(String(100), nullable=True)  # Type classification
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_data = relationship("SourceData")

    __table_args__ = (
        Index("idx_document_summaries_topic", "topic_name"),
        Index("idx_document_summaries_doc_id", "document_id"),
        Index("uq_doc_topic_summary", "document_id", "topic_name", unique=True),
    )

    def __repr__(self):
        return f"<DocumentSummary(doc_id={self.document_id}, topic={self.topic_name})>"