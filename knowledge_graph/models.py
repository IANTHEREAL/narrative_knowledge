import uuid
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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from tidb_vector.sqlalchemy import VectorType

Base = declarative_base()


class SourceData(Base):
    """Source document entity - serves as data source for knowledge extraction"""

    __tablename__ = "source_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    content = Column(LONGTEXT, nullable=True)
    link = Column(String(512), nullable=True)
    source_type = Column(
        Enum(
            "document",
            "code",
            "image",
            "video",
            "pdf",
            "spreadsheet",
            "sql",
            "markdown",
        ),
        nullable=False,
        default="document",
    )
    attributes = Column(JSON, nullable=True)
    hash = Column(
        String(64), nullable=True
    )  # SHA-256 hash for content deduplication and change detection
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    block_mappings = relationship(
        "BlockSourceMapping", back_populates="source", cascade="all, delete-orphan"
    )
    graph_mappings = relationship(
        "SourceGraphMapping", back_populates="source", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("uq_source_data_link", "link", unique=True),
        Index("idx_source_data_name", "name"),
        Index("idx_source_source_type", "source_type"),
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
        Enum("qa", "paragraph", "synopsis", "image", "video", "code"), nullable=False
    )
    content = Column(LONGTEXT, nullable=True)
    context = Column(Text, nullable=True)
    content_vec = Column(VectorType(1536), nullable=True)
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
    description_vec = Column(VectorType(1536), nullable=True)
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
    relationship_desc_vec = Column(VectorType(1536), nullable=True)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source_entity = relationship("Entity", foreign_keys=[source_entity_id])
    target_entity = relationship("Entity", foreign_keys=[target_entity_id])

    __table_args__ = (
        Index("idx_relationships_source_entity_id", "source_entity_id"),
        Index("idx_relationships_target_entity_id", "target_entity_id"),
    )

    def __repr__(self):
        return f"<Relationship(source={self.source_entity_id}, target={self.target_entity_id}, desc={self.relationship_desc})>"


class AnalysisBlueprint(Base):
    """Analysis blueprint for each client - stores extraction strategy"""

    __tablename__ = "analysis_blueprints"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    topic_name = Column(String(255), nullable=False)
    suggested_entity_types = Column(JSON, nullable=True)
    key_narrative_themes = Column(JSON, nullable=True)
    processing_instructions = Column(Text, nullable=True)
    attributes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    __table_args__ = (Index("idx_analysis_blueprints_topic_name", "topic_name"),)

    def __repr__(self):
        return f"<AnalysisBlueprint(client={self.topic_name}, created_at={self.created_at})>"


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


class GraphBuildStatus(Base):
    """Graph build status tracking for each topic-source combination"""

    __tablename__ = "graph_build_status"

    topic_name = Column(String(255), primary_key=True, nullable=False)
    source_id = Column(
        String(36), primary_key=True, nullable=False
    )  # Removed FK constraint for multi-db support
    external_database_uri = Column(
        String(512), nullable=False, default=""
    )  # Track external database
    storage_directory = Column(
        String(512), nullable=True
    )  # Directory path where document and metadata are stored
    status = Column(
        Enum("uploaded", "pending", "processing", "completed", "failed"),
        nullable=False,
        default="uploaded",
    )
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )
    scheduled_at = Column(
        DateTime, default=func.current_timestamp(), nullable=False
    )  # Schedule time for processing, defaults to created_at but can be modified
    error_message = Column(Text, nullable=True)
    progress_info = Column(JSON, nullable=True)  # Store build progress details

    # Note: Removed source_data relationship due to multi-database support
    # The source_data may exist in different databases

    __table_args__ = (
        Index("idx_graph_build_status_topic", "topic_name"),
        Index("idx_graph_build_status_source", "source_id"),
        Index("idx_graph_build_status_status", "status"),
        Index("idx_graph_build_status_created", "created_at"),
        Index("idx_graph_build_status_external_db", "external_database_uri"),
    )

    def __repr__(self):
        return f"<GraphBuildStatus(topic={self.topic_name}, source={self.source_id}, status={self.status})>"
