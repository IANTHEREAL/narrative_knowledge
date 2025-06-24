import logging
import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy import text, func
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import aliased

from knowledge_graph.models import (
    Entity,
    Relationship,
    AnalysisBlueprint,
    SourceGraphMapping,
)
from setting.db import SessionLocal, db_manager
from llm.embedding import get_text_embedding

logger = logging.getLogger(__name__)


class NarrativeGraphQuery:
    """Query interface for narrative knowledge graphs with multi-database support"""

    def __init__(self, database_uri: Optional[str] = None):
        """
        Initialize query interface for specific database.

        Args:
            database_uri: Database URI to query from. None or empty string means local database.
        """
        self.database_uri = database_uri
        self.session_factory = db_manager.get_session_factory(database_uri)
        self.is_local = db_manager.is_local_mode(database_uri)

    def get_topic_blueprint(self, topic_name: str) -> Optional[AnalysisBlueprint]:
        """Get the latest analysis blueprint for a topic"""
        with self.session_factory() as db:
            return (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name.like(f"%{topic_name}%"))
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

    def get_topic_entities(self, topic_name: str) -> pd.DataFrame:
        """Get all entities for a specific topic"""
        with self.session_factory() as db:
            query = text(
                """
                SELECT 
                    e.id,
                    e.name,
                    e.description,
                    e.attributes,
                    e.created_at
                FROM entities e
                WHERE JSON_EXTRACT(e.attributes, '$.topic_name') LIKE :topic_name
                ORDER BY e.created_at DESC
            """
            )

            result = db.execute(query, {"topic_name": f"%{topic_name}%"})
            return pd.DataFrame(result.fetchall())

    def get_topic_relationships(self, topic_name: str) -> pd.DataFrame:
        """Get all relationships for a specific topic"""
        with self.session_factory() as db:
            query = text(
                """
                SELECT 
                    r.id,
                    e1.name as source_entity,
                    r.relationship_desc,
                    e2.name as target_entity,
                    r.attributes,
                    r.created_at
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id = e1.id
                JOIN entities e2 ON r.target_entity_id = e2.id
                WHERE JSON_EXTRACT(r.attributes, '$.topic_name') LIKE :topic_name
                ORDER BY r.created_at DESC
            """
            )

            result = db.execute(query, {"topic_name": f"%{topic_name}%"})
            return pd.DataFrame(result.fetchall())

    def export_topic_graph_to_json(self, topic_name: str) -> Dict:
        """Export complete topic knowledge graph to JSON format"""
        entities_df = self.get_topic_entities(topic_name)
        relationships_df = self.get_topic_relationships(topic_name)
        blueprint = self.get_topic_blueprint(topic_name)

        return {
            "topic_name": topic_name,
            "database_uri": "local" if self.is_local else "external",
            "blueprint": {
                "processing_items": blueprint.processing_items if blueprint else {},
                "processing_instructions": (
                    blueprint.processing_instructions if blueprint else ""
                ),
            },
            "entities": entities_df.to_dict("records") if not entities_df.empty else [],
            "relationships": (
                relationships_df.to_dict("records")
                if not relationships_df.empty
                else []
            ),
        }

    def query_existing_knowledge(self, source_id: str, topic_name: str) -> Dict:
        """
        Query existing entities and relationships related to the document and topic with optimized performance.

        Uses efficient joins to avoid N+1 queries and fetch all related data in minimal database roundtrips.

        Args:
            source_id: Source document ID
            topic_name: Topic name to filter by

        Returns:
            Dict with existing_entities, existing_relationships, total counts
        """
        with self.session_factory() as db:
            # Get all graph elements mapped to this source in one query
            source_mappings = (
                db.query(SourceGraphMapping)
                .filter(
                    SourceGraphMapping.source_id == source_id,
                    SourceGraphMapping.attributes["topic_name"] == topic_name,
                )
                .all()
            )

            entity_ids = [
                mapping.graph_element_id
                for mapping in source_mappings
                if mapping.graph_element_type == "entity"
            ]
            relationship_ids = [
                mapping.graph_element_id
                for mapping in source_mappings
                if mapping.graph_element_type == "relationship"
            ]

            # Query entities efficiently
            existing_entities = []
            entity_map = {}  # id -> entity_data mapping
            if entity_ids:
                entities = db.query(Entity).filter(Entity.id.in_(entity_ids)).all()
                for entity in entities:
                    entity_data = {
                        "id": entity.id,
                        "name": entity.name,
                        "description": entity.description,
                        "attributes": entity.attributes or {},
                    }
                    existing_entities.append(entity_data)
                    entity_map[entity.id] = entity_data

            # Query relationships with entity details in one efficient query using joins
            existing_relationships = []
            if relationship_ids:
                # Use aliases for clarity
                source_entity = aliased(Entity)
                target_entity = aliased(Entity)

                # Single query to get relationships with all entity details
                relationship_query = (
                    db.query(
                        Relationship,
                        source_entity.id.label("source_id"),
                        source_entity.name.label("source_name"),
                        source_entity.description.label("source_description"),
                        target_entity.id.label("target_id"),
                        target_entity.name.label("target_name"),
                        target_entity.description.label("target_description"),
                    )
                    .join(
                        source_entity, Relationship.source_entity_id == source_entity.id
                    )
                    .join(
                        target_entity, Relationship.target_entity_id == target_entity.id
                    )
                    .filter(Relationship.id.in_(relationship_ids))
                    .all()
                )

                for row in relationship_query:
                    rel = row.Relationship
                    rel_data = {
                        "id": rel.id,
                        "source_entity": {
                            "id": row.source_id,
                            "name": row.source_name,
                            "description": row.source_description,
                        },
                        "target_entity": {
                            "id": row.target_id,
                            "name": row.target_name,
                            "description": row.target_description,
                        },
                        "relationship_desc": rel.relationship_desc,
                        "attributes": rel.attributes or {},
                    }
                    existing_relationships.append(rel_data)

            logger.info(
                f"Queried existing knowledge for source {source_id}: "
                f"{len(existing_entities)} entities, {len(existing_relationships)} relationships"
            )

            return {
                "existing_entities": list(entity_map.values()),
                "existing_relationships": existing_relationships,
                "total_entities": len(existing_entities),
                "total_relationships": len(existing_relationships),
            }


# Convenience functions
def query_topic_graph(topic_name: str, database_uri: Optional[str] = None) -> Dict:
    """Quick function to get complete topic graph overview from specified database"""
    query = NarrativeGraphQuery(database_uri)
    return query.export_topic_graph_to_json(topic_name)


def query_existing_knowledge(
    source_id: str, topic_name: str, database_uri: Optional[str] = None
) -> Dict:
    """
    Quick function to query existing knowledge for a specific source and topic.

    Args:
        source_id: Source document ID
        topic_name: Topic name to filter by
        database_uri: Database URI to query from. None means local database.

    Returns:
        Dict with existing_entities, existing_relationships, and entity_map
    """
    query = NarrativeGraphQuery(database_uri)
    return query.query_existing_knowledge(source_id, topic_name)


def search_relationships_by_vector_similarity(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.6,
    database_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search relationships by vector similarity to query text from specified database.

    Args:
        query: The query text to search for
        top_k: Number of top similar relationships to return
        similarity_threshold: Minimum similarity score (0-1)
        database_uri: Database URI to search from. None means local database.

    Returns:
        DataFrame with relationships and entities, ordered by similarity
    """
    try:
        # Generate embedding for the query
        query_embedding = get_text_embedding(query)
        session_factory = db_manager.get_session_factory(database_uri)
        with session_factory() as db:
            # Build the base query with vector similarity search
            base_query = """
                SELECT 
                    r.id,
                    e1.name as source_entity,
                    e1.description as source_entity_description,
                    r.relationship_desc,
                    e2.name as target_entity,
                    e2.description as target_entity_description,
                    r.attributes,
                    r.created_at,
                    VEC_COSINE_DISTANCE(r.relationship_desc_vec, :query_vector) as similarity_distance,
                    (1 - VEC_COSINE_DISTANCE(r.relationship_desc_vec, :query_vector)) as similarity_score
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id = e1.id
                JOIN entities e2 ON r.target_entity_id = e2.id
                WHERE r.relationship_desc_vec IS NOT NULL
                ORDER BY similarity_distance ASC LIMIT :top_k
            """

            params = {"query_vector": str(query_embedding), "top_k": top_k * 5}

            result = db.execute(text(base_query), params)
            columns = [
                "id",
                "source_entity",
                "source_entity_description",
                "relationship_desc",
                "target_entity",
                "target_entity_description",
                "attributes",
                "created_at",
                "similarity_distance",
                "similarity_score",
            ]

            rows = result.fetchall()
            logger.info(f"found rows: {len(rows)}")
            df = pd.DataFrame(rows, columns=columns)

            # Round similarity scores for better readability
            if not df.empty:
                df["similarity_score"] = df["similarity_score"].round(4)
                df["similarity_distance"] = df["similarity_distance"].round(4)

            # filter out rows with similarity_score less than the threshold
            df = df[df["similarity_score"] >= similarity_threshold]

            # Sort by similarity_score in descending order and return top 10
            df = df.sort_values("similarity_score", ascending=False).head(top_k)

            return df

    except Exception as e:
        logger.error(f"Error in vector similarity search: {str(e)}")
        # Fallback to keyword search if vector search fails
        raise e
