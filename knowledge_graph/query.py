import logging
import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy import text, func
from sqlalchemy.orm import joinedload

from knowledge_graph.models import Entity, Relationship, AnalysisBlueprint
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


# Convenience functions
def query_topic_graph(topic_name: str, database_uri: Optional[str] = None) -> Dict:
    """Quick function to get complete topic graph overview from specified database"""
    query = NarrativeGraphQuery(database_uri)
    return query.export_topic_graph_to_json(topic_name)


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
