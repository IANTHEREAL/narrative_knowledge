"""
Background daemon for processing sources without graph mappings.
"""

import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import and_, func, or_, text
from sqlalchemy.orm import Session

from setting.db import SessionLocal, db_manager
from knowledge_graph.models import SourceData, SourceGraphMapping
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding

logger = logging.getLogger(__name__)


class KnowledgeGraphDaemon:
    """
    Background daemon for building knowledge graphs from sources without graph mappings.
    """

    def __init__(
        self,
        llm_client: Optional[LLMInterface] = None,
        embedding_func=None,
        check_interval: int = 120,
        worker_count: int = 3
    ):
        """
        Initialize the knowledge graph daemon.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            check_interval: Interval in seconds to check for unmapped sources
            worker_count: Number of workers for graph building
        """
        self.llm_client = llm_client or LLMInterface("openai_like", "qwen3-32b")
        self.embedding_func = embedding_func or get_text_embedding
        self.check_interval = check_interval
        self.worker_count = worker_count
        self.is_running = False

    def start(self):
        """Start the daemon."""
        self.is_running = True
        logger.info("Knowledge graph daemon started")

        while self.is_running:
            try:
                self._process_unmapped_sources()
            except Exception as e:
                logger.error(f"Error in graph daemon main loop: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def stop(self):
        """Stop the daemon."""
        self.is_running = False
        logger.info("Knowledge graph daemon stopped")

    def _process_unmapped_sources(self):
        """Find and process sources that don't have graph mappings."""
        # First get completed topics from GraphBuild table
        completed_topics = self._get_completed_topics()
        
        if not completed_topics:
            logger.debug("No completed topics found")
            return
        
        # Process each completed topic
        for topic_info in completed_topics:
            topic_name = topic_info["topic_name"]
            database_uri = topic_info["database_uri"]
            
            # Get unmapped sources for this specific topic and database
            sources = self._get_unmapped_sources_for_topic(database_uri, topic_name)
            
            if sources:
                logger.info(f"Found {len(sources)} unmapped sources for topic '{topic_name}' in database: {'local' if not database_uri else database_uri}")
                self._process_sources_batch(sources, database_uri)

    def _get_completed_topics(self) -> List[Dict]:
        """Get list of completed topics with their database URIs."""
        with SessionLocal() as db:
            result = db.execute(
                text("""
                SELECT DISTINCT topic_name, external_database_uri 
                FROM graph_build 
                WHERE status = 'completed'
                ORDER BY topic_name, external_database_uri
                """)
            ).fetchall()
            
            topics = []
            for row in result:
                topics.append({
                    "topic_name": row[0],
                    "database_uri": row[1] or ""  # Convert None to empty string
                })
            
            return topics

    def _get_external_databases(self) -> List[str]:
        """Get list of external database URIs from GraphBuild table."""
        with SessionLocal() as db:
            result = db.execute(
                text("""
                SELECT DISTINCT external_database_uri 
                FROM graph_build 
                WHERE external_database_uri != '' 
                AND external_database_uri IS NOT NULL
                """)
            ).fetchall()
            
            return [row[0] for row in result]

    def _get_unmapped_sources_for_topic(self, database_uri: str, topic_name: str) -> List[Dict]:
        """
        Get sources for a specific topic that don't have any graph mappings.
        
        Args:
            database_uri: Database URI to query from
            topic_name: Topic name to filter sources
            
        Returns:
            List of source data dictionaries
        """
        if db_manager.is_local_mode(database_uri):
            session_factory = SessionLocal
        else:
            session_factory = db_manager.get_session_factory(database_uri)
        
        with session_factory() as db:
            # Query for SourceData that don't have any SourceGraphMapping entries
            # and match the specific topic_name
            query = text("""
                SELECT 
                    sd.id,
                    sd.name,
                    cs.content,
                    sd.link,
                    sd.source_type,
                    sd.attributes,
                    sd.created_at
                FROM source_data sd
                LEFT JOIN content_store cs ON sd.content_hash = cs.content_hash
                LEFT JOIN source_graph_mapping sgm ON sd.id = sgm.source_id
                WHERE sgm.source_id IS NULL
                AND cs.content IS NOT NULL
                AND JSON_UNQUOTE(JSON_EXTRACT(sd.attributes, '$.topic_name')) = :topic_name
                ORDER BY sd.created_at ASC
            """)
            
            result = db.execute(query, {
                "topic_name": topic_name
            }).fetchall()
            
            sources = []
            for row in result:
                attributes = row.attributes or {}
                
                sources.append({
                    "source_id": row.id,
                    "source_name": row.name,
                    "source_content": row.content,
                    "source_link": row.link,
                    "source_type": row.source_type,
                    "source_attributes": attributes,
                    "topic_name": topic_name,
                })
            
            return sources

    def _process_sources_batch(self, sources: List[Dict], database_uri: str):
        """
        Process a batch of sources for graph building.
        
        Args:
            sources: List of source data
            database_uri: Database URI where sources are located
        """
        if not sources:
            return
        
        # Group sources by topic_name
        topic_groups = {}
        for source in sources:
            topic_name = source["topic_name"]
            if topic_name not in topic_groups:
                topic_groups[topic_name] = []
            topic_groups[topic_name].append(source)
        
        # Process each topic group
        for topic_name, topic_sources in topic_groups.items():
            try:
                self._build_graph_for_topic(topic_name, topic_sources, database_uri)
            except Exception as e:
                logger.error(f"Failed to build graph for topic {topic_name}: {e}", exc_info=True)

    def _build_graph_for_topic(self, topic_name: str, sources: List[Dict], database_uri: str):
        """
        Build knowledge graph for a specific topic and sources.
        
        Args:
            topic_name: Name of the topic
            sources: List of source data for this topic
            database_uri: Database URI where sources are located
        """
        if db_manager.is_local_mode(database_uri):
            session_factory = SessionLocal
            logger.info(f"Building knowledge graph for local topic: {topic_name} with {len(sources)} sources")
        else:
            session_factory = db_manager.get_session_factory(database_uri)
            logger.info(f"Building knowledge graph for external topic: {topic_name} with {len(sources)} sources")

        # Prepare extracted sources format for KnowledgeGraphBuilder
        extracted_sources = []
        for source in sources:
            extracted_sources.append({
                "source_id": source["source_id"],
                "source_name": source["source_name"],
                "source_content": source["source_content"],
                "source_link": source["source_link"],
                "source_attributes": source["source_attributes"],
            })

        try:
            graph_builder = KnowledgeGraphBuilder(
                self.llm_client, self.embedding_func, session_factory, self.worker_count
            )
            
            result = graph_builder.build_knowledge_graph(topic_name, extracted_sources)
            
            logger.info(f"Successfully built knowledge graph for topic: {topic_name}")
            logger.info(f"Graph build results: {result}")
            
        except Exception as e:
            logger.error(f"Knowledge graph building failed for topic {topic_name}: {str(e)}", exc_info=True)
            raise

        try:
            result = graph_builder.enhance_knowledge_graph(topic_name, extracted_sources)
            logger.info(f"Successfully enhanced knowledge graph for topic: {topic_name}")
            logger.info(f"Enhancement results: {result}")
        except Exception as e:
            logger.error(f"Failed to enhance knowledge graph: {e}", exc_info=True)
            raise


    def get_daemon_status(self) -> Dict:
        """
        Get current daemon status and statistics.

        Returns:
            Dictionary with daemon status information
        """
        status = {
            "is_running": self.is_running,
            "check_interval": self.check_interval,
            "worker_count": self.worker_count,
        }
        
        # Get stats from completed topics
        completed_topics = self._get_completed_topics()
        topic_stats = {}
        total_unmapped = 0
        
        for topic_info in completed_topics:
            topic_name = topic_info["topic_name"]
            database_uri = topic_info["database_uri"]
            topic_key = f"{topic_name} ({'local' if not database_uri else database_uri})"
            
            try:
                unmapped_count = len(self._get_unmapped_sources_for_topic(database_uri, topic_name))
                topic_stats[topic_key] = unmapped_count
                total_unmapped += unmapped_count
            except Exception as e:
                logger.error(f"Error getting stats for topic {topic_name} in database {database_uri}: {e}")
                topic_stats[topic_key] = f"Error: {str(e)}"
        
        status["completed_topics"] = topic_stats
        status["total_unmapped_sources"] = total_unmapped
        
        # Get graph mapping statistics
        try:
            with SessionLocal() as db:
                total_mappings = db.query(SourceGraphMapping).count()
                entity_mappings = db.query(SourceGraphMapping).filter(
                    SourceGraphMapping.graph_element_type == "entity"
                ).count()
                relationship_mappings = db.query(SourceGraphMapping).filter(
                    SourceGraphMapping.graph_element_type == "relationship"
                ).count()
                
                status["total_graph_mappings"] = total_mappings
                status["entity_mappings"] = entity_mappings
                status["relationship_mappings"] = relationship_mappings
        except Exception as e:
            logger.error(f"Error getting mapping statistics: {e}")
            status["mapping_stats_error"] = str(e)
        
        return status 