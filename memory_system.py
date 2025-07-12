import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, text

from knowledge_graph.models import (
    SourceData,
    KnowledgeBlock,
    AnalysisBlueprint,
    BlockSourceMapping,
    ContentStore,
    GraphBuild,
)
from knowledge_graph.knowledge import KnowledgeBuilder
from knowledge_graph.graph import NarrativeKnowledgeGraphBuilder
from setting.db import db_manager
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding

logger = logging.getLogger(__name__)

PersonalBlueprint_processing_instructions = """<Specification>
User Memory Extraction specification outlines the principles for extracting holistic user insights from conversation histories. The goal is to build a rich, multi-faceted understanding of the user's personality, interests, and life context.

<Part1>Key Information Patterns to Monitor

You must scan for signals that reveal the user as a person. The focus is expanded from purely technical skills to encompass personal life, interests, and individual characteristics.

1. Explicit Declarations of Identity & Role
    * Signal: Direct statements about their roles (professional or personal), characteristics, or affiliations.
    * Examples: "As a product manager...", "I'm a father of two.", "I'm an avid bird-watcher."
    * Value: Foundational, high-confidence facts about the user's life context.

2. Recurring Topics & Interests
    * Signal: Subjects the user frequently returns to, whether professional, academic, or purely for leisure.
    * Examples: Repeatedly discussing "18th-century French literature," "sustainable gardening," "the latest AI research," or "training for a marathon."
    * Value: Indicates passions, hobbies, and areas of deep, sustained interest.

3. Goals, Plans, & Challenges
    * Signal: Descriptions of future aspirations, current projects, or obstacles they face in any aspect of their life.
    * Examples: "My goal is to learn to cook Italian food.", "We're planning a family trip to Japan next year.", "I'm struggling to find time to read more."
    * Value: Reveals the user's current focus, motivations, and what matters to them right now.

4. Significant Life Events & Achievements
    * Signal: Mentions of key milestones, accomplishments, or significant changes in their life.
    * Examples: "My paper just got published.", "We finally launched our project after six months.", "I'm moving to London next month.", "I just ran my first 10k."
    * Value: These are crucial markers of change and progress in a user's life journey, providing major contextual shifts.

5.  Learning & Growth
    * Signal: Evidence of a user learning any new skill or topic, often shown by a progression from basic to more advanced questions.
    * Examples: "What are the basic chords on a guitar?" followed later by "How do I play a barre chord properly?". Or, "Can you recommend a good history podcast?"
    * Value: Illustrates the user's capacity for growth and their current areas of self-improvement.

6.  Opinions, Preferences, & Affinities
    * Signal: Expressions of taste, values, and personal preference for or against things.
    * Examples: "I find that Nike running shoes work best for me.", "I prefer reading physical books over e-books.", "I'm not a big fan of modern art."
    * Value: Builds a nuanced picture of the user's personality and what they value.

7.  Mentioned Resources, Tools, & Brands
    * Signal: Specific products, services, authors, or resources the user relies on or interacts with.
    * Examples: "I use Headspace for meditation.", "I'm reading a book by Haruki Murakami.", "We booked our flights on Kayak."
    * Value: Connects the user to a network of real-world entities, enriching their profile.

8. Additional memory-related insights can be extracted to provide users with meaningful data that facilitates better self-understanding and personal reflection.
</Part1>

<Part2>Strategy for Information Processing and Organization

The underlying logic for processing remains robust but is now applied to this broader set of patterns.

1.  Extraction and Organization Process

    * Step 1: Cluster: Group related signals. For example, cluster messages about "training schedules," "running shoes," and "signing up for a 10k race" under a single theme.
    * Step 2: Synthesize & Structure: Formulate a concise insight and structure it with key attributes. Each insight must include:
        *   **Insight Statement**: A clear, concise summary of the user trait. (e.g., "User is actively training for a running event.")
        *   **Insight declaration attributes**:
            *   **Confidence Level**: An assessment of confidence ("High", "Medium", or "Low"). This is determined by the quantity and quality of evidence. Explicit statements ("I am a...") or repeated mentions of a topic signal High confidence.
            *   **Temporal Context**: The valid time frame for the insight. This could be a specific date, a time range (e.g., "January-March 2024"), or "ongoing" if it seems to be a stable trait.
    * Step 3: Link Evidence: Every insight must be backed by evidence (a message summary and its timestamp) to ensure it is traceable and verifiable.

2.  Handling Special Cases

    * Handling Conflicts (as Evolution): Treat contradictions as evidence of personal growth or change.
        * Core Principle: A change of mind is a valuable insight into a person's journey.
        * Strategy: Create separate, time-bound insights. The Temporal Context is crucial here.
        * Example:
            1.  Insight A: "User expressed a dislike for modern art.", attributes: { confidence: "High", temporal_context: "Observed in 2023", other_attributes... }
            2.  Insight B: "User shared their excitement about visiting the Tate Modern gallery.", attributes: { confidence: "Medium", temporal_context: "Observed in 2024", other_attributes...}
        * Value: This combination tells a story: the user's taste in art is evolving.

    * Handling Repetition (as Confidence): Use repetition to strengthen an insight's validity.
        * Core Principle: When a user repeatedly brings up a topic, it signifies its importance to them.
        * Strategy: Instead of creating duplicates, add new messages as evidence to the existing insight and update its attributes. The Confidence Level should be increased (e.g., from "Medium" to "High"). The Temporal Context should also be updated to reflect the latest evidence.
        * Example: If a user who has a "Medium" confidence insight "User has an interest in hiking" now talks about buying new hiking boots, the insight is reinforced. Its Confidence Level should be upgraded to "High" and its Temporal Context updated.

</Part2>
</Specification>
"""


def generate_topic_name_for_personal_memory(user_id: str) -> str:
    """
    Generate a topic name for the user.
    """
    return f"The personal information of {user_id}"


def _generate_build_id_for_chat_batch(chat_link: str, database_uri: str = "") -> str:
    """
    Generate a deterministic build_id for chat batch based on chat_link and database_uri.

    Args:
        chat_link: The chat batch link
        database_uri: The database URI (empty string for local)

    Returns:
        SHA256 hash of the combined string
    """
    # Combine chat_link and database_uri for hash generation
    combined_string = f"{chat_link}||{database_uri}"
    
    # Generate SHA256 hash
    hash_object = hashlib.sha256(combined_string.encode("utf-8"))
    return hash_object.hexdigest()


class PersonalMemorySystem:
    """
    Personal Memory System for managing user chat history and generating insights.

    Key Features:
    - Process chat message batches into summarized knowledge blocks
    - Maintain personal insight analysis blueprints
    - Generate user insights as graph triplets
    - Handle memory conflicts and deduplication
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        embedding_func: Optional[Callable] = None,
        session_factory=None,
    ):
        """
        Initialize the personal memory system.

        Args:
            llm_client: LLM interface for processing
            embedding_func: Function to generate embeddings
            session_factory: Database session factory
        """
        self.llm_client = llm_client
        self.embedding_func = embedding_func or get_text_embedding
        self.SessionLocal = session_factory or db_manager.get_session_factory()
        self.knowledge_builder = KnowledgeBuilder(
            llm_client, embedding_func, session_factory
        )
        self.graph_builder = NarrativeKnowledgeGraphBuilder(
            llm_client, embedding_func, session_factory
        )

    def process_chat_batch(
        self,
        chat_messages: List[Dict],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Process a batch of chat messages into summarized knowledge.

        Args:
            chat_messages: List of chat message dicts with structure:
                {
                    "message_content": str,
                    "session_id": str,
                    "conversation_title": str,
                    "date": str (ISO format),
                    "role": "user" | "assistant"
                }
            user_id: User identifier

        Returns:
            Dict with processing results
        """
        topic_name = generate_topic_name_for_personal_memory(user_id)
        logger.info(
            f"Processing chat batch for user {user_id}: {len(chat_messages)} messages"
        )

        # Step 1: Store as SourceData
        source_data = self._store_chat_batch_as_source(
            chat_messages, user_id, topic_name
        )

        # Step 2: Create summary knowledge block
        knowledge_block = self._create_summary_knowledge_block(
            source_data, user_id, topic_name
        )

        # Step 3: create personal blueprint
        self.create_personal_blueprint(user_id, topic_name)

        # Step 4: Create GraphBuild record for background processing
        build_id = self._create_graph_build_task(
            source_data, user_id, topic_name
        )

        return {
            "status": "success",
            "source_id": source_data["id"],
            "knowledge_block_id": knowledge_block["id"],
            "build_id": build_id,
            "summary": knowledge_block["content"],
            "topic_name": topic_name,
        }

    def _store_chat_batch_as_source(
        self, chat_messages: List[Dict], user_id: str, topic_name: str
    ) -> SourceData:
        """
        Store chat batch as SourceData with summary.

        Args:
            chat_messages: Original chat messages
            chat_summary: Generated summary
            user_id: User identifier
            topic_name: Topic for categorization

        Returns:
            Created SourceData object
        """
        # Get the latest timestamp from the batch for use as a unique identifier
        if chat_messages and any(msg.get("date") for msg in chat_messages):
            # Dates are in ISO format, so string comparison is sufficient to find the max
            latest_timestamp_str = max(
                msg["date"] for msg in chat_messages if msg.get("date")
            )
            batch_timestamp = latest_timestamp_str
        else:
            # Fallback if for some reason the batch is empty or contains no dates
            batch_timestamp = datetime.now().isoformat()

        chat_link = f"memory://{user_id}/chat_batch/{batch_timestamp}"

        # Prepare attributes
        attributes = {
            "user_id": user_id,
            "topic_name": topic_name,
            "batch_type": "chat_messages",
            "message_count": len(chat_messages),
            "session_id": chat_messages[0].get("session_id", ""),
            "conversation_title": chat_messages[0].get("conversation_title", ""),
            "last_message_date": batch_timestamp,
        }

        # Serialize content data
        content_json = json.dumps(chat_messages, ensure_ascii=False, indent=2)
        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(content_json.encode("utf-8")).hexdigest()

        with self.SessionLocal() as db:
            # Check if source data already exists by doc_link
            existing_source = (
                db.query(SourceData).filter(SourceData.link == chat_link).first()
            )

            if existing_source:
                logger.info(
                    f"Chat batch already exists with link: {chat_link}, reusing: {existing_source.id}"
                )
                return {
                    "id": existing_source.id,
                    "name": existing_source.name,
                    "content": existing_source.effective_content,
                    "attributes": existing_source.attributes,
                }

            # Check if content already exists
            content_store = (
                db.query(ContentStore).filter_by(content_hash=content_hash).first()
            )

            if not content_store:
                # Create new content store
                content_store = ContentStore(
                    content_hash=content_hash,
                    content=content_json,
                    content_size=len(content_json.encode("utf-8")),
                    content_type="application/json",
                    name=f"chat_batch_{user_id}_{batch_timestamp}",
                    link=chat_link,
                )
                db.add(content_store)
                logger.info(
                    f"Created new content store entry with hash: {content_hash[:8]}..."
                )
            else:
                logger.info(
                    f"Reusing existing content store entry with hash: {content_hash[:8]}..."
                )

            # Create source data
            source_data = SourceData(
                name=f"chat_batch_{user_id}_{batch_timestamp}",
                link=chat_link,
                source_type="application/json",
                content_hash=content_store.content_hash,
                attributes=attributes,
            )

            db.add(source_data)
            db.commit()
            db.refresh(source_data)
            db.refresh(content_store)

            logger.info(f"Stored chat batch as SourceData: {source_data.id}")
            return {
                "id": source_data.id,
                "name": source_data.name,
                "content": content_store.content,
                "attributes": source_data.attributes,
            }

    def _create_summary_knowledge_block(
        self, source_data: Dict, user_id: str, topic_name: str
    ) -> Dict[str, Any]:
        """
        Create a summary knowledge block for the chat batch.

        Args:
            source_data: Source data object
            user_id: User identifier
            topic_name: Topic name

        Returns:
            Dict containing the created KnowledgeBlock information.
        """
        with self.SessionLocal() as db:
            # Check if a knowledge block already exists for this source
            existing_mapping = (
                db.query(BlockSourceMapping)
                .filter(BlockSourceMapping.source_id == source_data["id"])
                .first()
            )

            if existing_mapping:
                existing_block = (
                    db.query(KnowledgeBlock)
                    .filter(KnowledgeBlock.id == existing_mapping.block_id)
                    .first()
                )
                if existing_block:
                    logger.info(
                        f"Summary knowledge block already exists for source {source_data['id']}, reusing block {existing_block.id}"
                    )
                    return {
                        "id": existing_block.id,
                        "name": existing_block.name,
                        "content": existing_block.content,
                        "attributes": existing_block.attributes,
                    }

        # Create summary content for knowledge block
        summary_prompt = f"""Generate a narrative summary of this conversation following these guidelines:

## Core Principles
- **Fidelity First**: Stay true to the original conversation content
- **Narrative Flow**: Present events in a story-like sequence
- **Preserve Authenticity**: Use direct quotes from the original messages
- **Respect Timeline**: Maintain the actual temporal sequence of events

## What to Include
- **Actual timestamps and session details**
- **Direct quotes from user and assistant messages**
- **Real conversation flow and progression**
- **Specific technical details mentioned**
- **Concrete actions or decisions stated**

## What to Avoid
- **Emotional interpretations** not present in the original text
- **Psychological analysis** or assumed motivations
- **Embellished descriptions** that add fictional elements
- **Speculation** about unstated thoughts or feelings
- **Metaphorical language** that wasn't in the original

## Structure
1. **Opening**: Set the scene with actual time/session context
2. **Core Events**: Present the conversation flow with direct quotes
3. **Conclusion**: Summarize the actual outcomes or decisions made

---

## Conversation Data:

**Title**: {source_data["attributes"].get("conversation_title", source_data["name"])}
**Date**: {source_data["attributes"].get("last_message_date", "unknown")}

**Conversation Content**:
{source_data["content"]}

**Additional Attributes**:
{source_data["attributes"]}

Generate a concise narrative summary that captures the essence of this conversation while remaining faithful to the original content.
"""
        try:
            response = self.llm_client.generate(summary_prompt, max_tokens=4096)
            # remove <think> and </think> section if present
            if "</think>" in response:
                summary_content = response.split("</think>")[1]
            else:
                summary_content = response
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            raise

        # Generate hash for deduplication
        content_hash = hashlib.sha256(summary_content.encode("utf-8")).hexdigest()

        # Create knowledge block
        with self.SessionLocal() as db:
            # Create new block if it doesn't exist
            knowledge_block = KnowledgeBlock(
                name=f"Chat Summary - {user_id} - {source_data['attributes'].get('conversation_title', 'unknown')} at {source_data['attributes'].get('last_message_date', 'unknown')}",
                knowledge_type="chat_summary",
                context=source_data["content"],
                content=summary_content,
                content_vec=self.embedding_func(summary_content),
                hash=content_hash,
                attributes={"user_id": user_id, "topic_name": topic_name},
            )

            db.add(knowledge_block)
            db.flush()

            # Create source mapping
            mapping = BlockSourceMapping(
                block_id=knowledge_block.id,
                source_id=source_data["id"],
                position_in_source=0,
            )
            db.add(mapping)
            db.commit()
            db.refresh(knowledge_block)

            logger.info(f"Created summary knowledge block: {knowledge_block.id}")
            return {
                "id": knowledge_block.id,
                "name": knowledge_block.name,
                "content": knowledge_block.content,
                "attributes": knowledge_block.attributes,
            }

    def create_personal_blueprint(
        self,
        user_id: str,
        topic_name: str,
    ) -> AnalysisBlueprint:
        """
        Get or create personal insight analysis blueprint for a user.

        Args:
            user_id: User identifier
            topic_name: Topic name for memory categorization
            force_update: Whether to force blueprint regeneration

        Returns:
            Personal AnalysisBlueprint object
        """

        with self.SessionLocal() as db:
            # Check for existing personal blueprint
            existing_blueprint = (
                db.query(AnalysisBlueprint)
                .filter(AnalysisBlueprint.topic_name == topic_name)
                .order_by(AnalysisBlueprint.created_at.desc())
                .first()
            )

            if existing_blueprint:
                logger.info(f"Using existing personal blueprint for user {user_id}")
                return existing_blueprint

            blueprint = AnalysisBlueprint(
                topic_name=topic_name,
                processing_instructions=PersonalBlueprint_processing_instructions,
            )
            db.add(blueprint)
            db.commit()
            db.refresh(blueprint)
            logger.info(f"Created personal memory blueprint for user {user_id}")
            return blueprint

    def _create_graph_build_task(
        self,
        source_data: Dict,
        user_id: str,
        topic_name: str
    ) -> str:
        """
        Create a background processing task for chat batch.

        This function creates a GraphBuild record that will be picked up
        by the GraphBuildDaemon for asynchronous graph processing.

        Args:
            source_data: Source data object containing chat batch info
            user_id: User identifier
            topic_name: Topic name for memory categorization

        Returns:
            Generated build_id for the task

        Raises:
            Exception: If task creation fails
        """
        try:
            # Generate build_id for chat batch using the actual source link
            # The source_data contains the actual chat_link used to store the batch
            with self.SessionLocal() as db:
                source_record = db.query(SourceData).filter(SourceData.id == source_data["id"]).first()
                if not source_record:
                    raise Exception(f"Source data not found with id: {source_data['id']}")
                
                chat_batch_link = source_record.link
            # Use empty string for external_db_uri for local mode
            external_db_uri = ""
            build_id = _generate_build_id_for_chat_batch(
                chat_batch_link, external_db_uri
            )

            # Create task record in local database only
            # All task scheduling is centralized in local database
            with self.SessionLocal() as db:
                # Check if GraphBuild already exists for this build_id
                existing_build_status = (
                    db.query(GraphBuild)
                    .filter(
                        GraphBuild.build_id == build_id,
                        GraphBuild.topic_name == topic_name,
                        GraphBuild.external_database_uri == external_db_uri,
                    )
                    .first()
                )

                if existing_build_status:
                    logger.info(
                        f"Graph build task already exists for build_id: {build_id}, updating status to pending"
                    )
                    existing_build_status.status = "pending"
                    db.commit()
                    db.refresh(existing_build_status)
                    return build_id

                # Create new GraphBuild record
                build_status = GraphBuild(
                    topic_name=topic_name,
                    build_id=build_id,
                    external_database_uri=external_db_uri,
                    storage_directory=f"memory://{user_id}/chat_batch/",  # Virtual storage for memory
                    doc_link=chat_batch_link,
                    status="pending",
                )
                db.add(build_status)
                db.commit()

                logger.info(
                    f"Created graph build task for chat batch: {build_id} (user: {user_id})"
                )
                return build_id

        except Exception as e:
            logger.error(f"Failed to create graph build task for chat batch: {e}")
            raise

    def retrieve_user_memory(
        self,
        query: str,
        user_id: str,
        memory_types: List[str] = None,
        time_range: Optional[Dict] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Retrieve user memory based on query.

        Args:
            query: Search query
            user_id: User identifier
            memory_types: Types to search ["conversation", "insights"]
            time_range: Time range filter
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score for vector search (0-1)

        Returns:
            Dict with retrieved memory results
        """
        topic_name = generate_topic_name_for_personal_memory(user_id)

        if not memory_types:
            memory_types = ["conversation", "insights"]

        results = {}

        # Search conversation summaries
        if "conversation" in memory_types:
            conversations = self._search_conversation_summaries(
                query, user_id, topic_name, time_range, top_k, similarity_threshold
            )
            results["conversations"] = conversations

        # Search user insights
        if "insights" in memory_types:
            insights = self._search_user_insights(
                query, user_id, topic_name, time_range, top_k, similarity_threshold
            )
            results["insights"] = insights

        return {
            "query": query,
            "user_id": user_id,
            "topic_name": topic_name,
            "results": results,
            "total_found": sum(len(r) for r in results.values()),
        }

    def _search_conversation_summaries(
        self,
        query: str,
        user_id: str,
        topic_name: str,
        time_range: Optional[Dict],
        top_k: int,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search conversation summaries using vector similarity."""

        try:
            # Generate query embedding
            query_embedding = self.embedding_func(query)

            with self.SessionLocal() as db:
                # Build time filter conditions for SQL
                time_conditions = []
                params = {
                    "query_vector": str(query_embedding),
                    "topic_name": topic_name,
                    "knowledge_type": "chat_summary",
                    "top_k": top_k * 3,  # Fetch more initially for filtering
                }

                if time_range:
                    if time_range.get("start"):
                        time_conditions.append("AND kb.created_at >= :start_time")
                        params["start_time"] = time_range["start"]
                    if time_range.get("end"):
                        time_conditions.append("AND kb.created_at <= :end_time")
                        params["end_time"] = time_range["end"]

                time_filter_sql = " ".join(time_conditions)

                # Vector similarity search query
                base_query = f"""
                    SELECT 
                        kb.id,
                        kb.name,
                        kb.content,
                        kb.context,
                        kb.attributes,
                        kb.created_at,
                        VEC_COSINE_DISTANCE(kb.content_vec, :query_vector) as similarity_distance,
                        (1 - VEC_COSINE_DISTANCE(kb.content_vec, :query_vector)) as similarity_score
                    FROM knowledge_blocks kb
                    WHERE JSON_EXTRACT(kb.attributes, '$.topic_name') = :topic_name
                      AND kb.knowledge_type = :knowledge_type
                      AND kb.content_vec IS NOT NULL
                      {time_filter_sql}
                    ORDER BY similarity_distance ASC 
                    LIMIT :top_k
                """

                result = db.execute(text(base_query), params)
                rows = result.fetchall()

                # Convert to structured results and filter by similarity threshold
                results = []
                for row in rows:
                    similarity_score = round(1 - row.similarity_distance, 4)
                    if similarity_score >= similarity_threshold:
                        results.append(
                            {
                                "id": row.id,
                                "name": row.name,
                                "content": row.content,
                                "context": row.context,
                                "created_at": (
                                    row.created_at.isoformat()
                                    if row.created_at
                                    else None
                                ),
                                "attributes": row.attributes,
                                "similarity_score": similarity_score,
                            }
                        )

                # Sort by similarity score and return top k
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return results[:top_k]

        except Exception as e:
            logger.error(f"Error in conversation summary vector search: {str(e)}")
            # Fallback to basic text search if vector search fails
            raise e

    def _search_user_insights(
        self,
        query: str,
        user_id: str,
        topic_name: str,
        time_range: Optional[Dict],
        top_k: int,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search user insights using vector similarity on relationship triplets."""

        try:
            # Generate query embedding
            query_embedding = self.embedding_func(query)

            with self.SessionLocal() as db:
                # Build time filter conditions for SQL
                time_conditions = []
                params = {
                    "query_vector": str(query_embedding),
                    "topic_name": topic_name,
                    "top_k": top_k * 3,  # Fetch more initially for filtering
                }

                if time_range:
                    if time_range.get("start"):
                        time_conditions.append("AND r.created_at >= :start_time")
                        params["start_time"] = time_range["start"]
                    if time_range.get("end"):
                        time_conditions.append("AND r.created_at <= :end_time")
                        params["end_time"] = time_range["end"]

                time_filter_sql = " ".join(time_conditions)

                # Vector similarity search query for relationship triplets
                base_query = f"""
                    SELECT 
                        r.id,
                        e1.id as source_entity_id,
                        e1.name as source_entity_name,
                        e1.description as source_entity_description,
                        e1.attributes as source_entity_attributes,
                        r.relationship_desc,
                        e2.id as target_entity_id,
                        e2.name as target_entity_name,
                        e2.description as target_entity_description,
                        e2.attributes as target_entity_attributes,
                        r.attributes,
                        r.created_at,
                        VEC_COSINE_DISTANCE(r.relationship_desc_vec, :query_vector) as similarity_distance,
                        (1 - VEC_COSINE_DISTANCE(r.relationship_desc_vec, :query_vector)) as similarity_score
                    FROM relationships r
                    JOIN entities e1 ON r.source_entity_id = e1.id
                    JOIN entities e2 ON r.target_entity_id = e2.id
                    WHERE JSON_EXTRACT(r.attributes, '$.topic_name') = :topic_name
                      AND r.relationship_desc_vec IS NOT NULL
                      {time_filter_sql}
                    ORDER BY similarity_distance ASC 
                    LIMIT :top_k
                """

                result = db.execute(text(base_query), params)
                rows = result.fetchall()

                # Convert to structured results and filter by similarity threshold
                results = []
                for row in rows:
                    similarity_score = round(1 - row.similarity_distance, 4)
                    if similarity_score >= similarity_threshold:
                        results.append(
                            {
                                "id": row.id,
                                "name": f"{row.source_entity_name} -> {row.target_entity_name}",
                                "description": f"{row.source_entity_name} {row.relationship_desc} {row.target_entity_name}",
                                "source_entity": {
                                    "id": row.source_entity_id,
                                    "name": row.source_entity_name,
                                    "description": row.source_entity_description,
                                    "attributes": row.source_entity_attributes,
                                },
                                "target_entity": {
                                    "id": row.target_entity_id,
                                    "name": row.target_entity_name,
                                    "description": row.target_entity_description,
                                    "attributes": row.target_entity_attributes,
                                },
                                "relationship_desc": row.relationship_desc,
                                "created_at": (
                                    row.created_at.isoformat()
                                    if row.created_at
                                    else None
                                ),
                                "attributes": row.attributes,
                                "similarity_score": similarity_score,
                            }
                        )

                # Sort by similarity score and return top k
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return results[:top_k]

        except Exception as e:
            logger.error(f"Error in user insights vector search: {str(e)}")
            # Fallback to basic text search if vector search fails
            raise e
