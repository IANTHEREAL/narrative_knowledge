import json
import logging
from typing import Dict, List

from knowledge_graph.models import DocumentSummary
from utils.json_utils import extract_json
from setting.db import SessionLocal
from llm.factory import LLMInterface

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Generates topic-focused summaries of documents for efficient knowledge graph construction.
    Summaries are cached in database to avoid recomputation.
    """

    def __init__(self, llm_client: LLMInterface, session_factory=None):
        """
        Initialize the document summarizer.

        Args:
            llm_client: LLM interface for generating summaries
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.SessionLocal = session_factory or SessionLocal

    def get_or_create_summary(
        self, topic_name: str, document: Dict, force_regenerate: bool = False
    ) -> DocumentSummary:
        """
        Get existing summary or create a new one for the given document and topic.

        Args:
            topic_name: Topic to focus the summary on
            document: Document dict with source_id, source_name, source_content, etc.
            force_regenerate: Whether to regenerate existing summary

        Returns:
            DocumentSummary object
        """
        with self.SessionLocal() as db:
            # Check for existing summary
            logger.info(
                f"Checking for existing summary for document: {document['source_name']}"
            )
            existing_summary = (
                db.query(DocumentSummary)
                .filter(
                    DocumentSummary.document_id == document["source_id"],
                    DocumentSummary.topic_name == topic_name,
                )
                .first()
            )

            if existing_summary and not force_regenerate:
                logger.info(
                    f"Using cached summary for document: {document['source_name']}"
                )
                return {
                    "source_id": existing_summary.document_id,
                    "source_name": f"Summary of {existing_summary.source_data.name if existing_summary.source_data else 'Unknown'}",
                    "source_content": self._format_summary_content(existing_summary),
                }

        try:
            # Generate new summary
            logger.info(f"Generating summary for document: {document['source_name']}")
            summary_data = self._generate_document_summary(topic_name, document)
        except Exception as e:
            logger.error(
                f"Error generating summary for document: {document['source_name']}: {e}",
                exc_info=True,
            )
            raise e

        with self.SessionLocal() as db:

            if existing_summary:
                # Update existing summary
                existing_summary.summary_content = summary_data["summary_content"]
                existing_summary.key_entities = summary_data["key_entities"]
                existing_summary.main_themes = summary_data["main_themes"]
                existing_summary.business_context = summary_data["business_context"]
                existing_summary.document_type = summary_data["document_type"]
                db.commit()
                db.refresh(existing_summary)
                return {
                    "source_id": existing_summary.document_id,
                    "source_name": f"Summary of {existing_summary.source_data.name if existing_summary.source_data else 'Unknown'}",
                    "source_content": self._format_summary_content(existing_summary),
                }
            else:
                # Create new summary
                summary = DocumentSummary(
                    document_id=document["source_id"],
                    topic_name=topic_name,
                    summary_content=summary_data["summary_content"],
                    key_entities=summary_data["key_entities"],
                    main_themes=summary_data["main_themes"],
                    business_context=summary_data["business_context"],
                    document_type=summary_data["document_type"],
                )
                db.add(summary)
                db.commit()
                db.refresh(summary)

            return {
                "source_id": summary.document_id,
                "source_name": f"Summary of {summary.source_data.name if summary.source_data else 'Unknown'}",
                "source_content": self._format_summary_content(summary),
            }

    def batch_summarize_documents(
        self, topic_name: str, documents: List[Dict], force_regenerate: bool = False
    ) -> List[DocumentSummary]:
        """
        Generate summaries for a batch of documents.

        Args:
            topic_name: Topic to focus summaries on
            documents: List of document dicts
            force_regenerate: Whether to regenerate existing summaries

        Returns:
            List of DocumentSummary objects
        """
        summaries = []
        for doc in documents:
            try:
                summary = self.get_or_create_summary(topic_name, doc, force_regenerate)
                summaries.append(summary)
            except Exception as e:
                logger.error(
                    f"Failed to generate summary for {doc['source_name']}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Failed to generate summary for {doc['source_name']}: {e}"
                )

        logger.info(f"Generated {len(summaries)} summaries for topic: {topic_name}")
        return summaries

    def _generate_document_summary(self, topic_name: str, document: Dict) -> Dict:
        """
        Generate a topic-focused summary for a single document using LLM.

        Args:
            topic_name: Topic to focus the summary on
            document: Document dict with source_content

        Returns:
            Dict with summary data
        """
        # Use document source content directly
        doc_content = (
            f"Document: {document['source_name']}\n\n{document['source_content']}"
        )

        # Generate summary prompt
        summary_prompt = f"""Analyze this document in the context of "{topic_name}" and generate a comprehensive summary.

<document>
{doc_content}
</document>

Focus on aspects relevant to {topic_name} and provide a structured summary in JSON format:

```json
{{
    "summary_content": "2-3 paragraph summary focusing on {topic_name} aspects",
    "key_entities": ["entity1", "entity2", "entity3"],
    "main_themes": ["theme1", "theme2", "theme3"], 
    "business_context": "Why this document is important for {topic_name}",
    "document_type": "meeting_notes|technical_spec|proposal|report|other",
}}
```

Focus on:
- Key entities and stakeholders mentioned
- Main themes and topics relevant to {topic_name}
- Business context and significance
- Important relationships and dependencies
- Decision points and outcomes

Return only the JSON, no other text."""

        try:
            response = self.llm_client.generate(summary_prompt)

            # Extract and clean JSON
            json_str = extract_json(response)
            json_str = "".join(
                char for char in json_str if ord(char) >= 32 or char in "\r\t"
            )

            summary_data = json.loads(json_str)

            # Validate required fields
            required_fields = ["summary_content", "key_entities", "main_themes"]
            for field in required_fields:
                if field not in summary_data:
                    summary_data[field] = (
                        []
                        if field != "summary_content"
                        else "Summary generation failed"
                    )

            # Set defaults for optional fields
            summary_data.setdefault("business_context", "")
            summary_data.setdefault("document_type", "unknown")

            return summary_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            raise e

    def get_summaries_for_topic(self, topic_name: str) -> List[DocumentSummary]:
        """
        Retrieve all existing summaries for a given topic.

        Args:
            topic_name: Topic name to filter by

        Returns:
            List of DocumentSummary objects
        """
        with SessionLocal() as db:
            return (
                db.query(DocumentSummary)
                .filter(DocumentSummary.topic_name == topic_name)
                .order_by(DocumentSummary.created_at.desc())
                .all()
            )

    def _format_summary_content(self, summary: DocumentSummary) -> str:
        """
        Format document summary into structured content for blueprint generation.

        Args:
            summary: DocumentSummary object

        Returns:
            Formatted summary content string
        """
        content = f"Summary: {summary.summary_content}\n\n"

        if summary.key_entities:
            content += f"Key Entities: {', '.join(summary.key_entities)}\n\n"

        if summary.main_themes:
            content += f"Main Themes: {', '.join(summary.main_themes)}\n\n"

        if summary.business_context:
            content += f"Business Context: {summary.business_context}\n\n"

        content += f"Document Type: {summary.document_type or 'unknown'}\n"

        return content
