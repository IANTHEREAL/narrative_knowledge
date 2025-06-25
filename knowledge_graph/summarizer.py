import json
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge_graph.models import DocumentSummary
from utils.json_utils import robust_json_parse
from setting.db import SessionLocal
from llm.factory import LLMInterface

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Generates topic-focused summaries of documents for efficient knowledge graph construction.
    Summaries are cached in database to avoid recomputation.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        session_factory=None,
        worker_count: int = 3,
    ):
        """
        Initialize the document summarizer.

        Args:
            llm_client: LLM interface for generating summaries
            session_factory: Database session factory. If None, uses default SessionLocal.
        """
        self.llm_client = llm_client
        self.SessionLocal = session_factory or SessionLocal
        self.worker_count = worker_count

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
                # Re-query the existing summary in the current session to ensure it's attached
                current_summary = (
                    db.query(DocumentSummary)
                    .filter(
                        DocumentSummary.document_id == document["source_id"],
                        DocumentSummary.topic_name == topic_name,
                    )
                    .first()
                )

                if current_summary:
                    # Update existing summary
                    current_summary.summary_content = summary_data["summary_content"]
                    current_summary.key_entities = summary_data["key_entities"]
                    current_summary.main_themes = summary_data["main_themes"]
                    current_summary.business_context = summary_data["business_context"]
                    current_summary.document_type = summary_data["document_type"]
                    db.commit()
                    db.refresh(current_summary)
                    return {
                        "source_id": current_summary.document_id,
                        "source_name": f"Summary of {current_summary.source_data.name if current_summary.source_data else 'Unknown'}",
                        "source_content": self._format_summary_content(current_summary),
                    }

            # Create new summary (either no existing summary or re-query failed)
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
        Generate summaries for a batch of documents in parallel.

        Args:
            topic_name: Topic to focus summaries on
            documents: List of document dicts
            force_regenerate: Whether to regenerate existing summaries

        Returns:
            List of DocumentSummary objects for successfully processed documents
        """
        if not documents:
            return []

        summaries = []
        errors = []

        def process_document(doc_with_index):
            """Worker function to process a single document."""
            index, doc = doc_with_index
            try:
                summary = self.get_or_create_summary(topic_name, doc, force_regenerate)
                return index, summary, None
            except Exception as e:
                error_msg = f"Failed to generate summary for {doc['source_name']}: {e}"
                logger.error(error_msg, exc_info=True)
                return index, None, error_msg

        # Create list of documents with their original indices to maintain order
        indexed_documents = list(enumerate(documents))

        logger.info(
            f"Starting parallel summarization of {len(documents)} documents using {self.worker_count} workers"
        )

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(process_document, doc_with_index): doc_with_index[1]
                for doc_with_index in indexed_documents
            }

            # Collect results as they complete
            results = [None] * len(documents)  # Pre-allocate to maintain order
            completed_count = 0

            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                completed_count += 1
                try:
                    index, summary, error = future.result()
                    if error:
                        errors.append(error)
                        logger.warning(
                            f"Document processing failed ({completed_count}/{len(documents)}): {doc['source_name']}"
                        )
                    else:
                        results[index] = summary
                        logger.info(
                            f"Document processed successfully ({completed_count}/{len(documents)}): {doc['source_name']}"
                        )
                except Exception as e:
                    error_msg = f"Unexpected error processing {doc['source_name']}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

        # Filter out None values (failed documents) while maintaining order
        summaries = [result for result in results if result is not None]

        # Log final results
        success_count = len(summaries)
        failure_count = len(errors)

        if errors:
            logger.warning(
                f"Batch summarization completed with {failure_count} failures:"
            )
            for error in errors:
                logger.warning(f"  - {error}")

            raise RuntimeError(
                f"Failed to generate summaries for {failure_count}/{len(documents)} documents"
            )

        logger.info(f"Generated {success_count} summaries for topic: {topic_name}")
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
            f"Document: {document['source_name']}\n\n{document['source_content']}\n\n"
            f"Document attributes: {document['source_attributes']}"
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

            # Use robust JSON parsing with escape error fixing and LLM fallback
            summary_data = robust_json_parse(response, "object", self.llm_client)

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

        except Exception as e:
            logger.error(
                f"Error generating summary: {e}",
                exc_info=True,
            )
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
