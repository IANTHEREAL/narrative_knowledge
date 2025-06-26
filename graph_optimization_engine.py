"""
Graph Optimization Engine - High-Quality Modular Design

A modular and extensible knowledge graph optimization system that identifies
and resolves quality issues in knowledge graphs through AI-powered analysis.

Key Features:
- Pluggable graph data providers
- Configurable optimization strategies
- Concurrent issue processing
- Comprehensive error handling and monitoring
- Clean separation of concerns
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os

from knowledge_graph.query import search_relationships_by_vector_similarity
from setting.db import db_manager
from opt.helper import GRAPH_OPTIMIZATION_ACTION_SYSTEM_PROMPT_WO_MR, extract_issues
from opt.evaluator import batch_evaluate_issues, Issue
from llm.factory import LLMInterface
from knowledge_graph.models import Entity, Relationship, SourceGraphMapping
from opt.optimizer import (
    process_entity_quality_issue,
    process_redundancy_entity_issue,
    process_relationship_quality_issue,
    process_redundancy_relationship_issue,
)

logger = logging.getLogger(__name__)


# ================== Configuration Management ==================


@dataclass
class LLMConfig:
    """LLM configuration for optimization and critique"""

    optimization_provider: str = "openai_like"
    optimization_model: str = "graph_optimization_14b"
    critique_provider: str = "bedrock"
    critique_model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    max_tokens: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Processing configuration for the optimization engine"""

    max_concurrent_issues: int = 3
    confidence_threshold: float = 0.9
    similarity_threshold: float = 0.3
    top_k_retrieval: int = 30
    state_file_path: str = "optimization_state.pkl"
    max_retries: int = 3


@dataclass
class OptimizationConfig:
    """Main configuration for the graph optimization engine"""

    database_uri: Optional[str] = None
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)

    def __post_init__(self):
        if self.database_uri is None:
            self.database_uri = os.getenv("GRAPH_DATABASE_URI")


# ================== Data Abstractions ==================


class GraphData:
    """Container for graph data with entities and relationships"""

    def __init__(self, entities: List[Dict], relationships: List[Dict]):
        self.entities = entities
        self.relationships = relationships

    def to_dict(self) -> Dict:
        return {"entities": self.entities, "relationships": self.relationships}

    def __len__(self) -> int:
        return len(self.entities) + len(self.relationships)


class GraphDataProvider(ABC):
    """Abstract base class for graph data providers"""

    @abstractmethod
    def retrieve_graph_data(self, **kwargs) -> GraphData:
        """Retrieve graph data based on provider-specific parameters"""
        pass


class VectorSearchGraphProvider(GraphDataProvider):
    """Graph data provider using vector similarity search"""

    def __init__(self, database_uri: str, similarity_threshold: float = 0.3):
        self.database_uri = database_uri
        self.similarity_threshold = similarity_threshold

    def retrieve_graph_data(self, query: str, top_k: int = 30, **kwargs) -> GraphData:
        """Retrieve graph data using vector similarity search"""
        try:
            res = search_relationships_by_vector_similarity(
                query,
                similarity_threshold=self.similarity_threshold,
                top_k=top_k,
                database_uri=self.database_uri,
            )

            entities = {}
            relationships = {}

            for index, row in res.iterrows():
                entities[row["source_entity_id"]] = {
                    "id": row["source_entity_id"],
                    "name": row["source_entity_name"],
                    "description": row["source_entity_description"],
                    "attributes": row["source_entity_attributes"],
                }
                entities[row["target_entity_id"]] = {
                    "id": row["target_entity_id"],
                    "name": row["target_entity_name"],
                    "description": row["target_entity_description"],
                    "attributes": row["target_entity_attributes"],
                }
                relationships[row["id"]] = {
                    "id": row["id"],
                    "source_entity": row["source_entity_name"],
                    "target_entity": row["target_entity_name"],
                    "description": row["relationship_desc"],
                    "attributes": row["attributes"],
                }

            return GraphData(
                entities=list(entities.values()),
                relationships=list(relationships.values()),
            )
        except Exception as e:
            logger.error(f"Error retrieving graph data: {e}")
            raise


# ================== Issue Management ==================


class IssueKey:
    """Utility for generating consistent issue keys"""

    @staticmethod
    def generate(issue: Dict) -> Tuple[str, tuple]:
        """Generate a unique key for an issue based on its type and affected IDs"""
        return (issue["issue_type"], tuple(sorted(issue["affected_ids"])))


class IssueDetector:
    """Detects quality issues in graph data using LLM analysis"""

    def __init__(self, llm_client: LLMInterface):
        self.llm_client = llm_client

    def detect_issues(
        self, graph_data: GraphData, analysis_context: str = "graph quality analysis"
    ) -> List[Issue]:
        """Detect quality issues in the provided graph data"""
        try:
            prompt = (
                GRAPH_OPTIMIZATION_ACTION_SYSTEM_PROMPT_WO_MR
                + " Now Optimize the following graph:\n"
                + json.dumps(graph_data.to_dict(), indent=2, ensure_ascii=False)
            )

            response = self.llm_client.generate(prompt)
            analysis_list = extract_issues(response)

            issues = []
            for analysis in analysis_list.values():
                for issue_data in analysis:
                    issue = Issue(
                        issue_type=issue_data["issue_type"],
                        affected_ids=issue_data["affected_ids"],
                        reasoning=issue_data["reasoning"],
                        source_graph=graph_data.to_dict(),
                        analysis_context=analysis_context,
                    )
                    issues.append(issue)

            logger.info(f"Detected {len(issues)} potential issues")
            return issues

        except Exception as e:
            logger.error(f"Error detecting issues: {e}")
            raise


class IssueEvaluator:
    """Evaluates and validates detected issues using critic LLMs"""

    def __init__(self, critic_clients: Dict[str, LLMInterface]):
        self.critic_clients = critic_clients

    def evaluate_issues(self, issues: List[Issue]) -> List[Issue]:
        """Evaluate issues using critic LLMs and update validation scores"""
        try:
            return batch_evaluate_issues(self.critic_clients, issues)
        except Exception as e:
            logger.error(f"Error evaluating issues: {e}")
            raise


# ================== Issue Processing ==================


class IssueProcessor:
    """Processes and resolves different types of quality issues"""

    def __init__(
        self,
        session_factory,
        llm_client: LLMInterface,
        models: Dict[str, Any],
        max_concurrent_issues: int = 1,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.models = models
        self.max_concurrent_issues = max_concurrent_issues

    def process_issues_list(self, issues: List[Issue]) -> int:
        """
        Modern interface to process a list of Issue objects directly.

        Args:
            issues: List of Issue objects to process

        Returns:
            Number of successfully resolved issues
        """
        if not issues:
            return 0

        # Group issues by type for efficient processing
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.issue_type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        total_resolved = 0

        # Process each issue type
        for issue_type, type_issues in issues_by_type.items():
            logger.info(f"Processing {len(type_issues)} {issue_type} issues")

            try:
                resolved_count = self._process_issue_type_modern(
                    issue_type, type_issues
                )
                total_resolved += resolved_count
                logger.info(
                    f"Successfully processed {resolved_count}/{len(type_issues)} {issue_type} issues"
                )

            except Exception as e:
                logger.error(f"Error processing {issue_type} issues: {e}")

        return total_resolved

    def _process_issue_type_modern(self, issue_type: str, issues: List[Issue]) -> int:
        """Process issues of a specific type using modern Issue objects"""

        # Get the appropriate processing function and arguments
        if issue_type == "entity_quality_issue":
            process_func = process_entity_quality_issue
            base_args = [
                self.session_factory,
                self.llm_client,
                self.models["Entity"],
                self.models["Relationship"],
            ]
        elif issue_type == "redundancy_entity":
            process_func = process_redundancy_entity_issue
            base_args = [
                self.session_factory,
                self.llm_client,
                self.models["Entity"],
                self.models["Relationship"],
                self.models["SourceGraphMapping"],
            ]
        elif issue_type == "relationship_quality_issue":
            process_func = process_relationship_quality_issue
            base_args = [
                self.session_factory,
                self.llm_client,
                self.models["Relationship"],
            ]
        elif issue_type == "redundancy_relationship":
            process_func = process_redundancy_relationship_issue
            base_args = [
                self.session_factory,
                self.llm_client,
                self.models["Relationship"],
                self.models["SourceGraphMapping"],
            ]
        else:
            logger.warning(f"Unknown issue type: {issue_type}")
            return 0

        resolved_count = 0

        # Process issues with concurrency control
        batch_size = min(self.max_concurrent_issues, len(issues))

        for i in range(0, len(issues), batch_size):
            batch = issues[i : i + batch_size]

            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {}

                for issue in batch:
                    # Generate a meaningful issue key for logging and tracking
                    issue_key = IssueKey.generate(
                        {
                            "issue_type": issue.issue_type,
                            "affected_ids": issue.affected_ids,
                        }
                    )

                    # Prepare arguments for the processing function
                    args = base_args.copy()
                    args.extend([issue_key, issue.to_dict()])  # row_key, issue_dict

                    futures[executor.submit(process_func, *args)] = issue

                # Collect results
                for future in as_completed(futures):
                    issue = futures[future]
                    try:
                        success = future.result()
                        if success:
                            issue.is_resolved = True
                            resolved_count += 1
                            logger.info(
                                f"Successfully processed {issue_type} issue for {issue.affected_ids}"
                            )
                        else:
                            logger.warning(
                                f"Failed to process {issue_type} issue for {issue.affected_ids}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing {issue_type} issue {issue.affected_ids}: {e}"
                        )

        return resolved_count


# ================== State Management ==================


class OptimizationState:
    """Manages optimization state persistence and recovery with efficient duplicate detection"""

    def __init__(self, state_file_path: str):
        self.state_file_path = state_file_path
        self.issues: List[Issue] = []
        self.issue_keys: set = set()
        self._load_state()

    def _load_state(self):
        """Load optimization state from file and build issue keys cache"""
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.issues = [Issue.from_dict(issue_data) for issue_data in data]
        else:
            self.issues = []

        # Build issue keys cache
        self._rebuild_issue_keys_cache()
        logger.info(
            f"Loaded {len(self.issues)} issues with {len(self.issue_keys)} unique keys"
        )

    def _rebuild_issue_keys_cache(self):
        """Rebuild the issue keys cache from current issues"""
        self.issue_keys = set()
        for issue in self.issues:
            issue_key = IssueKey.generate(
                {"issue_type": issue.issue_type, "affected_ids": issue.affected_ids}
            )
            self.issue_keys.add(issue_key)

    def get_issues(self) -> List[Issue]:
        """Get all current issues"""
        return self.issues.copy()

    def has_issue(self, issue: Issue) -> bool:
        """Check if an issue already exists based on its key"""
        issue_key = IssueKey.generate(
            {"issue_type": issue.issue_type, "affected_ids": issue.affected_ids}
        )
        return issue_key in self.issue_keys

    def add_unique_issues(self, new_issues: List[Issue]) -> List[Issue]:
        """Add only unique issues that don't already exist

        Args:
            new_issues: List of potentially new issues

        Returns:
            List of issues that were actually added (unique ones)
        """
        added_issues = []
        duplicates_count = 0

        for issue in new_issues:
            if not self.has_issue(issue):
                self.issues.append(issue)
                issue_key = IssueKey.generate(
                    {"issue_type": issue.issue_type, "affected_ids": issue.affected_ids}
                )
                self.issue_keys.add(issue_key)
                added_issues.append(issue)
            else:
                duplicates_count += 1
                logger.debug(
                    f"Skipping duplicate issue: {issue.issue_type} for {issue.affected_ids}"
                )

        if duplicates_count > 0:
            logger.info(
                f"Added {len(added_issues)} unique issues, filtered {duplicates_count} duplicates"
            )
        elif added_issues:
            logger.info(f"Added {len(added_issues)} new unique issues")

        return added_issues

    def update_issues(self, updated_issues: List[Issue]):
        """Update the entire issues list and rebuild cache"""
        self.issues = updated_issues
        self._rebuild_issue_keys_cache()

    def save_state(self):
        """Save optimization state to file"""
        data = [issue.to_dict() for issue in self.issues]
        with open(self.state_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_state(self):
        """Clear optimization state file and memory"""
        if os.path.exists(self.state_file_path):
            os.remove(self.state_file_path)
        self.issues = []
        self.issue_keys = set()

    def get_stats(self) -> Dict[str, int]:
        """Get basic statistics about current state"""
        return {
            "total_issues": len(self.issues),
            "unique_keys": len(self.issue_keys),
            "resolved_issues": len(
                [issue for issue in self.issues if issue.is_resolved]
            ),
        }

    def get_optimization_stats(
        self, confidence_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        validated_issues = [
            issue
            for issue in self.issues
            if issue.validation_score >= confidence_threshold
        ]
        resolved_issues = [issue for issue in self.issues if issue.is_resolved]

        return {
            "total_issues": len(self.issues),
            "unique_keys": len(self.issue_keys),
            "validated_issues": len(validated_issues),
            "resolved_issues": len(resolved_issues),
            "resolution_rate": (
                len(resolved_issues) / len(validated_issues) if validated_issues else 0
            ),
            "issues_by_type": self._calculate_issue_type_stats(confidence_threshold),
        }

    def _calculate_issue_type_stats(
        self, confidence_threshold: float = 0.9
    ) -> Dict[str, Dict[str, int]]:
        """Calculate statistics by issue type"""
        stats = {}
        for issue_type in [
            "entity_quality_issue",
            "redundancy_entity",
            "relationship_quality_issue",
            "redundancy_relationship",
        ]:
            type_issues = [
                issue for issue in self.issues if issue.issue_type == issue_type
            ]
            stats[issue_type] = {
                "detected": len(type_issues),
                "validated": len(
                    [
                        issue
                        for issue in type_issues
                        if issue.validation_score >= confidence_threshold
                    ]
                ),
                "resolved": len([issue for issue in type_issues if issue.is_resolved]),
            }
        return stats

    def get_current_status_summary(self, confidence_threshold: float = 0.9) -> str:
        """Get a human-readable status summary"""
        stats = self.get_optimization_stats(confidence_threshold)
        return (
            f"{stats['total_issues']} total issues "
            f"({stats['validated_issues']} validated, {stats['resolved_issues']} resolved)"
        )


# ================== Main Optimization Engine ==================


class GraphOptimizationEngine:
    """
    Main graph optimization engine that orchestrates the entire optimization process.

    This engine follows a multi-stage pipeline:
    1. Graph data retrieval
    2. Issue detection
    3. Issue evaluation and validation
    4. Issue processing and resolution
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all engine components based on configuration"""
        # Initialize LLM clients
        self.optimization_llm = LLMInterface(
            self.config.llm_config.optimization_provider,
            self.config.llm_config.optimization_model,
        )

        self.critic_llm = LLMInterface(
            self.config.llm_config.critique_provider,
            self.config.llm_config.critique_model,
        )

        self.critic_clients = {"llm-critic": self.critic_llm}

        # Initialize database session factory
        self.session_factory = db_manager.get_session_factory(self.config.database_uri)

        # Initialize data provider
        self.graph_provider = VectorSearchGraphProvider(
            self.config.database_uri, self.config.processing_config.similarity_threshold
        )

        # Initialize core components
        self.issue_detector = IssueDetector(self.optimization_llm)
        self.issue_evaluator = IssueEvaluator(self.critic_clients)
        self.issue_processor = IssueProcessor(
            self.session_factory,
            self.critic_llm,
            {
                "Entity": Entity,
                "Relationship": Relationship,
                "SourceGraphMapping": SourceGraphMapping,
            },
            self.config.processing_config.max_concurrent_issues,
        )

        # Initialize state management
        self.state_manager = OptimizationState(
            self.config.processing_config.state_file_path
        )

        logger.info("Graph optimization engine initialized successfully")

    def set_graph_provider(self, provider: GraphDataProvider):
        """Set a custom graph data provider"""
        self.graph_provider = provider
        logger.info(f"Graph provider updated to: {type(provider).__name__}")

    def optimize_graph(self, **provider_kwargs) -> Dict[str, Any]:
        """
        Main optimization method that processes a graph through the full pipeline.

        Args:
            **provider_kwargs: Arguments passed to the graph data provider

        Returns:
            Dictionary containing optimization results and statistics
        """
        logger.info("Starting graph optimization process")
        stats = {
            "issues_detected": 0,
            "issues_validated": 0,
            "issues_resolved": 0,
            "issues_by_type": {},
        }

        try:
            # Get initial status from state manager
            initial_stats = self.state_manager.get_optimization_stats(
                self.config.processing_config.confidence_threshold
            )
            stats.update(initial_stats)

            logger.info(
                f"Starting optimization: {self.state_manager.get_current_status_summary(self.config.processing_config.confidence_threshold)}"
            )

            # Stage 1: Issue Detection (if needed)
            issues = self.state_manager.get_issues()
            if self._should_detect_new_issues(issues):
                new_issues = self._detect_new_issues(**provider_kwargs)
                if new_issues:
                    # Add only unique issues using the state manager
                    added_issues = self.state_manager.add_unique_issues(new_issues)
                    stats["issues_detected"] = len(added_issues)
                    if added_issues:
                        logger.info(
                            f"Successfully added {len(added_issues)} new unique issues"
                        )
                    else:
                        logger.info(
                            "No new unique issues detected (all were duplicates)"
                        )
                else:
                    logger.info("No new issues detected")
            else:
                logger.info(
                    "Skipping new issue detection (existing issues need processing)"
                )

            # Log current status
            logger.info(
                f"After detection: {self.state_manager.get_current_status_summary(self.config.processing_config.confidence_threshold)}"
            )

            # Stage 2: Issue Evaluation
            issues = self.state_manager.get_issues()
            updated_issues = self._evaluate_issues(issues)
            self.state_manager.update_issues(updated_issues)

            # Log evaluation progress
            logger.info(
                f"After evaluation: {self.state_manager.get_current_status_summary(self.config.processing_config.confidence_threshold)}"
            )

            # Stage 3: Issue Processing
            final_issues = self.state_manager.get_issues()
            resolved_count = self._process_issues(final_issues)

            # Log final processing results
            logger.info(
                f"After processing: {self.state_manager.get_current_status_summary(self.config.processing_config.confidence_threshold)}"
            )

            # Get final statistics from state manager
            final_stats = self.state_manager.get_optimization_stats(
                self.config.processing_config.confidence_threshold
            )
            stats.update(final_stats)

            # Save final state
            self.state_manager.save_state()

            # Log completion summary
            final_summary = self.state_manager.get_current_status_summary(
                self.config.processing_config.confidence_threshold
            )
            logger.info(f"Optimization completed. Final status: {final_summary}")

            return stats

        except Exception as e:
            logger.error(f"Error during graph optimization: {e}")
            raise

    def _should_detect_new_issues(self, issues: List[Issue]) -> bool:
        """Determine if new issue detection is needed"""
        if len(issues) == 0:
            return True

        unresolved_high_confidence = [
            issue
            for issue in issues
            if not issue.is_resolved
            and issue.validation_score
            >= self.config.processing_config.confidence_threshold
        ]

        all_evaluated = all(len(issue.critic_evaluations) > 0 for issue in issues)

        return len(unresolved_high_confidence) == 0 and all_evaluated

    def _detect_new_issues(self, **provider_kwargs) -> List[Issue]:
        """Detect new issues using the configured graph provider"""
        logger.info("Detecting new issues...")

        # Retrieve graph data
        graph_data = self.graph_provider.retrieve_graph_data(**provider_kwargs)
        logger.info(f"Retrieved graph data with {len(graph_data)} elements")

        # Detect issues
        analysis_context = provider_kwargs.get("query", "graph quality analysis")
        return self.issue_detector.detect_issues(graph_data, analysis_context)

    def _evaluate_issues(self, issues: List[Issue]) -> List[Issue]:
        """Evaluate issues that haven't been critiqued yet"""
        if len(issues) == 0:
            return issues

        # Keep evaluating until all issues have been critiqued
        while True:
            unevaluated = any(len(issue.critic_evaluations) == 0 for issue in issues)

            if not unevaluated:
                break

            logger.info("Evaluating issues...")
            issues = self.issue_evaluator.evaluate_issues(issues)
            self.state_manager.update_issues(issues)
            self.state_manager.save_state()

        return issues

    def _process_issues(self, issues: List[Issue]) -> int:
        """Process all validated issues using the IssueProcessor"""
        logger.info("Processing validated issues...")

        # Filter issues that meet confidence threshold and are not yet resolved
        issues_to_process = [
            issue
            for issue in issues
            if (
                issue.validation_score
                >= self.config.processing_config.confidence_threshold
                and not issue.is_resolved
            )
        ]

        if not issues_to_process:
            logger.info("No issues to process")
            return 0

        logger.info(f"Processing {len(issues_to_process)} validated issues")

        # Use the modern IssueProcessor interface
        return self.issue_processor.process_issues_list(issues_to_process)

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics"""
        optimization_stats = self.state_manager.get_optimization_stats(
            self.config.processing_config.confidence_threshold
        )

        # Add config to the status
        optimization_stats["config"] = self.config
        return optimization_stats

    def reset_optimization_state(self):
        """Reset optimization state and clear cache"""
        self.state_manager.clear_state()
        logger.info("Optimization state reset")


# ================== Factory Functions ==================


def create_optimization_engine(
    config: Optional[OptimizationConfig] = None,
) -> GraphOptimizationEngine:
    """Factory function to create a configured optimization engine"""
    if config is None:
        config = OptimizationConfig()

    return GraphOptimizationEngine(config)


def create_vector_search_engine(
    database_uri: Optional[str] = None,
    similarity_threshold: float = 0.3,
    **config_kwargs,
) -> GraphOptimizationEngine:
    """Factory function to create an engine with vector search provider"""
    config = OptimizationConfig(database_uri=database_uri, **config_kwargs)
    config.processing_config.similarity_threshold = similarity_threshold

    return GraphOptimizationEngine(config)
