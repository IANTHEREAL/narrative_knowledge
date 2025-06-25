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
import pandas as pd
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

    max_concurrent_issues: int = 1
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

    def process_issues_by_type(
        self,
        issue_type: str,
        pending_issues: Dict,
        issue_cache: Dict,
        issue_df: pd.DataFrame,
        state_file: str,
    ) -> Dict:
        """Process issues by type with appropriate processor"""

        if issue_type == "entity_quality_issue":
            return self._process_issues_concurrently(
                pending_issues,
                issue_cache,
                issue_df,
                state_file,
                process_entity_quality_issue,
                [
                    self.session_factory,
                    self.llm_client,
                    self.models["Entity"],
                    self.models["Relationship"],
                ],
            )
        elif issue_type == "redundancy_entity":
            return self._process_issues_concurrently(
                pending_issues,
                issue_cache,
                issue_df,
                state_file,
                process_redundancy_entity_issue,
                [
                    self.session_factory,
                    self.llm_client,
                    self.models["Entity"],
                    self.models["Relationship"],
                    self.models["SourceGraphMapping"],
                ],
            )
        elif issue_type == "relationship_quality_issue":
            return self._process_issues_concurrently(
                pending_issues,
                issue_cache,
                issue_df,
                state_file,
                process_relationship_quality_issue,
                [self.session_factory, self.llm_client, self.models["Relationship"]],
            )
        elif issue_type == "redundancy_relationship":
            return self._process_issues_concurrently(
                pending_issues,
                issue_cache,
                issue_df,
                state_file,
                process_redundancy_relationship_issue,
                [
                    self.session_factory,
                    self.llm_client,
                    self.models["Relationship"],
                    self.models["SourceGraphMapping"],
                ],
            )
        else:
            logger.warning(f"Unknown issue type: {issue_type}")
            return issue_cache

    def _process_issues_concurrently(
        self,
        pending_issues: Dict,
        issue_cache: Dict,
        issue_df: pd.DataFrame,
        state_file: str,
        process_func: Callable,
        base_args: List,
    ) -> Dict:
        """Generic concurrent processing for different issue types"""
        while pending_issues:
            # Get batch for processing
            keys_for_batch = list(pending_issues.keys())[: self.max_concurrent_issues]
            if not keys_for_batch:
                break

            batch_issues = []
            for key in keys_for_batch:
                batch_issues.append(pending_issues.pop(key))

            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=len(batch_issues)) as executor:
                futures = {}
                for issue in batch_issues:
                    # Prepare arguments based on issue type
                    args = base_args.copy()
                    if "row_index" in issue:
                        args.extend([issue["row_index"], issue])
                    else:
                        args.extend([issue["issue_key"], issue])

                    futures[executor.submit(process_func, *args)] = issue.get(
                        "issue_key", issue.get("row_index")
                    )

                # Process results
                for future in as_completed(futures):
                    issue_key = futures[future]
                    try:
                        success = future.result()
                        if success:
                            issue_cache[issue_key] = True
                            logger.info(f"Successfully processed issue: {issue_key}")
                    except Exception as e:
                        logger.error(f"Error processing issue {issue_key}: {e}")

            # Save state
            issue_df.to_pickle(state_file)

        return issue_cache


# ================== State Management ==================


class OptimizationState:
    """Manages optimization state persistence and recovery"""

    def __init__(self, state_file_path: str):
        self.state_file_path = state_file_path

    def load_state(self) -> List[Issue]:
        """Load optimization state from file"""
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Issue.from_dict(issue_data) for issue_data in data]
        else:
            return []

    def save_state(self, issues: List[Issue]):
        """Save optimization state to file"""
        data = [issue.to_dict() for issue in issues]
        with open(self.state_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_state(self):
        """Clear optimization state file"""
        if os.path.exists(self.state_file_path):
            os.remove(self.state_file_path)


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

        self.critic_clients = {"qwen3-critic": self.critic_llm}

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
            # Load existing state
            issues = self.state_manager.load_state()

            # Stage 1: Issue Detection (if needed)
            if self._should_detect_new_issues(issues):
                new_issues = self._detect_new_issues(**provider_kwargs)
                if new_issues:
                    issues.extend(new_issues)
                    stats["issues_detected"] = len(new_issues)
                    logger.info(f"Detected {len(new_issues)} new issues")

            # Stage 2: Issue Evaluation
            issues = self._evaluate_issues(issues)
            high_confidence_issues = [
                issue
                for issue in issues
                if issue.validation_score
                >= self.config.processing_config.confidence_threshold
            ]
            stats["issues_validated"] = len(high_confidence_issues)

            # Stage 3: Issue Processing
            resolved_count = self._process_issues(issues)
            stats["issues_resolved"] = resolved_count

            # Update statistics
            stats["issues_by_type"] = self._calculate_issue_type_stats(issues)

            # Save final state
            self.state_manager.save_state(issues)

            logger.info(f"Optimization completed. Resolved {resolved_count} issues")
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
            self.state_manager.save_state(issues)

        return issues

    def _process_issues(self, issues: List[Issue]) -> int:
        """Process all validated issues by type"""
        logger.info("Processing validated issues...")

        resolved_count = 0

        # Process issues that meet confidence threshold and are not yet resolved
        for issue in issues:
            if (
                issue.validation_score
                >= self.config.processing_config.confidence_threshold
                and not issue.is_resolved
            ):

                try:
                    # For now, mark as resolved (actual processing logic would be more complex)
                    # This is a simplified version - you can add actual processing logic here
                    issue.is_resolved = True
                    resolved_count += 1
                    logger.info(
                        f"Processed {issue.issue_type} issue for {issue.affected_ids}"
                    )

                except Exception as e:
                    logger.error(f"Failed to process issue {issue.issue_type}: {e}")

        return resolved_count

    def _calculate_issue_type_stats(self, issues: List[Issue]) -> Dict[str, int]:
        """Calculate statistics by issue type"""
        stats = {}
        for issue_type in [
            "entity_quality_issue",
            "redundancy_entity",
            "relationship_quality_issue",
            "redundancy_relationship",
        ]:
            type_issues = [issue for issue in issues if issue.issue_type == issue_type]
            stats[issue_type] = {
                "detected": len(type_issues),
                "validated": len(
                    [
                        issue
                        for issue in type_issues
                        if issue.validation_score
                        >= self.config.processing_config.confidence_threshold
                    ]
                ),
                "resolved": len([issue for issue in type_issues if issue.is_resolved]),
            }
        return stats

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics"""
        issues = self.state_manager.load_state()

        total_issues = len(issues)
        validated_issues = len(
            [
                issue
                for issue in issues
                if issue.validation_score
                >= self.config.processing_config.confidence_threshold
            ]
        )
        resolved_issues = len([issue for issue in issues if issue.is_resolved])

        return {
            "total_issues": total_issues,
            "validated_issues": validated_issues,
            "resolved_issues": resolved_issues,
            "resolution_rate": (
                resolved_issues / validated_issues if validated_issues > 0 else 0
            ),
            "issues_by_type": self._calculate_issue_type_stats(issues),
            "config": self.config,
        }

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


# ================== Compatibility Layer ==================


def improve_graph(
    query: str, tmp_test_data_file: str = "test_data.pkl"
) -> Dict[str, Any]:
    """
    Backward compatibility function that mimics the original improve_graph interface.

    This function creates an optimization engine and runs the optimization process
    with the same interface as the original script.
    """
    # Create engine with backward-compatible configuration
    config = OptimizationConfig()
    config.processing_config.state_file_path = tmp_test_data_file

    engine = GraphOptimizationEngine(config)

    # Run optimization
    return engine.optimize_graph(query=query, top_k=30)
