import logging
import json
import os
import pandas as pd
from typing import Tuple
import concurrent.futures

from knowledge_graph.query import search_relationships_by_vector_similarity
from setting.db import db_manager
from opt.helper import GRAPH_OPTIMIZATION_ACTION_SYSTEM_PROMPT_WO_MR
from opt.evaluator import batch_evaluate_issues
from opt.helper import extract_issues
from llm.factory import LLMInterface
from knowledge_graph.models import Entity, Relationship, SourceGraphMapping
from opt.optimizer import (
    process_entity_quality_issue,
    process_redundancy_entity_issue,
    process_relationship_quality_issue,
    process_redundancy_relationship_issue,
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger
logger = logging.getLogger(__name__)

optimization_llm_client = LLMInterface("openai_like", "graph_optimization_14b")

qwen3_critic_client = LLMInterface("bedrock", "us.anthropic.claude-sonnet-4-20250514-v1:0")
# sonnet_critic_client = LLMInterface("bedrock", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
# deepseek_critic_client = LLMInterface("openai_like", "deepseek/deepseek-r1-0528-qwen3-8b")
critic_clients = {
    "qwen3-critic": qwen3_critic_client,
    # "sonnet-3.7-critic": sonnet_critic_client,
    #"deepseek-R1-critic": deepseek_critic_client,
}

session_factory = db_manager.get_session_factory(os.getenv("GRAPH_DATABASE_URI"))


def get_issue_key(issue: dict) -> Tuple[str, tuple]:
    """Generate a unique key for an issue based on its type and affected IDs."""
    return (issue["issue_type"], tuple(sorted(issue["affected_ids"])))

def graph_retrieve(query: str, top_k: int = 10, similarity_threshold: float = 0.3):
    res = search_relationships_by_vector_similarity(
        query,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        database_uri=os.getenv("GRAPH_DATABASE_URI")
    )
    entities = {}
    relationships = {}

    for index, row in res.iterrows():
        entities[row['source_entity_id']] = {
            "id": row['source_entity_id'],
            "name": row['source_entity_name'],
            "description": row['source_entity_description'],
            "attributes": row['source_entity_attributes']
        }
        entities[row['target_entity_id']] = {
            "id": row['target_entity_id'],
            "name": row['target_entity_name'],
            "description": row['target_entity_description'],
            "attributes": row['target_entity_attributes']
        }
        relationships[row['id']] = {
            "id": row['id'],
            "source_entity": row['source_entity_name'],
            "target_entity": row['target_entity_name'],
            "description": row['relationship_desc'],
            "attributes": row['attributes']
        }

    return {
        "entities": list(entities.values()),
        "relationships": list(relationships.values())
    }


def improve_graph(query: str, tmp_test_data_file: str = "test_data.pkl"):
    if os.path.exists(tmp_test_data_file):
        issue_df = pd.read_pickle(tmp_test_data_file)
    else:
        issue_df = pd.DataFrame(
            columns=[
                "graph",
                "question",
                "issue",
                "confidence",
                "qwen3-critic",
                "resolved",
            ]
        )

    new_issue_list = []
    # if having unresolved issue, we need to handle these issue first
    # if no issue need to be critized, we need to retrieve new issues
    if (
        issue_df[
            (issue_df["resolved"] == False) & (issue_df["confidence"] >= 0.9)
        ].shape[0]
        == 0
        and issue_df[
            ["qwen3-critic"]
        ]
        .notnull()
        .all()
        .all()
    ):
        print(
            "no unresolved issue and all issues have complete critic evaluations, retrieving new issues"
        )
        retrieval_results = graph_retrieve(query, top_k=30, similarity_threshold=0.3)

        graph_data = {
            "entities": retrieval_results["entities"],
            "relationships": retrieval_results["relationships"],
        }

        prompt = (
            GRAPH_OPTIMIZATION_ACTION_SYSTEM_PROMPT_WO_MR
            + " Now Optimize the following graph:\n"
            + json.dumps(graph_data, indent=2, ensure_ascii=False)
        )
        response = optimization_llm_client.generate(prompt)

        analysis_list = extract_issues(response)
        print("analysis:", analysis_list)
        for analysis in analysis_list.values():
            for issue in analysis:
                issue_data = {
                    "graph": graph_data,
                    "question": "what is write hotspot?",
                    "issue": issue,
                    "confidence": 0.0,
                    "qwen3-critic": None,
                    "resolved": False,
                }
                new_issue_list.append(issue_data)

        if len(new_issue_list) > 0:
            issue_df = pd.concat([issue_df, pd.DataFrame(new_issue_list)])

    issue_df.to_pickle(tmp_test_data_file)

    print(f"Found new issues {len(new_issue_list)}, total issues {issue_df.shape[0]}")

    for row in new_issue_list:
        print(row["issue"])

    print("=" * 60)

    # if there are issue that need to be critized, we need to evaluate them
    while (
        issue_df[["qwen3-critic"]]
        .isnull()
        .any()
        .any()
    ):
        issue_df = batch_evaluate_issues(critic_clients, issue_df)

    issue_df.to_pickle(tmp_test_data_file)
    print(f"Identified {issue_df[issue_df['confidence'] >= 0.9].shape[0]} valid issues")

    issue_cache = {}
    for index, row in issue_df.iterrows():
        if row["resolved"] is not True:
            continue

        if (
            row["issue"]["issue_type"] == "entity_quality_issue"
            or row["issue"]["issue_type"] == "relationship_quality_issue"
        ):
            for affected_id in row["issue"]["affected_ids"]:
                issue = {
                    "issue_type": row["issue"]["issue_type"],
                    "affected_ids": [affected_id],
                    "reasoning": row["issue"]["reasoning"],
                }
                issue_cache[get_issue_key(issue)] = True
        else:
            issue_cache[get_issue_key(row["issue"])] = True

    print("issue is resolved", issue_cache, len(issue_cache))

    ## process entity quality issue

    pending_entity_quality_issue_list = {}
    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "entity_quality_issue"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue

        # Check if any entities need processing
        for affected_id in row["issue"]["affected_ids"]:
            issue = {
                "issue_type": row["issue"]["issue_type"],
                "reasoning": row["issue"]["reasoning"],
                "affected_ids": [affected_id],
                "row_index": index,
            }
            issue_key = get_issue_key(issue)
            if (
                issue_cache.get(issue_key, False)
                or pending_entity_quality_issue_list.get(issue_key, None) is not None
            ):
                logger.info(f"Entity quality issue {index} already processed or pending, marking as resolved")
                issue_df.at[index, "resolved"] = True
                issue_df.to_pickle(tmp_test_data_file)
                continue
            issue["issue_key"] = issue_key
            pending_entity_quality_issue_list[issue_key] = issue

    print(
        "pendding entity quality issues number", len(pending_entity_quality_issue_list)
    )

    parallel_count = 1
    while True:
        # Find up to 3 issues that need processing
        keys_for_batch = []
        for key in pending_entity_quality_issue_list:
            if len(keys_for_batch) < parallel_count:
                keys_for_batch.append(key)
            else:
                break

        # Exit if no more rows to process
        if len(keys_for_batch) == 0:
            break

        batch_issues = []
        for key in keys_for_batch:
            batch_issues.append(pending_entity_quality_issue_list.pop(key))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch_issues)
        ) as executor:
            futures = {}
            for row_issue in batch_issues:
                futures[
                    executor.submit(
                        process_entity_quality_issue,
                        session_factory,
                        qwen3_critic_client,
                        Entity,
                        Relationship,
                        row_issue["row_index"],
                        row_issue,
                    )
                ] = row_issue["issue_key"]

            # Process results and update dataframe
            for future in concurrent.futures.as_completed(futures):
                issue_key = futures[future]
                success = future.result()
                if success:
                    issue_cache[issue_key] = True

    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "entity_quality_issue"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue
        success = True
        for affected_id in row["issue"]["affected_ids"]:
            tmp_key = get_issue_key(
                {
                    "issue_type": row["issue"]["issue_type"],
                    "reasoning": row["issue"]["reasoning"],
                    "affected_ids": [affected_id],
                }
            )
            if issue_cache.get(tmp_key, False) is False:
                success = False
                break
        if success:
            print(f"Success to resolve entity {index}")
            issue_df.at[index, "resolved"] = True

    # Save dataframe after each batch
    issue_df.to_pickle(tmp_test_data_file)

    ## process redundancy entity issue

    pending_redundancy_entity_issue_list = {}
    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "redundancy_entity"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue

        affected_ids = set(row["issue"]["affected_ids"])
        need_merge_ids = set(affected_ids)
        need_merge_reasoning = set([row["issue"]["reasoning"]])
        handled_index = set([index])

        found = True
        while found:
            found = False
            for other_row_index, other_row in issue_df.iterrows():
                if other_row_index == index or other_row_index in handled_index:
                    continue
                if (
                    other_row["issue"]["issue_type"] != "redundancy_entity"
                    or other_row["confidence"] < 0.9
                    or other_row["resolved"] is True
                ):
                    continue
                other_affected_ids = set(other_row["issue"]["affected_ids"])
                if need_merge_ids.isdisjoint(other_affected_ids):
                    continue

                handled_index.add(other_row_index)
                need_merge_ids.update(other_row["issue"]["affected_ids"])
                need_merge_reasoning.add(other_row["issue"]["reasoning"])
                found = True

        if len(need_merge_ids) > 1:
            redundancy_entity_issue = {
                "issue_type": "redundancy_entity",
                "affected_ids": list(need_merge_ids),
                "reasoning": "\n".join(list(need_merge_reasoning)),
                "row_indexes": list(handled_index),
            }

            issue_key = get_issue_key(redundancy_entity_issue)
            if pending_redundancy_entity_issue_list.get(
                issue_key, None
            ) is not None or issue_cache.get(issue_key, False):
                logger.info(f"Redundancy entity issue {index} already processed or pending, marking as resolved")
                issue_df.at[index, "resolved"] = True
                issue_df.to_pickle(tmp_test_data_file)
                continue

            redundancy_entity_issue["issue_key"] = issue_key
            pending_redundancy_entity_issue_list[issue_key] = redundancy_entity_issue

    print(
        "pendding redundancy entity number", len(pending_redundancy_entity_issue_list)
    )

    # Main processing loop with batched concurrency
    parallel_count = 1
    while True:
        # Find up to 3 issues that need processing
        keys_for_batch = []
        for key in pending_redundancy_entity_issue_list:
            if len(keys_for_batch) < parallel_count:
                keys_for_batch.append(key)
            else:
                break

        # Exit if no more rows to process
        if len(keys_for_batch) == 0:
            break

        batch_issues = {}
        for key in keys_for_batch:
            batch_issues[key] = pending_redundancy_entity_issue_list.pop(key)

        # Process batch concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch_issues)
        ) as executor:
            futures = {}
            for row_issue in batch_issues.values():
                futures[
                    executor.submit(
                        process_redundancy_entity_issue,
                        session_factory,
                        qwen3_critic_client,
                        Entity,
                        Relationship,
                        SourceGraphMapping,
                        row_issue["issue_key"],
                        row_issue,
                    )
                ] = row_issue["issue_key"]

            # Process results and update dataframe
            for future in concurrent.futures.as_completed(futures):
                issue_key = futures[future]
                success = future.result()
                if success:
                    issue = batch_issues[issue_key]
                    issue_cache[issue_key] = True
                    row_indexes = issue["row_indexes"]
                    for row_index in row_indexes:
                        issue_df.at[row_index, "resolved"] = True

        issue_df.to_pickle(tmp_test_data_file)

    ## process relationship quality issue

    pending_relationship_quality_issue_list = {}
    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "relationship_quality_issue"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue

        # Check if any entities need processing
        for affected_id in row["issue"]["affected_ids"]:
            issue = {
                "issue_type": row["issue"]["issue_type"],
                "reasoning": row["issue"]["reasoning"],
                "affected_ids": [affected_id],
                "row_index": index,
            }
            issue_key = get_issue_key(issue)
            if (
                issue_cache.get(issue_key, False)
                or pending_relationship_quality_issue_list.get(issue_key, None)
                is not None
            ):
                logger.info(f"Relationship quality issue {index} already processed or pending, marking as resolved")
                issue_df.at[index, "resolved"] = True
                issue_df.to_pickle(tmp_test_data_file)
                continue
            issue["issue_key"] = issue_key
            pending_relationship_quality_issue_list[issue_key] = issue

    print(
        "pendding relationship quality issues number",
        len(pending_relationship_quality_issue_list),
    )

    parallel_count = 1
    while True:
        # Find up to 3 issues that need processing
        keys_for_batch = []
        for key in pending_relationship_quality_issue_list:
            if len(keys_for_batch) < parallel_count:
                keys_for_batch.append(key)
            else:
                break

        # Exit if no more rows to process
        if len(keys_for_batch) == 0:
            break

        batch_issues = []
        for key in keys_for_batch:
            batch_issues.append(pending_relationship_quality_issue_list.pop(key))

        # Process batch concurrently

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch_issues)
        ) as executor:
            futures = {}
            for row_issue in batch_issues:
                futures[
                    executor.submit(
                        process_relationship_quality_issue,
                        session_factory,
                        qwen3_critic_client,
                        Relationship,
                        row_issue["row_index"],
                        row_issue,
                    )
                ] = row_issue["issue_key"]

            # Process results and update dataframe
            for future in concurrent.futures.as_completed(futures):
                issue_key = futures[future]
                success = future.result()
                if success:
                    print(f"Success to resolve relationship {issue_key}")
                    issue_cache[issue_key] = True

    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "relationship_quality_issue"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue
        success = True
        for affected_id in row["issue"]["affected_ids"]:
            tmp_key = get_issue_key(
                {
                    "issue_type": row["issue"]["issue_type"],
                    "reasoning": row["issue"]["reasoning"],
                    "affected_ids": [affected_id],
                }
            )
            if issue_cache.get(tmp_key, False) is False:
                success = False
                break
        if success:
            print(f"Success to resolve entity {index}")
            issue_df.at[index, "resolved"] = True

    # Save dataframe after each batch
    issue_df.to_pickle(tmp_test_data_file)

    ## process redundancy relationship issue

    pending_redundancy_relationships_issue_list = {}
    for index, row in issue_df.iterrows():
        if (
            row["issue"]["issue_type"] != "redundancy_relationship"
            or row["confidence"] < 0.9
            or row["resolved"] is True
        ):
            continue

        affected_ids = set(row["issue"]["affected_ids"])
        need_merge_ids = set(affected_ids)
        need_merge_reasoning = set([row["issue"]["reasoning"]])
        handled_index = set([index])

        found = True
        while found:
            found = False
            for other_row_index, other_row in issue_df.iterrows():
                if other_row_index == index or other_row_index in handled_index:
                    continue
                if (
                    other_row["issue"]["issue_type"] != "redundancy_relationship"
                    or other_row["confidence"] < 0.9
                    or other_row["resolved"] is True
                ):
                    continue
                other_affected_ids = set(other_row["issue"]["affected_ids"])
                if need_merge_ids.isdisjoint(other_affected_ids):
                    continue

                handled_index.add(other_row_index)
                need_merge_ids.update(other_row["issue"]["affected_ids"])
                need_merge_reasoning.add(other_row["issue"]["reasoning"])
                found = True

        if len(need_merge_ids) > 1:
            redundancy_relationship_issue = {
                "issue_type": "redundancy_relationship",
                "affected_ids": list(need_merge_ids),
                "reasoning": "\n".join(list(need_merge_reasoning)),
                "row_indexes": list(handled_index),
            }

            issue_key = get_issue_key(redundancy_relationship_issue)
            if pending_redundancy_relationships_issue_list.get(
                issue_key, None
            ) is not None or issue_cache.get(issue_key, False):
                logger.info(f"Redundancy relationship issue {index} already processed or pending, marking as resolved")
                issue_df.at[index, "resolved"] = True
                issue_df.to_pickle(tmp_test_data_file)
                continue

            redundancy_relationship_issue["issue_key"] = issue_key
            pending_redundancy_relationships_issue_list[issue_key] = (
                redundancy_relationship_issue
            )

    print(
        "pendding redundancy relationships number",
        len(pending_redundancy_relationships_issue_list),
    )

    # Main processing loop with batched concurrency

    parallel_count = 1
    while True:
        # Find up to 3 issues that need processing
        keys_for_batch = []
        for key in pending_redundancy_relationships_issue_list:
            if len(keys_for_batch) < parallel_count:
                keys_for_batch.append(key)
            else:
                break

        # Exit if no more rows to process
        if len(keys_for_batch) == 0:
            break

        batch_issues = {}
        for key in keys_for_batch:
            batch_issues[key] = pending_redundancy_relationships_issue_list.pop(key)

        # Process batch concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch_issues)
        ) as executor:
            futures = {}
            for row_issue in batch_issues.values():
                futures[
                    executor.submit(
                        process_redundancy_relationship_issue,
                        session_factory,
                        qwen3_critic_client,
                        Relationship,
                        SourceGraphMapping,
                        row_issue["issue_key"],
                        row_issue,
                    )
                ] = row_issue["issue_key"]

            # Process results and update dataframe
            for future in concurrent.futures.as_completed(futures):
                issue_key = futures[future]
                success = future.result()
                if success:
                    issue = batch_issues[issue_key]
                    issue_cache[issue_key] = True
                    row_indexes = issue["row_indexes"]
                    for row_index in row_indexes:
                        issue_df.at[row_index, "resolved"] = True

        issue_df.to_pickle(tmp_test_data_file)
