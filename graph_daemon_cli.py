#!/usr/bin/env python3
"""
Knowledge Graph Daemons Command Line Interface.

This unified CLI tool provides daemon management and status checking for both
knowledge extraction and graph building daemons.
"""

import logging
import signal
import sys
import os
import argparse
import threading
import time
from dotenv import load_dotenv

from knowledge_graph.graph_builder_daemon import KnowledgeExtractionDaemon
from knowledge_graph.graph_daemon import KnowledgeGraphDaemon
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding

load_dotenv()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("knowledge_graph_daemons.log", mode="a"),
        ],
    )


def signal_handler(signum, frame, daemons):
    """Handle shutdown signals gracefully."""
    logging.info(f"Received signal {signum}, shutting down daemon(s)...")
    if isinstance(daemons, list):
        for daemon in daemons:
            if daemon and hasattr(daemon, 'stop'):
                daemon.stop()
    else:
        if daemons and hasattr(daemons, 'stop'):
            daemons.stop()
    sys.exit(0)


def start_single_daemon(args, daemon_type):
    """Start a single daemon with given arguments."""
    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    daemon_name = "Knowledge Extraction Daemon" if daemon_type == "extraction" else "Knowledge Graph Daemon"
    logger.info(f"Starting {daemon_name}...")
    
    config_info = (
        f"Configuration: check_interval={args.check_interval}s, log_level={args.log_level}, "
        f"llm_provider={args.llm_provider}, llm_model={args.llm_model}, "
        f"embedding_model endpoint={os.getenv('EMBEDDING_BASE_URL')}, "
        f"LLM endpoint={os.getenv('OPENAI_LIKE_BASE_URL')}"
    )
    
    if daemon_type == "graph":
        config_info += f", worker_count={args.worker_count}"
    
    logger.info(config_info)

    try:
        # Initialize LLM client
        llm_client = LLMInterface(args.llm_provider, args.llm_model)
        logger.info(f"Initialized LLM client: {args.llm_provider}/{args.llm_model}")

        # Initialize daemon based on type
        if daemon_type == "extraction":
            daemon = KnowledgeExtractionDaemon(
                llm_client=llm_client,
                embedding_func=get_text_embedding,
                check_interval=args.check_interval,
            )
        else:  # graph
            daemon = KnowledgeGraphDaemon(
                llm_client=llm_client,
                embedding_func=get_text_embedding,
                check_interval=args.check_interval,
                worker_count=args.worker_count,
            )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, daemon))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, daemon))

        # Start the daemon
        logger.info(f"{daemon_name} is now running. Press Ctrl+C to stop.")
        daemon.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error starting daemon: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"{daemon_name} stopped.")


def start_both_daemons(args):
    """Start both daemons concurrently."""
    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting both Knowledge Extraction and Graph Building daemons...")
    
    config_info = (
        f"Configuration: extraction_interval={args.extraction_interval}s, "
        f"graph_interval={args.graph_interval}s, log_level={args.log_level}, "
        f"llm_provider={args.llm_provider}, llm_model={args.llm_model}, "
        f"worker_count={args.worker_count}, "
        f"embedding_model endpoint={os.getenv('EMBEDDING_BASE_URL')}, "
        f"LLM endpoint={os.getenv('OPENAI_LIKE_BASE_URL')}"
    )
    logger.info(config_info)

    try:
        # Initialize LLM client
        llm_client = LLMInterface(args.llm_provider, args.llm_model)
        logger.info(f"Initialized LLM client: {args.llm_provider}/{args.llm_model}")

        # Initialize both daemons
        extraction_daemon = KnowledgeExtractionDaemon(
            llm_client=llm_client,
            embedding_func=get_text_embedding,
            check_interval=args.extraction_interval,
        )
        
        graph_daemon = KnowledgeGraphDaemon(
            llm_client=llm_client,
            embedding_func=get_text_embedding,
            check_interval=args.graph_interval,
            worker_count=args.worker_count,
        )

        daemons = [extraction_daemon, graph_daemon]

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, daemons))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, daemons))

        # Start both daemons in separate threads
        def run_extraction():
            try:
                extraction_daemon.start()
            except Exception as e:
                logger.error(f"Error in extraction daemon: {e}", exc_info=True)

        def run_graph():
            try:
                graph_daemon.start()
            except Exception as e:
                logger.error(f"Error in graph daemon: {e}", exc_info=True)

        extraction_thread = threading.Thread(target=run_extraction, name="ExtractionDaemon")
        graph_thread = threading.Thread(target=run_graph, name="GraphDaemon")

        extraction_thread.daemon = True
        graph_thread.daemon = True

        extraction_thread.start()
        graph_thread.start()

        logger.info("Both daemons are now running. Press Ctrl+C to stop.")

        # Keep main thread alive
        try:
            while True:
                time.sleep(10)
                # Optional: log status
                if extraction_thread.is_alive() and graph_thread.is_alive():
                    logger.debug("Both daemons are running normally...")
                else:
                    logger.warning("One or more daemons have stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")

        # Wait for threads to finish
        extraction_thread.join(timeout=5)
        graph_thread.join(timeout=5)

    except Exception as e:
        logger.error(f"Error starting daemons: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("All daemons stopped.")


def show_status(args):
    """Show daemon and task status."""
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs for status display

    # Create daemon instances (but don't start them)
    extraction_daemon = KnowledgeExtractionDaemon()
    graph_daemon = KnowledgeGraphDaemon()

    # Get status information from both daemons
    extraction_status = extraction_daemon.get_daemon_status()
    graph_status = graph_daemon.get_daemon_status()

    print("KNOWLEDGE EXTRACTION DAEMON STATUS:")
    print("=" * 50)
    print(f"Daemon Running:   {extraction_status['is_running']}")
    print(f"Check Interval:   {extraction_status['check_interval']}s")
    print(f"Pending Tasks:    {extraction_status['pending_tasks']}")
    print(f"Processing Tasks: {extraction_status['processing_tasks']}")
    print(f"Completed Tasks:  {extraction_status['completed_tasks']}")
    print(f"Failed Tasks:     {extraction_status['failed_tasks']}")
    print(f"Total Tasks:      {extraction_status['total_tasks']}")
    print()

    print("KNOWLEDGE GRAPH DAEMON STATUS:")
    print("=" * 50)
    print(f"Daemon Running:      {graph_status['is_running']}")
    print(f"Check Interval:      {graph_status['check_interval']}s")
    print(f"Worker Count:        {graph_status['worker_count']}")
    print(f"Total Unmapped:      {graph_status['total_unmapped_sources']}")
    print(f"Total Mappings:      {graph_status.get('total_graph_mappings', 'N/A')}")
    print(f"Entity Mappings:     {graph_status.get('entity_mappings', 'N/A')}")
    print(f"Relationship Mappings: {graph_status.get('relationship_mappings', 'N/A')}")
    print()

    # Show completed topics with unmapped sources
    if graph_status.get('completed_topics'):
        print("COMPLETED TOPICS WITH UNMAPPED SOURCES:")
        print("-" * 50)
        for topic_key, count in graph_status['completed_topics'].items():
            if isinstance(count, int) and count > 0:
                print(f"  • {topic_key}: {count} unmapped sources")
        print()

    # Show detailed pending tasks
    if extraction_status["pending_tasks"] > 0:
        from setting.db import SessionLocal
        from knowledge_graph.models import GraphBuild

        print("DETAILED PENDING EXTRACTION TASKS:")
        print("-" * 50)

        with SessionLocal() as db:
            pending_tasks = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "pending")
                .order_by(GraphBuild.scheduled_at.asc())
                .limit(10)  # Limit to avoid too much output
                .all()
            )

            for task in pending_tasks:
                scheduled_str = task.scheduled_at.strftime("%Y-%m-%d %H:%M:%S")
                db_type = "local" if not task.external_database_uri else "external"

                print(f"  • Topic: {task.topic_name}")
                print(f"    Build ID: {task.build_id}")
                print(f"    Database: {db_type}")
                print(f"    Scheduled: {scheduled_str}")
                print()

    # Show recent failed tasks if requested or if there are failed tasks
    if args.show_failed or extraction_status["failed_tasks"] > 0:
        from setting.db import SessionLocal
        from knowledge_graph.models import GraphBuild

        print("RECENT FAILED EXTRACTION TASKS:")
        print("-" * 50)

        with SessionLocal() as db:
            failed_tasks = (
                db.query(GraphBuild)
                .filter(GraphBuild.status == "failed")
                .order_by(GraphBuild.updated_at.desc())
                .limit(args.failed_limit if hasattr(args, "failed_limit") else 5)
                .all()
            )

            if not failed_tasks:
                print("  No failed tasks found.")
                print()
            else:
                for task in failed_tasks:
                    updated_str = task.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                    error_msg = task.error_message or "No error message"
                    db_type = "local" if not task.external_database_uri else "external"

                    print(f"  • Topic: {task.topic_name}")
                    print(f"    Build ID: {task.build_id}")
                    print(f"    Database: {db_type}")
                    print(f"    Failed: {updated_str}")
                    print(f"    Error: {error_msg}")
                    print()


def main():
    """Main function with subcommands."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Daemons CLI - Manage and monitor knowledge extraction and graph building",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extraction                     # Start knowledge extraction daemon
  %(prog)s graph                          # Start knowledge graph daemon  
  %(prog)s both                           # Start both daemons concurrently
  %(prog)s status                         # Show current status of both daemons
  %(prog)s status --show-failed           # Show status including failed tasks
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments for all start commands
    def add_common_args(parser_obj):
        parser_obj.add_argument(
            "--llm-provider",
            type=str,
            default="openai_like",
            help="LLM provider to use (default: openai_like)",
        )
        parser_obj.add_argument(
            "--llm-model",
            type=str,
            default="qwen3-32b",
            help="LLM model to use (default: qwen3-32b)",
        )
        parser_obj.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level (default: INFO)",
        )

    # Knowledge extraction daemon subcommand
    extraction_parser = subparsers.add_parser("extraction", help="Start the knowledge extraction daemon")
    add_common_args(extraction_parser)
    extraction_parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Interval in seconds to check for pending tasks (default: 60)",
    )

    # Knowledge graph daemon subcommand
    graph_parser = subparsers.add_parser("graph", help="Start the knowledge graph daemon")
    add_common_args(graph_parser)
    graph_parser.add_argument(
        "--check-interval",
        type=int,
        default=120,
        help="Interval in seconds to check for unmapped sources (default: 120)",
    )
    graph_parser.add_argument(
        "--worker-count",
        type=int,
        default=5,
        help="Number of workers for graph building (default: 5)",
    )

    # Both daemons subcommand
    both_parser = subparsers.add_parser("both", help="Start both daemons concurrently")
    add_common_args(both_parser)
    both_parser.add_argument(
        "--extraction-interval",
        type=int,
        default=60,
        help="Interval in seconds for extraction daemon (default: 60)",
    )
    both_parser.add_argument(
        "--graph-interval",
        type=int,
        default=120,
        help="Interval in seconds for graph daemon (default: 120)",
    )
    both_parser.add_argument(
        "--worker-count",
        type=int,
        default=5,
        help="Number of workers for graph building (default: 5)",
    )

    # Status subcommand
    status_parser = subparsers.add_parser("status", help="Show daemon and task status")
    status_parser.add_argument(
        "--show-failed", action="store_true", help="Always show failed tasks section"
    )
    status_parser.add_argument(
        "--failed-limit",
        type=int,
        default=5,
        help="Maximum number of failed tasks to show (default: 5)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "extraction":
            start_single_daemon(args, "extraction")
        elif args.command == "graph":
            start_single_daemon(args, "graph")
        elif args.command == "both":
            start_both_daemons(args)
        elif args.command == "status":
            show_status(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
