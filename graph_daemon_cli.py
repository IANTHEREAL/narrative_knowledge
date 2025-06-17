#!/usr/bin/env python3
"""
Graph Build Daemon Command Line Interface.

This unified CLI tool provides both daemon management and status checking functionality.
"""

import logging
import signal
import sys
import os
import argparse
from dotenv import load_dotenv

from knowledge_graph.graph_builder_daemon import GraphBuildDaemon
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
            logging.FileHandler("graph_build_daemon.log", mode="a"),
        ],
    )


def signal_handler(signum, frame, daemon):
    """Handle shutdown signals gracefully."""
    logging.info(f"Received signal {signum}, shutting down daemon...")
    daemon.stop()
    sys.exit(0)


def start_daemon(args):
    """Start the daemon with given arguments."""
    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Graph Build Daemon...")
    logger.info(
        f"Configuration: check_interval={args.check_interval}s, "
        f"llm_provider={args.llm_provider}, llm_model={args.llm_model}, "
        f"log_level={args.log_level}, max_retries={args.max_retries}, "
        f"embedding_model endpoint={os.getenv('EMBEDDINF_BASE_URL')}, "
        f"LLM endpoint={os.getenv('OPENAI_LIKE_BASE_URL')}"
    )

    try:
        # Initialize LLM client
        llm_client = LLMInterface(args.llm_provider, args.llm_model)
        logger.info(f"Initialized LLM client: {args.llm_provider}/{args.llm_model}")

        # Initialize daemon
        daemon = GraphBuildDaemon(
            llm_client=llm_client,
            embedding_func=get_text_embedding,
            check_interval=args.check_interval,
            max_retries=args.max_retries,
        )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, daemon))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, daemon))

        # Print initial status
        status = daemon.get_daemon_status()
        logger.info(f"Initial daemon status: {status}")

        # Start the daemon
        logger.info("Graph Build Daemon is now running. Press Ctrl+C to stop.")
        daemon.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error starting daemon: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Graph Build Daemon stopped.")


def show_status(args):
    """Show daemon and task status."""
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs for status display

    # Create daemon instance (but don't start it)
    daemon = GraphBuildDaemon()

    # Get status information
    status = daemon.get_daemon_status()

    print("TASK STATUS SUMMARY:")
    print("-" * 30)
    print(f"Pending Tasks:    {status['pending_tasks']}")
    print(f"Processing Tasks: {status['processing_tasks']}")
    print(f"Completed Tasks:  {status['completed_tasks']}")
    print(f"Failed Tasks:     {status['failed_tasks']}")
    print(f"Total Tasks:      {status['total_tasks']}")
    print()

    # Show detailed pending tasks
    if status["pending_tasks"] > 0:
        from setting.db import SessionLocal
        from knowledge_graph.models import GraphBuildStatus

        print("DETAILED PENDING TASKS:")
        print("-" * 30)

        with SessionLocal() as db:
            pending_tasks = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "pending")
                .order_by(GraphBuildStatus.scheduled_at.asc())
                .all()
            )

            for task in pending_tasks:
                scheduled_str = task.scheduled_at.strftime("%Y-%m-%d %H:%M:%S")

                print(f"  • Topic: {task.topic_name}")
                print(f"    Source ID: {task.source_id}")
                print(f"    Scheduled: {scheduled_str}")
                print()

    # Show recent failed tasks if requested or if there are failed tasks
    if args.show_failed or status["failed_tasks"] > 0:
        from setting.db import SessionLocal
        from knowledge_graph.models import GraphBuildStatus

        print("RECENT FAILED TASKS:")
        print("-" * 30)

        with SessionLocal() as db:
            failed_tasks = (
                db.query(GraphBuildStatus)
                .filter(GraphBuildStatus.status == "failed")
                .order_by(GraphBuildStatus.updated_at.desc())
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

                    print(f"  • Topic: {task.topic_name}")
                    print(f"    Source ID: {task.source_id}")
                    print(f"    Failed: {updated_str}")
                    print(f"    Error: {error_msg}")
                    print()


def main():
    """Main function with subcommands."""
    parser = argparse.ArgumentParser(
        description="Graph Build Daemon CLI - Manage and monitor graph build tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                          # Start daemon with default settings
  %(prog)s start --check-interval 60     # Start with 60s check interval
  %(prog)s status                         # Show current status
  %(prog)s status --show-failed           # Show status including failed tasks
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start daemon subcommand
    start_parser = subparsers.add_parser("start", help="Start the graph build daemon")
    start_parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Interval in seconds to check for pending tasks (default: 30)",
    )
    start_parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai_like",
        help="LLM provider to use (default: openai_like)",
    )
    start_parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3-32b",
        help="LLM model to use (default: qwen3-32b)",
    )
    start_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    start_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed tasks (default: 3)",
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

    if args.command == "start":
        start_daemon(args)
    elif args.command == "status":
        show_status(args)


if __name__ == "__main__":
    main()
