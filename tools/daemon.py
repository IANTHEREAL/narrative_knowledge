import logging
import time
import json
import hashlib
import argparse
from collections import defaultdict
from typing import Dict, Any, Optional, Generator
from setting.db import SessionLocal
from knowledge_graph.models import RawDataSource, BackgroundTask
from tools.orchestrator import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineDaemon:
    """
    Daemon for managing pipeline execution.
    This class is responsible for orchestrating the execution of pipelines
    based on the provided configuration and context.
    """

    def __init__(self, check_interval: int = 5, mode: str = "files"):
        """
        Initialize the PipelineDaemon.

        Args:
            check_interval: Interval in seconds to check for new tasks
            mode: Processing mode - "files" for uploaded files or "memory" for memory chat batch
        """
        self.check_interval = check_interval
        self.mode = mode
        logger.info(
            f"PipelineDaemon initialized with check interval: {self.check_interval} seconds, mode: {mode}"
        )

    def start(self):
        """
        Start the PipelineDaemon.
        """
        self.is_running = True
        logger.info(f"PipelineDaemon started in {self.mode} mode")

        while self.is_running:
            try:
                if self.mode == "files":
                    result = self._process_uploaded_files()
                elif self.mode == "memory":
                    result = self._process_memory_chat_batch()
                else:
                    logger.error(f"Invalid mode: {self.mode}")
                    break
                
                if result:
                    logger.info(f"Processed result: {result}")
                    # self.is_running = False  # Stop after processing
            except Exception as e:
                logger.error(f"Error in daemon main loop: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def stop(self):
        """
        Stop the PipelineDaemon.
        """
        self.is_running = False
        logger.info("PipelineDaemon stopped")


    def register_background_task(self, task_id: str, task_type: str, topic_name: str, user_id: Optional[str] = None, message_count: int = 1) -> None:
        """Register a new background task for file/memory processing immediately."""
        with SessionLocal() as db:

            task = BackgroundTask(
                task_id=task_id,
                task_type=task_type,
                topic_name=topic_name,
                user_id=user_id,
                status="processing",
                message_count=message_count
            )
        db.add(task)
        db.commit()            
    
    
    def _process_uploaded_files(self):
        """
        Process uploaded files in the pipeline.
        """
        logger.info("Processing uploaded files...")

        with SessionLocal() as db:
            uploaded_data_list = (
                db.query(RawDataSource)
                .filter(
                    RawDataSource.status == "uploaded",
                    RawDataSource.target_type.in_(["knowledge_graph", "knowledge_build"]),
                )
                .all()
            )
            logger.info(
                f"Found {len(uploaded_data_list)} uploaded data sources to process"
            )

        grouped = defaultdict(list)
        for rds in uploaded_data_list:
            key = (rds.topic_name, rds.target_type, json.dumps(rds.process_strategy, sort_keys=True))
            grouped[key].append(rds)
        
        # Split groups larger than 5 files into smaller batches
        final_groups = []
        for key, rds_list in grouped.items():
            # Split into chunks of max 5 files each
            for i in range(0, len(rds_list), 5):
                chunk = rds_list[i:i+5]
                final_groups.append((key, chunk))

        logger.info(f"Grouped {len(uploaded_data_list)} uploaded files into {len(final_groups)} batches (max 5 files each) for processing")

        for (topic_name, target_type, process_strategy_json), rds_list in final_groups:
            process_strategy = json.loads(process_strategy_json) if process_strategy_json else None

            context = {
                "topic_name": topic_name,
                "target_type": target_type,
                "process_strategy": process_strategy,
                "force_regenerate": False,
                "files": []
            }

            for rds in rds_list:
                metadata = {"topic_name": rds.topic_name}   # file-specific metadata
                link = None
                content_type = None

                if rds.raw_data_source_metadata:
                    link = rds.raw_data_source_metadata.get("doc_link")
                    custom_metadata = rds.raw_data_source_metadata.get("custom_metadata", {})   # input metadata
                    force_regenerate = custom_metadata.get("force_regenerate", False)
                    content_type = custom_metadata.get("content_type")
                    context["metadata"] = custom_metadata
                    context["force_regenerate"] = force_regenerate

                context["files"].append({
                    "filename": rds.original_filename,
                    "metadata": metadata,
                    "link": link,
                    "content_type": content_type
                })
            logger.info(f"Context prepared for topic '{topic_name}': {len(context['files'])} files")

            execution_id = hashlib.sha256(topic_name.encode("utf-8")).hexdigest()
            logger.info(f"Execution ID for topic '{topic_name}': {execution_id}")

            try:
                self.register_background_task(
                    task_id=execution_id,
                    task_type="file_processing",
                    topic_name=topic_name,
                    message_count=len(context['files'])
                )
                
                orchestrator = PipelineOrchestrator()
                result = orchestrator.execute_with_process_strategy(context, execution_id)
                logger.info(
                    f"Pipeline execution completed for topic '{topic_name}': {result}"
                )

                result_dict = result.to_dict()
                error = result_dict.get("error_message")
                if error:
                    with SessionLocal() as db:
                        task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                        if task:
                            task.status = "failed"
                            task.result = result_dict
                            task.error = str(error)
                            db.commit()
                    # TODO: Trace back RDS data and mark as 'uploaded'

                    total_result = {"status": "Failed", "message": f'Error: {str(error)}'}
                    return  total_result 
                
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "completed"
                        task.result = result_dict
                        db.commit()
            except ValueError as e:
                logger.error(
                    f"Invalid process strategy for topic '{topic_name}': {e}",
                    exc_info=True
                )
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "failed"
                        task.error = str(e)
                        db.commit()
                # TODO: Trace back RDS data and mark as 'uploaded'

                total_result = {"status": "Failed", "message": f'Error: {str(e)}'}
                return  total_result 
                
                
            except Exception as e:
                logger.error(
                    f"Error executing pipeline for topic '{topic_name}': {e}",
                    exc_info=True
                )
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "failed"
                        task.error = str(e)
                        db.commit()
                # TODO: Trace back RDS data and mark as 'uploaded'
                
                total_result = {"status": "Failed", "message": f'Error: {str(e)}'}
                return  total_result 
        
        total_result = {
            "status": "Success",
            "message": f"Processed {len(uploaded_data_list)} uploaded files across {len(grouped)} topics"
        }
        return  total_result       

    def _process_memory_chat_batch(self):
        """
        Process memory chat batch in the pipeline.
        """
        logger.info("Processing memory chat batch...")

        with SessionLocal() as db:
            uploaded_data_list = (
                db.query(RawDataSource)
                .filter(
                    RawDataSource.status == "uploaded",
                    RawDataSource.target_type == "personal_memory",
                )
                .all()
            )
            logger.info(
                f"Found {len(uploaded_data_list)} uploaded data sources to process"
            )

        grouped = defaultdict(list)

        for rds in uploaded_data_list:
            key = (rds.topic_name, rds.target_type, json.dumps(rds.process_strategy, sort_keys=True))
            grouped[key].append(rds)
        
        # Split groups larger than 5 files into smaller batches
        final_groups = []
        for key, rds_list in grouped.items():
            # Split into chunks of max 5 files each
            for i in range(0, len(rds_list), 5):
                chunk = rds_list[i:i+5]
                final_groups.append((key, chunk))

        logger.info(f"Grouped {len(uploaded_data_list)} memory messages into {len(final_groups)} batches (max 5 files each) for processing")

        for (topic_name, target_type, process_strategy_json), rds_list in final_groups:
            process_strategy = json.loads(process_strategy_json) if process_strategy_json else None
            
            messages = []
            user_id = ""
            build_id = ""
            metadata = {}
            force_regenerate = False
            for rds in rds_list:
                rds_metadata = rds.raw_data_source_metadata
                if rds_metadata:
                    # Extract messages from raw_data_source_metadata
                    metadata = rds_metadata.get("metadata", {})     # Dict
                    force_regenerate = metadata.get("force_regenerate", False)
                    user_id = metadata.get("user_id")
                    if 'chat_messages' in rds_metadata:
                        messages.extend(rds_metadata['chat_messages'])
                build_id = rds.build_id
            
            execution_id = hashlib.sha256(topic_name.encode("utf-8")).hexdigest()
            try:
                self.register_background_task(
                    task_id=execution_id,
                    task_type="memory_processing",
                    topic_name=topic_name,
                    user_id=user_id,
                    message_count=len(messages)
                )
                
                # Prepare context for pipeline orchestrator
                context = {
                    "target_type": target_type,
                    "process_strategy": process_strategy,
                    "chat_messages": messages,
                    "user_id": user_id,
                    "source_id": build_id,
                    "topic_name": topic_name,
                    "force_regenerate": force_regenerate,
                    "metadata": metadata
                }

                logger.info(f"Request data prepared for topic '{topic_name}' with {len(messages)} messages")

                # Process the existing source data
                orchestrator = PipelineOrchestrator()
                result = orchestrator.execute_with_process_strategy(context, execution_id)
                
                result_dict = result.to_dict()
                error = result_dict.get("error_message")
                if error:
                    with SessionLocal() as db:
                        task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                        if task:
                            task.status = "failed"
                            task.error = str(error)
                            db.commit()
                    # TODO: Trace back RDS data and mark as 'uploaded'
                    total_result = {"status": "Failed", "message": f'Error: {str(error)}'}
                    return  total_result 
                
                # Update task status with success and clean up chat_messages
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "completed"
                        task.result = result.to_dict() if hasattr(result, 'to_dict') else result
                        db.commit()
                    
                    # Update RawDataSource to remove chat_messages from metadata and set status
                    for rds in rds_list:
                        rds_metadata = rds.raw_data_source_metadata
                        if rds_metadata and 'chat_messages' in rds_metadata and task.status == "completed":
                            # Remove chat_messages only and keep the rest
                            rds.raw_data_source_metadata = rds_metadata.get("metadata", {})
                            rds.status = "etl_completed"
                            db.add(rds)
                    db.commit()
                
                logger.info(f"Background processing completed: {result.to_dict()}")
                
            except ValueError as e:
                # Update task status with error
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "failed"
                        task.error = str(e)
                        db.commit()
                logger.error(f"Background processing failed: {e}")
                total_result = {"status": "Failed", "message": f'Error: {str(e)}'}
                return  total_result 
            
            except Exception as e:
                with SessionLocal() as db:
                    task = db.query(BackgroundTask).filter_by(task_id=execution_id, status="processing").first()
                    if task:
                        task.status = "failed"
                        task.error = str(e)
                        db.commit()
                logger.error(
                    f"Error executing pipeline for topic '{topic_name}': {e}",
                    exc_info=True
                )
                total_result = {"status": "Failed", "message": f'Error: {str(e)}'}
                return  total_result 

        total_result = {
            "status": "Success",
            "message": f"Processed {len(uploaded_data_list)} uploaded inputs across {len(grouped)} topics"
        }
        return total_result

def main():
    """
    Main entry point for the PipelineDaemon.
    """
    parser = argparse.ArgumentParser(description='Pipeline Daemon for processing uploaded files or memory chat batches')
    parser.add_argument(
        '--mode',
        choices=['files', 'memory'],
        default='files',
        help='Processing mode: "files" for uploaded files or "memory" for memory chat batch (default: files)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Check interval in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    
    daemon = PipelineDaemon(check_interval=args.interval, mode=args.mode)
    try:
        daemon.start()
    except KeyboardInterrupt:
        daemon.stop()
        logger.info("PipelineDaemon stopped by user")       
      
if __name__ == "__main__":
    main()
