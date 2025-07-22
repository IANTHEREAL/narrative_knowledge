"""
Base job classes and utilities for the pipeline job system
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from knowledge_graph.models import JobExecution, PipelineJob


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of job execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "execution_time_seconds": self.execution_time_seconds,
        }


@dataclass
class JobContext:
    """Context information for job execution"""
    execution_id: str
    job_type: str
    input_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "execution_id": self.execution_id,
            "job_type": self.job_type,
            "input_data": self.input_data,
            "config": self.config,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class BaseJob(ABC):
    """
    Base class for all pipeline jobs
    
    Each job is responsible for:
    1. Validating input data
    2. Executing the specific job logic
    3. Updating state objects
    4. Returning structured results
    """
    
    def __init__(self, session_factory, llm_client=None, embedding_func=None):
        """
        Initialize base job
        
        Args:
            session_factory: Database session factory
            llm_client: LLM client for jobs that need LLM processing
            embedding_func: Embedding function for jobs that need embeddings
        """
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.embedding_func = embedding_func
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def job_type(self) -> str:
        """Return the job type identifier"""
        pass
    
    @property
    @abstractmethod
    def job_name(self) -> str:
        """Return the human-readable job name"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Return the expected input data schema"""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Return the expected output data schema"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the job's input schema
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - can be overridden by specific jobs
        required_fields = self.input_schema.get("required", [])
        for field in required_fields:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        return True
    
    @abstractmethod
    def execute_job_logic(self, context: JobContext) -> JobResult:
        """
        Execute the core job logic
        
        Args:
            context: Job execution context
            
        Returns:
            JobResult with execution results
        """
        pass
    
    def execute(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> JobResult:
        """
        Execute the job with full error handling and state tracking
        
        Args:
            input_data: Input data for the job
            config: Optional job configuration
            
        Returns:
            JobResult with execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create job context
        context = JobContext(
            execution_id=execution_id,
            job_type=self.job_type,
            input_data=input_data,
            config=config or {}
        )
        
        self.logger.info(f"Starting job execution: {execution_id} ({self.job_name})")
        
        # Create job execution record
        job_execution = None
        try:
            with self.session_factory() as db:
                # Get or create pipeline job definition
                pipeline_job = self._get_or_create_pipeline_job(db)
                
                # Create job execution record
                job_execution = JobExecution(
                    job_id=pipeline_job.id,
                    execution_context=context.to_dict(),
                    status=JobStatus.RUNNING.value,
                    started_at=start_time,
                )
                db.add(job_execution)
                db.commit()
                db.refresh(job_execution)
                
                context.execution_id = job_execution.id
        
        except Exception as e:
            self.logger.error(f"Failed to create job execution record: {e}")
            return JobResult(
                success=False,
                error_message=f"Failed to create job execution record: {e}"
            )
        
        # Validate input
        if not self.validate_input(input_data):
            error_msg = "Input validation failed"
            self._update_job_execution_failed(job_execution, error_msg)
            return JobResult(success=False, error_message=error_msg)
        
        # Execute job logic
        try:
            result = self.execute_job_logic(context)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            result.execution_time_seconds = execution_time
            
            # Update job execution record
            self._update_job_execution_success(job_execution, result)
            
            self.logger.info(f"Job execution completed successfully: {execution_id} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"Job execution failed: {e}"
            
            self.logger.error(f"Job execution failed: {execution_id} - {error_msg}", exc_info=True)
            
            result = JobResult(
                success=False,
                error_message=error_msg,
                error_details={"exception_type": type(e).__name__, "exception_str": str(e)},
                execution_time_seconds=execution_time
            )
            
            # Update job execution record
            self._update_job_execution_failed(job_execution, error_msg, result.error_details)
            
            return result
    
    def _get_or_create_pipeline_job(self, db: Session) -> PipelineJob:
        """Get or create the pipeline job definition"""
        pipeline_job = db.query(PipelineJob).filter(
            PipelineJob.job_type == self.job_type
        ).first()
        
        if not pipeline_job:
            pipeline_job = PipelineJob(
                job_type=self.job_type,
                job_name=self.job_name,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                is_active=True
            )
            db.add(pipeline_job)
            db.commit()
            db.refresh(pipeline_job)
        
        return pipeline_job
    
    def _update_job_execution_success(self, job_execution: JobExecution, result: JobResult):
        """Update job execution record on success"""
        try:
            with self.session_factory() as db:
                execution = db.query(JobExecution).filter(
                    JobExecution.id == job_execution.id
                ).first()
                
                if execution:
                    execution.status = JobStatus.COMPLETED.value
                    execution.completed_at = datetime.now()
                    execution.execution_result = result.to_dict()
                    db.commit()
        except Exception as e:
            self.logger.error(f"Failed to update job execution record: {e}")
    
    def _update_job_execution_failed(self, job_execution: JobExecution, error_message: str, error_details: Optional[Dict] = None):
        """Update job execution record on failure"""
        try:
            with self.session_factory() as db:
                execution = db.query(JobExecution).filter(
                    JobExecution.id == job_execution.id
                ).first()
                
                if execution:
                    execution.status = JobStatus.FAILED.value
                    execution.completed_at = datetime.now()
                    execution.error_message = error_message
                    execution.error_details = error_details
                    db.commit()
        except Exception as e:
            self.logger.error(f"Failed to update job execution record: {e}")
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get execution history for this job type
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of execution records
        """
        try:
            with self.session_factory() as db:
                pipeline_job = db.query(PipelineJob).filter(
                    PipelineJob.job_type == self.job_type
                ).first()
                
                if not pipeline_job:
                    return []
                
                executions = db.query(JobExecution).filter(
                    JobExecution.job_id == pipeline_job.id
                ).order_by(JobExecution.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        "execution_id": exec.id,
                        "status": exec.status,
                        "started_at": exec.started_at.isoformat() if exec.started_at else None,
                        "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                        "duration_seconds": exec.duration_seconds,
                        "error_message": exec.error_message,
                        "execution_context": exec.execution_context,
                        "execution_result": exec.execution_result,
                    }
                    for exec in executions
                ]
        except Exception as e:
            self.logger.error(f"Failed to get execution history: {e}")
            return [] 