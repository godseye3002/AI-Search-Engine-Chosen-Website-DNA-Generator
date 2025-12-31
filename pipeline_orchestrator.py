"""
GodsEye Pipeline Orchestrator

Master control system for the content analysis pipeline.
Manages jobs, stages, batching, and parallel execution.
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from utils.batch_calculator import calculate_batches, create_batches_with_items, BatchInfo
from utils.timeout_handler import execute_with_timeout, ExecutionResult, TimeoutResult
from pipeline_models import Job, PipelineRun
from job_queue_manager import JobQueueManager
from stage_1_worker import Stage1Worker
from stage_2_worker import Stage2Worker
from stage_3_worker import Stage3Worker


class PipelineOrchestrator:
    """Master controller for the GodsEye pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.queue_manager = JobQueueManager(self.config, self.logger)
        self.stage_1_worker = Stage1Worker(self.config, self.logger)
        self.stage_2_worker = Stage2Worker(self.config, self.logger)
        self.stage_3_worker = Stage3Worker(self.config, self.logger)
        
        # Active runs
        self.active_runs: Dict[str, PipelineRun] = {}

        self._ai_response_by_run_id: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Pipeline Orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_file = log_config.get('file', 'pipeline.log')
        
        # Create logs directory
        logs_dir = self.config.get('pipeline', {}).get('logs_dir', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, log_file)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _create_directories(self):
        return
    
    def load_ai_response(self, file_path: str = "ai_response.json") -> Dict[str, Any]:
        """Load AI response from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ai_response = json.load(f)
            return ai_response
        except FileNotFoundError:
            raise FileNotFoundError(f"AI response file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in AI response file: {e}")
    
    def create_run(self, ai_response: Dict[str, Any], run_id_override: str = None) -> str:
        """
        Create a new pipeline run from AI response.
        
        Args:
            ai_response: AI response data with source_links
            run_id_override: Optional override for run_id (for batch processing)
            
        Returns:
            run_id for the created run
        """
        # Use provided run_id_override or generate new one
        if run_id_override:
            run_id = run_id_override
        else:
            run_id = f"{self.config.get('pipeline', {}).get('run_id_prefix', 'gods_eye_run')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Extract query and source links
        query = ai_response.get('query', '')
        source_links = ai_response.get('source_links', [])
        
        # Create pipeline run
        run = PipelineRun(
            run_id=run_id,
            created_at=datetime.now(),
            query=query,
            total_links=len(source_links)
        )
        
        # Create jobs for each source link
        for i, link in enumerate(source_links):
            job_id = f"{run_id}_job_{i+1:03d}"
            
            job = Job(
                job_id=job_id,
                run_id=run_id,
                source_link=link,
                url=link.get('url', ''),
                text=link.get('text', ''),
                position=link.get('position', i+1),
                max_retries=self.config.get('pipeline', {}).get('max_retries', 2)
            )
            
            run.jobs[job_id] = job
        
        # Calculate initial batches for Stage 1
        max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_1', 10)
        run.stage_1_batches = calculate_batches(len(source_links), max_parallel)
        
        # Store run
        self.active_runs[run_id] = run
        
        self.logger.info(f"Created run {run_id} with {len(source_links)} jobs")
        self.logger.info(f"Stage 1: {len(run.stage_1_batches)} batches with max {max_parallel} workers each")
        
        return run_id

    def run_pipeline_from_ai_response(self, ai_response: Dict[str, Any], run_id_override: Optional[str] = None) -> str:
        """Run the complete pipeline in-memory from an AI response.

        Concurrency-safe: all state is local to this function.
        """
        self.logger.info("Starting GodsEye pipeline execution")

        run_id = run_id_override or f"{self.config.get('pipeline', {}).get('run_id_prefix', 'gods_eye_run')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        query = ai_response.get('query', '')
        source_links = ai_response.get('source_links', [])

        run = PipelineRun(
            run_id=run_id,
            created_at=datetime.now(),
            query=query,
            total_links=len(source_links),
        )

        for i, link in enumerate(source_links):
            job_id = f"{run_id}_job_{i+1:03d}"
            job = Job(
                job_id=job_id,
                run_id=run_id,
                source_link=link,
                url=link.get('url', ''),
                text=link.get('text', ''),
                position=link.get('position', i+1),
                max_retries=self.config.get('pipeline', {}).get('max_retries', 2)
            )
            run.jobs[job_id] = job

        # Stage 1
        stage1_queue = self.queue_manager.create_stage_1_queue(run)
        stage1_batches = self.queue_manager.create_batches_for_stage(stage1_queue)
        for batch in stage1_batches:
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    batch_jobs.append(run.jobs[job_ref])
            processed = self.stage_1_worker.process_batch(batch_jobs)
            for job in processed:
                run.jobs[job.job_id] = job

        self.queue_manager.filter_stage_1_results(run)

        # Stage 2
        stage2_queue = self.queue_manager.create_stage_2_queue(run)
        stage2_batches = self.queue_manager.create_batches_for_stage(stage2_queue)
        for batch in stage2_batches:
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    batch_jobs.append(run.jobs[job_ref])
            processed = self.stage_2_worker.process_batch(batch_jobs, ai_response)
            for job in processed:
                run.jobs[job.job_id] = job

        self.queue_manager.filter_stage_2_results(run)

        # Stage 3
        stage_3_jobs = []
        for job in run.jobs.values():
            if job.selected_for_stage_3 and job.stage_2_status == 'completed':
                stage_3_jobs.append(job)

        if stage_3_jobs:
            self.stage_3_worker.process_run(run_id, query, stage_3_jobs)

        self.logger.info(f"Pipeline run {run_id} completed")
        return run_id
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline run"""
        if run_id not in self.active_runs:
            return None
        
        run = self.active_runs[run_id]
        job_summary = run.get_summary()
        
        return {
            'run_id': run_id,
            'status': run.status,
            'current_stage': run.current_stage,
            'query': run.query,
            'total_links': run.total_links,
            'created_at': run.created_at.isoformat(),
            'job_summary': job_summary,
            'stage_1_batches': len(run.stage_1_batches),
            'stage_2_batches': len(run.stage_2_batches),
            'stage_3_batches': len(run.stage_3_batches)
        }
    
    def save_run_state(self, run_id: str):
        return
    
    def run_pipeline(self, ai_response_file: str = "ai_response.json") -> str:
        """
        Run the complete pipeline on AI response data.
        
        Args:
            ai_response_file: Path to AI response JSON file
            
        Returns:
            run_id of the executed pipeline
        """
        self.logger.info("Starting GodsEye pipeline execution")
        
        # Load AI response
        ai_response = self.load_ai_response(ai_response_file)
        
        return self.run_pipeline_from_ai_response(ai_response)
    
    def _execute_stage_1(self, run_id: str):
        """
        Execute Stage 1: Website Classification
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Starting Stage 1 execution for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Create Stage 1 queue
        stage1_queue = self.queue_manager.create_stage_1_queue(run)
        
        # Create batches for Stage 1
        batches = self.queue_manager.create_batches_for_stage(stage1_queue)
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing Stage 1 batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch.items)} jobs)")
            
            # Extract jobs for this batch
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    # Handle case where batch contains job_ids instead of job objects
                    batch_jobs.append(run.jobs[job_ref])
            
            # Process batch
            processed_jobs = self.stage_1_worker.process_batch(batch_jobs)
            
            # Update run with processed jobs
            for job in processed_jobs:
                run.jobs[job.job_id] = job
            
            # In-memory pipeline: do not persist state to disk
            
            # Log batch summary
            summary = self.stage_1_worker.get_job_summary(processed_jobs)
            self.logger.info(f"Batch {batch_idx + 1} completed: {summary['completed']} successful, "
                           f"{summary['failed']} failed")
        
        # Log Stage 1 completion
        run_summary = run.get_summary()
        self.logger.info(f"Stage 1 completed for run {run_id}: "
                        f"{run_summary['stage_1_completed']} successful, "
                        f"{run_summary['stage_1_failed']} failed")
    
    def _filter_stage_1_results(self, run_id: str):
        """
        Filter Stage 1 results to determine which jobs proceed to Stage 2.
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Filtering Stage 1 results for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Use queue manager to filter results
        selected_count, rejected_count = self.queue_manager.filter_stage_1_results(run)
        
        # Calculate Stage 2 batches for selected jobs
        if selected_count > 0:
            max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_2', 10)
            run.stage_2_batches = calculate_batches(selected_count, max_parallel)
            self.logger.info(f"Calculated {len(run.stage_2_batches)} Stage 2 batches for {selected_count} selected jobs")
        
        # In-memory pipeline: do not persist state to disk
        
        self.logger.info(f"Stage 1 filtering completed: {selected_count} selected, {rejected_count} rejected")
    
    def _execute_stage_2(self, run_id: str):
        """
        Execute Stage 2: DNA Analysis
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Starting Stage 2 execution for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Create Stage 2 queue
        stage2_queue = self.queue_manager.create_stage_2_queue(run)
        
        if not stage2_queue.jobs:
            self.logger.info(f"No jobs selected for Stage 2 in run {run_id}")
            return
        
        # Create batches for Stage 2
        batches = self.queue_manager.create_batches_for_stage(stage2_queue)
        
        ai_response = self._ai_response_by_run_id.get(run_id) or self.load_ai_response()
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing Stage 2 batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch.items)} jobs)")
            
            # Extract jobs for this batch
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    batch_jobs.append(run.jobs[job_ref])
            
            # Process batch
            processed_jobs = self.stage_2_worker.process_batch(batch_jobs, ai_response)
            
            # Update run with processed jobs
            for job in processed_jobs:
                run.jobs[job.job_id] = job
            
            # In-memory pipeline: do not persist state to disk

            # Log batch summary
            summary = self.stage_2_worker.get_job_summary(processed_jobs)
            self.logger.info(f"Batch {batch_idx + 1} completed: {summary['completed']} successful, "
                           f"{summary['failed']} failed")
        
        # Log Stage 2 completion
        run_summary = run.get_summary()
        self.logger.info(f"Stage 2 completed for run {run_id}: "
                        f"{run_summary['stage_2_completed']} successful, "
                        f"{run_summary['stage_2_failed']} failed")
    
    def _filter_stage_2_results(self, run_id: str):
        """
        Filter Stage 2 results to determine which jobs proceed to Stage 3.
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Filtering Stage 2 results for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Use queue manager to filter results
        selected_count, rejected_count = self.queue_manager.filter_stage_2_results(run)
        
        # Calculate Stage 3 batches for selected jobs
        if selected_count > 0:
            max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_3', 5)
            run.stage_3_batches = calculate_batches(selected_count, max_parallel)
            self.logger.info(f"Calculated {len(run.stage_3_batches)} Stage 3 batches for {selected_count} selected jobs")
        
        # In-memory pipeline: do not persist state to disk

        self.logger.info(f"Stage 2 filtering completed: {selected_count} selected, {rejected_count} rejected")
    
    def _execute_stage_3(self, run_id: str):
        """
        Execute Stage 3: Final Aggregation
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Starting Stage 3 execution for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Get jobs selected for Stage 3
        stage_3_jobs = []
        for job in run.jobs.values():
            if job.selected_for_stage_3 and job.stage_2_status == 'completed':
                stage_3_jobs.append(job)
        
        if not stage_3_jobs:
            self.logger.info(f"No jobs selected for Stage 3 in run {run_id}")
            return
        
        self.logger.info(f"Processing {len(stage_3_jobs)} jobs for final aggregation")
        
        ai_response = self._ai_response_by_run_id.get(run_id) or self.load_ai_response()
        query = ai_response.get('query', '')
        
        # Process final aggregation
        result = self.stage_3_worker.process_run(run_id, query, stage_3_jobs)
        
        # Update run status
        if result['status'] == 'completed':
            run.status = 'completed'
            run.completed_at = datetime.now()
            
            # In-memory pipeline: do not persist state to disk
            
            self.logger.info(f"Stage 3 completed for run {run_id}: "
                           f"{result['total_analyzed']} sources analyzed")
            
            # Log summary
            summary = self.stage_3_worker.get_run_summary(result)
            self.logger.info(f"Final aggregation summary: {summary}")
            
        else:
            run.status = 'failed'
            self.logger.error(f"Stage 3 failed for run {run_id}: {result.get('error', 'Unknown error')}")
        
        # In-memory pipeline: do not persist state to disk



if __name__ == "__main__":
    # Example usage
    orchestrator = PipelineOrchestrator()
    
    print("=== Pipeline Orchestrator Test ===")
    
    try:
        # Test loading AI response
        ai_response = orchestrator.load_ai_response()
        print(f"Loaded AI response with {len(ai_response.get('source_links', []))} source links")
        print(f"Query: {ai_response.get('query', 'N/A')}")
        
        # Test creating a run
        run_id = orchestrator.create_run(ai_response)
        print(f"Created run: {run_id}")
        
        # Test getting run status
        status = orchestrator.get_run_status(run_id)
        print(f"Run status: {status['status']}")
        print(f"Job summary: {status['job_summary']}")
        
    except Exception as e:
        print(f"Error: {e}")
