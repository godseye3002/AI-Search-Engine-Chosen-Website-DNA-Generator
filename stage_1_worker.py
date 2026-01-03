"""
Stage 1 Worker - Website Classification

Handles individual job processing for Stage 1 of the pipeline.
Includes timeout protection and error handling.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from classification_core import classify_website, ClassificationResult
from utils.timeout_handler import execute_with_timeout, TimeoutResult
from utils.env_utils import is_production_mode
from pipeline_models import Job

# Configure logging based on environment
log_level = logging.INFO if not is_production_mode() else logging.ERROR
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [STAGE1] - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1Worker:
    """Worker for Stage 1: Website Classification"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage 1 specific settings
        self.timeout_per_link = config.get('stage_1_classification', {}).get('timeout_per_link', 30)
        self.output_dir = config.get('stage_1_classification', {}).get('output_dir', 'stage_1_results')
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        
        self.logger.info("Stage 1 Worker initialized")
    
    def process_job(self, job: Job) -> Job:
        """
        Process a single job through Stage 1 classification.
        
        Args:
            job: Job object to process
            
        Returns:
            Updated Job object with results
        """
        self.logger.info(f"Processing job {job.job_id} for Stage 1")
        self.logger.info(f"Job URL: {job.url}")
        self.logger.info(f"Job text: {job.text[:100]}..." if len(job.text) > 100 else f"Job text: {job.text}")
        
        # Mark job as started
        job.mark_stage_start(1)
        
        try:
            # Prepare input data for classification
            input_data = {
                'url': job.url,
                'text': job.text,
                'raw_url': job.source_link.get('raw_url'),
                'snippet': job.source_link.get('snippet'),
                'position': job.position,
                'related_to': job.source_link.get('related_to'),
                'related_claim': job.source_link.get('related_claim'),
                'extraction_order': job.source_link.get('extraction_order')
            }
            
            self.logger.info(f"Prepared input data for classification: {list(input_data.keys())}")
            
            # Execute classification with timeout
            self.logger.info(f"Starting classification with timeout: {self.timeout_per_link}s")
            result = execute_with_timeout(
                classify_website,
                args=(input_data,),
                timeout=self.timeout_per_link
            )
            
            # Process result
            if result.status == TimeoutResult.SUCCESS:
                classification_result = result.result
                self.logger.info(f"Classification successful: {classification_result.classification}")
                
                # Update job with results
                job.classification = classification_result.classification
                job.stage_1_error = classification_result.error
                job.stage_1_data = classification_result.to_dict()
                
                # Mark as completed (in-memory pipeline; no output_path)
                job.mark_stage_complete(1, None)
                
                self.logger.info(f"Job {job.job_id} completed Stage 1: {classification_result.classification}")
                
            elif result.status == TimeoutResult.TIMEOUT:
                error_msg = f"Stage 1 timeout after {self.timeout_per_link}s"
                job.mark_stage_failed(1, error_msg)
                self.logger.warning(f"Job {job.job_id} failed Stage 1: {error_msg}")
                
            else:  # ERROR
                error_msg = f"Stage 1 error: {result.error_message}"
                job.mark_stage_failed(1, error_msg)
                self.logger.error(f"Job {job.job_id} failed Stage 1: {error_msg}")
        
        except Exception as e:
            error_msg = f"Unexpected error in Stage 1 ({type(e).__name__}): {str(e)}"
            job.mark_stage_failed(1, error_msg)
            self.logger.exception(f"Job {job.job_id} failed Stage 1 with unexpected error")
            try:
                from error_email_sender import send_ai_error_email
                send_ai_error_email(
                    error=e,
                    error_context="Stage 1 worker failed while classifying a URL",
                    metadata={
                        "product_id": getattr(job, 'product_id', None),
                        "run_id": getattr(job, 'run_id', None),
                        "job_id": getattr(job, 'job_id', None),
                        "stage": "stage_1_classification",
                        "url": getattr(job, 'url', None),
                        "extra": {
                            "timeout_per_link": getattr(self, 'timeout_per_link', None),
                        },
                    },
                )
            except Exception as email_err:
                self.logger.error(f"Failed to send error email for job {job.job_id}: {email_err}")
        
        return job
    
    def _save_job_outputs(self, job_id: str, result: ClassificationResult) -> Dict[str, str]:
        """
        Save classification results to files.
        
        Args:
            job_id: Unique job identifier
            result: Classification result
            
        Returns:
            Dictionary with file paths
        """
        return {}
    
    def process_batch(self, jobs: list) -> list:
        """
        Process a batch of jobs in parallel.
        
        Args:
            jobs: List of Job objects to process
            
        Returns:
            List of updated Job objects
        """
        self.logger.info(f"Processing batch of {len(jobs)} jobs for Stage 1")
        
        max_workers = self.config.get('pipeline', {}).get('max_parallel_workers_stage_1', 10)

        processed_jobs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_job, job): job for job in jobs}
            for future in as_completed(futures):
                processed_jobs.append(future.result())
        
        # Count results
        completed = sum(1 for job in processed_jobs if job.stage_1_status == 'completed')
        failed = sum(1 for job in processed_jobs if job.stage_1_status == 'failed')
        
        self.logger.info(f"Batch completed: {completed} successful, {failed} failed")
        
        return processed_jobs
    
    def get_job_summary(self, jobs: list) -> Dict[str, Any]:
        """
        Get summary statistics for a batch of jobs.
        
        Args:
            jobs: List of Job objects
            
        Returns:
            Dictionary with summary statistics
        """
        total = len(jobs)
        completed = sum(1 for job in jobs if job.stage_1_status == 'completed')
        failed = sum(1 for job in jobs if job.stage_1_status == 'failed')
        pending = sum(1 for job in jobs if job.stage_1_status == 'pending')
        running = sum(1 for job in jobs if job.stage_1_status == 'running')
        
        # Classification breakdown
        classifications = {}
        for job in jobs:
            if job.classification:
                classifications[job.classification] = classifications.get(job.classification, 0) + 1
        
        return {
            'total_jobs': total,
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'running': running,
            'success_rate': completed / total if total > 0 else 0,
            'classifications': classifications
        }


if __name__ == "__main__":
    # Test Stage 1 worker
    import yaml
    from datetime import datetime
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test worker
    worker = Stage1Worker(config)
    
    print("=== Stage 1 Worker Test ===")
    
    # Create test job
    from pipeline_models import Job
    
    test_job = Job(
        job_id="test_job_1",
        run_id="test_run",
        source_link={
            "url": "https://example.com",
            "text": "Example website",
            "related_to": "test search query"
        },
        url="https://example.com",
        text="Example website",
        position=1
    )
    
    # Process job
    processed_job = worker.process_job(test_job)
    
    print(f"Job status: {processed_job.stage_1_status}")
    print(f"Classification: {processed_job.classification}")
    if processed_job.stage_1_error:
        print(f"Error: {processed_job.stage_1_error}")
    if processed_job.stage_1_output_path:
        print(f"Output saved to: {processed_job.stage_1_output_path}")
