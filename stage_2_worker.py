"""
Stage 2 Worker - DNA Analysis

Handles DNA analysis for jobs that passed Stage 1 filtering.
Processes classified data to extract winning content DNA.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dna_analysis_core import analyze_website_dna, DNAAnalysisResult
from utils.timeout_handler import execute_with_timeout, TimeoutResult
from utils.env_utils import is_production_mode
from pipeline_models import Job

# Configure logging based on environment
log_level = logging.INFO if not is_production_mode() else logging.ERROR
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [STAGE2] - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2Worker:
    """Worker for Stage 2: DNA Analysis"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage 2 specific settings
        self.timeout_per_job = config.get('stage_2_dna_analysis', {}).get('timeout_per_job', 120)
        self.output_dir = config.get('stage_2_dna_analysis', {}).get('output_dir', 'stage_2_results')
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        
        self.logger.info("Stage 2 Worker initialized")
    
    def process_job(self, job: Job, ai_response: Dict[str, Any]) -> Job:
        """
        Process a single job through Stage 2 DNA analysis.
        
        Args:
            job: Job object to process (must have completed Stage 1)
            ai_response: Original AI response for context
            
        Returns:
            Updated Job object with DNA analysis results
        """
        self.logger.info(f"Processing job {job.job_id} for Stage 2")
        
        # Mark job as started
        job.mark_stage_start(2)
        
        try:
            # Load Stage 1 classified data
            classified_data = self._load_stage_1_output(job)
            
            # Execute DNA analysis with timeout
            result = execute_with_timeout(
                analyze_website_dna,
                args=(classified_data, ai_response),
                timeout=self.timeout_per_job
            )
            
            # Process result
            if result.status == TimeoutResult.SUCCESS:
                dna_result = result.result
                job.stage_2_data = dna_result.to_dict()
                
                # Mark as completed (in-memory pipeline; no output_path)
                job.mark_stage_complete(2, None)
                
                self.logger.info(f"Job {job.job_id} completed Stage 2: DNA analysis complete")
                
            elif result.status == TimeoutResult.TIMEOUT:
                error_msg = f"Stage 2 timeout after {self.timeout_per_job}s"
                job.mark_stage_failed(2, error_msg)
                self.logger.warning(f"Job {job.job_id} failed Stage 2: {error_msg}")
                
            else:  # ERROR
                error_msg = f"Stage 2 error: {result.error_message}"
                job.mark_stage_failed(2, error_msg)
                self.logger.error(f"Job {job.job_id} failed Stage 2: {error_msg}")
        
        except Exception as e:
            error_msg = f"Unexpected error in Stage 2 ({type(e).__name__}): {str(e)}"
            job.mark_stage_failed(2, error_msg)
            self.logger.exception(f"Job {job.job_id} failed Stage 2 with unexpected error")
            try:
                from error_email_sender import send_ai_error_email
                send_ai_error_email(
                    error=e,
                    error_context="Stage 2 worker failed while running DNA analysis",
                    metadata={
                        "product_id": getattr(job, 'product_id', None),
                        "run_id": getattr(job, 'run_id', None),
                        "job_id": getattr(job, 'job_id', None),
                        "stage": "stage_2_dna_analysis",
                        "url": getattr(job, 'url', None),
                        "extra": {
                            "timeout_per_job": getattr(self, 'timeout_per_job', None),
                            "stage_1_output_path": getattr(job, 'stage_1_output_path', None),
                        },
                    },
                )
            except Exception as email_err:
                self.logger.error(f"Failed to send error email for job {job.job_id}: {email_err}")
        
        return job
    
    def _load_stage_1_output(self, job: Job) -> Optional[Dict[str, Any]]:
        """
        Load Stage 1 classified data for this job.
        
        Args:
            job: Job object with Stage 1 output path
            
        Returns:
            Classified data dictionary or None if not found
        """
        if not job.stage_1_data:
            raise FileNotFoundError(f"Stage 1 output missing in-memory for job {job.job_id}")

        classified_data = dict(job.stage_1_data)
        classified_data['job_id'] = job.job_id
        return classified_data
    
    def _save_job_outputs(self, job_id: str, result: DNAAnalysisResult) -> Dict[str, str]:
        """
        Save DNA analysis results to files.
        
        Args:
            job_id: Unique job identifier
            result: DNA analysis result
            
        Returns:
            Dictionary with file paths
        """
        return {}
    
    def process_batch(self, jobs: list, ai_response: Dict[str, Any]) -> list:
        """
        Process a batch of jobs in parallel.
        
        Args:
            jobs: List of Job objects to process
            ai_response: Original AI response for context
            
        Returns:
            List of updated Job objects
        """
        self.logger.info(f"Processing batch of {len(jobs)} jobs for Stage 2")
        
        max_workers = self.config.get('pipeline', {}).get('max_parallel_workers_stage_2', 10)

        processed_jobs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_job, job, ai_response): job for job in jobs}
            for future in as_completed(futures):
                processed_jobs.append(future.result())
        
        # Count results
        completed = sum(1 for job in processed_jobs if job.stage_2_status == 'completed')
        failed = sum(1 for job in processed_jobs if job.stage_2_status == 'failed')
        
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
        completed = sum(1 for job in jobs if job.stage_2_status == 'completed')
        failed = sum(1 for job in jobs if job.stage_2_status == 'failed')
        pending = sum(1 for job in jobs if job.stage_2_status == 'pending')
        running = sum(1 for job in jobs if job.stage_2_status == 'running')
        
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
    # Test Stage 2 worker
    import yaml
    from datetime import datetime
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test worker
    worker = Stage2Worker(config)
    
    print("=== Stage 2 Worker Test ===")
    
    # Create test job with Stage 1 completion
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
    
    # Simulate Stage 1 completion
    test_job.mark_stage_complete(1, "outputs/stage_1_results/job_test_job_1/classified_data.json")
    
    # Load AI response
    try:
        with open('ai_response.json', 'r') as f:
            ai_response = json.load(f)
    except FileNotFoundError:
        ai_response = {"query": "test query", "ai_overview": "test overview"}
    
    # Process job
    processed_job = worker.process_job(test_job, ai_response)
    
    print(f"Job status: {processed_job.stage_2_status}")
    if processed_job.stage_2_error:
        print(f"Error: {processed_job.stage_2_error}")
    if processed_job.stage_2_output_path:
        print(f"Output saved to: {processed_job.stage_2_output_path}")
