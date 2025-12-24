"""
Batch Processor

Handles processing multiple queries in parallel or sequential mode.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pipeline_orchestrator import PipelineOrchestrator
from pipeline_models import PipelineRun


@dataclass
class BatchJob:
    """Individual job in a batch"""
    job_id: str
    query: str
    ai_response_file: str
    status: str = "pending"  # pending, running, completed, failed
    run_id: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None


@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_processing_time: float
    results: List[Dict[str, Any]]
    errors: List[str]
    start_time: datetime
    end_time: datetime


class BatchProcessor:
    """Processes multiple pipeline runs efficiently"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Batch processing settings
        self.max_concurrent_runs = config.get('batch_processing', {}).get('max_concurrent_runs', 3)
        self.timeout_per_run = config.get('batch_processing', {}).get('timeout_per_run', 1800)  # 30 minutes
        self.output_dir = config.get('batch_processing', {}).get('output_dir', 'batch_results')
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        
        # Create output directory
        self.batch_output_dir = os.path.join(self.base_output_dir, self.output_dir)
        os.makedirs(self.batch_output_dir, exist_ok=True)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_runs)
        
        # Active batches
        self.active_batches: Dict[str, List[BatchJob]] = {}
        
        self.logger.info("Batch Processor initialized")
    
    def create_batch(self, batch_id: str, queries: List[str], 
                    ai_response_files: List[str] = None) -> List[BatchJob]:
        """
        Create a batch of jobs for processing.
        
        Args:
            batch_id: Unique identifier for the batch
            queries: List of queries to process
            ai_response_files: List of AI response files (optional, uses default if None)
            
        Returns:
            List of BatchJob objects
        """
        if ai_response_files is None:
            ai_response_files = ["ai_response.json"] * len(queries)
        
        if len(queries) != len(ai_response_files):
            raise ValueError("Queries and AI response files must have same length")
        
        jobs = []
        for i, (query, ai_file) in enumerate(zip(queries, ai_response_files)):
            job = BatchJob(
                job_id=f"{batch_id}_job_{i+1:03d}",
                query=query,
                ai_response_file=ai_file
            )
            jobs.append(job)
        
        self.active_batches[batch_id] = jobs
        self.logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")
        
        return jobs
    
    def process_batch_parallel(self, batch_id: str, 
                             progress_callback: Optional[Callable] = None) -> BatchResult:
        """
        Process batch jobs in parallel.
        
        Args:
            batch_id: Batch identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with processing outcomes
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        jobs = self.active_batches[batch_id]
        start_time = datetime.now()
        
        self.logger.info(f"Starting parallel processing for batch {batch_id} ({len(jobs)} jobs)")
        
        # Submit all jobs to thread pool
        futures = {}
        for job in jobs:
            if job.status == "pending":
                future = self.executor.submit(self._process_single_job, job)
                futures[future] = job
                job.status = "running"
                job.started_at = datetime.now()
        
        # Process completed jobs
        completed_count = 0
        failed_count = 0
        results = []
        errors = []
        
        for future in as_completed(futures, timeout=self.timeout_per_run):
            job = futures[future]
            
            try:
                result = future.result()
                job.status = "completed"
                job.completed_at = datetime.now()
                job.processing_time = (job.completed_at - job.started_at).total_seconds()
                job.run_id = result.get('run_id')
                results.append(result)
                completed_count += 1
                
                self.logger.info(f"Batch {batch_id} job {job.job_id} completed: {result.get('run_id')}")
                
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now()
                job.processing_time = (job.completed_at - job.started_at).total_seconds()
                errors.append(f"Job {job.job_id} failed: {str(e)}")
                failed_count += 1
                
                self.logger.error(f"Batch {batch_id} job {job.job_id} failed: {e}")
            
            # Call progress callback if provided
            if progress_callback:
                progress = (completed_count + failed_count) / len(jobs) * 100
                progress_callback(batch_id, job.job_id, job.status, progress)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            total_jobs=len(jobs),
            completed_jobs=completed_count,
            failed_jobs=failed_count,
            total_processing_time=total_time,
            results=results,
            errors=errors,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save batch results
        self._save_batch_results(batch_result)
        
        self.logger.info(f"Batch {batch_id} completed: {completed_count} successful, {failed_count} failed")
        
        return batch_result
    
    def process_batch_sequential(self, batch_id: str,
                               progress_callback: Optional[Callable] = None) -> BatchResult:
        """
        Process batch jobs sequentially (one at a time).
        
        Args:
            batch_id: Batch identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with processing outcomes
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        jobs = self.active_batches[batch_id]
        start_time = datetime.now()
        
        self.logger.info(f"Starting sequential processing for batch {batch_id} ({len(jobs)} jobs)")
        
        completed_count = 0
        failed_count = 0
        results = []
        errors = []
        
        for i, job in enumerate(jobs):
            if job.status != "pending":
                continue
            
            job.status = "running"
            job.started_at = datetime.now()
            
            try:
                result = self._process_single_job(job)
                job.status = "completed"
                job.completed_at = datetime.now()
                job.processing_time = (job.completed_at - job.started_at).total_seconds()
                job.run_id = result.get('run_id')
                results.append(result)
                completed_count += 1
                
                self.logger.info(f"Batch {batch_id} job {job.job_id} completed: {result.get('run_id')}")
                
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now()
                job.processing_time = (job.completed_at - job.started_at).total_seconds()
                errors.append(f"Job {job.job_id} failed: {str(e)}")
                failed_count += 1
                
                self.logger.error(f"Batch {batch_id} job {job.job_id} failed: {e}")
            
            # Call progress callback if provided
            if progress_callback:
                progress = (i + 1) / len(jobs) * 100
                progress_callback(batch_id, job.job_id, job.status, progress)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            total_jobs=len(jobs),
            completed_jobs=completed_count,
            failed_jobs=failed_count,
            total_processing_time=total_time,
            results=results,
            errors=errors,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save batch results
        self._save_batch_results(batch_result)
        
        self.logger.info(f"Batch {batch_id} completed: {completed_count} successful, {failed_count} failed")
        
        return batch_result
    
    def _process_single_job(self, job: BatchJob) -> Dict[str, Any]:
        """
        Process a single job (pipeline run).
        
        Args:
            job: BatchJob to process
            
        Returns:
            Dictionary with run results
        """
        # Create orchestrator instance for this thread
        orchestrator = PipelineOrchestrator()
        
        try:
            # Load AI response
            ai_response = orchestrator.load_ai_response(job.ai_response_file)
            
            # Override query if provided
            if job.query:
                ai_response['query'] = job.query
            
            # Create and run pipeline
            run_id = orchestrator.create_run(ai_response)
            orchestrator.run_pipeline()
            
            return {
                'job_id': job.job_id,
                'run_id': run_id,
                'query': job.query,
                'status': 'completed'
            }
            
        except Exception as e:
            raise Exception(f"Pipeline execution failed: {str(e)}")
    
    def _save_batch_results(self, result: BatchResult):
        """
        Save batch results to file.
        
        Args:
            result: BatchResult to save
        """
        # Create batch-specific directory
        batch_dir = os.path.join(self.batch_output_dir, f"batch_{result.batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save batch summary
        summary_file = os.path.join(batch_dir, 'batch_summary.json')
        summary_data = {
            'batch_id': result.batch_id,
            'total_jobs': result.total_jobs,
            'completed_jobs': result.completed_jobs,
            'failed_jobs': result.failed_jobs,
            'total_processing_time': result.total_processing_time,
            'success_rate': result.completed_jobs / result.total_jobs if result.total_jobs > 0 else 0,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'errors': result.errors
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save individual job results
        jobs_file = os.path.join(batch_dir, 'job_results.json')
        jobs_data = []
        
        for batch_job in self.active_batches[result.batch_id]:
            jobs_data.append({
                'job_id': batch_job.job_id,
                'query': batch_job.query,
                'status': batch_job.status,
                'run_id': batch_job.run_id,
                'error': batch_job.error,
                'started_at': batch_job.started_at.isoformat() if batch_job.started_at else None,
                'completed_at': batch_job.completed_at.isoformat() if batch_job.completed_at else None,
                'processing_time': batch_job.processing_time
            })
        
        with open(jobs_file, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f, indent=2, ensure_ascii=False)
        
        # Save detailed results for completed jobs
        results_data = []
        for result_item in result.results:
            if result_item.get('run_id'):
                try:
                    # Load final aggregation results
                    results_dir = os.path.join('outputs', 'stage_3_results', f"run_{result_item['run_id']}")
                    aggregation_file = os.path.join(results_dir, 'final_aggregation.json')
                    
                    if os.path.exists(aggregation_file):
                        with open(aggregation_file, 'r', encoding='utf-8') as f:
                            aggregation_data = json.load(f)
                        
                        results_data.append({
                            'job_id': result_item['job_id'],
                            'run_id': result_item['run_id'],
                            'query': result_item['query'],
                            'aggregation_results': aggregation_data
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to load results for run {result_item.get('run_id')}: {e}")
        
        if results_data:
            detailed_results_file = os.path.join(batch_dir, 'detailed_results.json')
            with open(detailed_results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved batch results for batch {result.batch_id} to {batch_dir}")
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Dictionary with batch status
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        jobs = self.active_batches[batch_id]
        
        status_counts = {
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0
        }
        
        for job in jobs:
            status_counts[job.status] += 1
        
        return {
            'batch_id': batch_id,
            'total_jobs': len(jobs),
            'status_counts': status_counts,
            'progress': (status_counts['completed'] + status_counts['failed']) / len(jobs) * 100
        }
    
    def load_batch_from_file(self, batch_file: str) -> List[BatchJob]:
        """
        Load batch configuration from file.
        
        Args:
            batch_file: Path to batch configuration file
            
        Returns:
            List of BatchJob objects
        """
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_config = json.load(f)
            
            batch_id = batch_config.get('batch_id', f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            queries = batch_config.get('queries', [])
            ai_response_files = batch_config.get('ai_response_files', [])
            
            return self.create_batch(batch_id, queries, ai_response_files)
            
        except Exception as e:
            raise Exception(f"Failed to load batch file: {str(e)}")
    
    def create_batch_from_directory(self, batch_dir: str) -> List[BatchJob]:
        """
        Create batch from directory containing multiple AI response files.
        
        Args:
            batch_dir: Directory containing AI response files
            
        Returns:
            List of BatchJob objects
        """
        if not os.path.exists(batch_dir):
            raise ValueError(f"Directory {batch_dir} does not exist")
        
        # Find all JSON files in directory
        json_files = []
        for file in os.listdir(batch_dir):
            if file.endswith('.json'):
                json_files.append(os.path.join(batch_dir, file))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {batch_dir}")
        
        # Load queries from files
        queries = []
        ai_files = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    ai_response = json.load(f)
                
                query = ai_response.get('query', f"Query from {os.path.basename(json_file)}")
                queries.append(query)
                ai_files.append(json_file)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        if not queries:
            raise ValueError("No valid queries found in files")
        
        batch_id = f"batch_{os.path.basename(batch_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.create_batch(batch_id, queries, ai_files)
    
    def cleanup_batch(self, batch_id: str):
        """
        Clean up completed batch from memory.
        
        Args:
            batch_id: Batch identifier
        """
        if batch_id in self.active_batches:
            del self.active_batches[batch_id]
            self.logger.info(f"Cleaned up batch {batch_id} from memory")
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self.executor.shutdown(wait=True)
        self.logger.info("Batch processor shutdown complete")


# Utility functions for batch processing
def create_batch_config_file(batch_file: str, queries: List[str], 
                           ai_response_files: List[str] = None):
    """
    Create a batch configuration file.
    
    Args:
        batch_file: Path to save configuration
        queries: List of queries
        ai_response_files: List of AI response files (optional)
    """
    if ai_response_files is None:
        ai_response_files = ["ai_response.json"] * len(queries)
    
    config = {
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'queries': queries,
        'ai_response_files': ai_response_files,
        'created_at': datetime.now().isoformat()
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create batch processor
    processor = BatchProcessor(config)
    
    print("=== Batch Processor Test ===")
    
    # Example batch
    queries = [
        "best SEO tools for content marketing",
        "how to improve website ranking",
        "content analysis techniques"
    ]
    
    # Create batch
    batch_id = f"test_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = processor.create_batch(batch_id, queries)
    
    print(f"Created batch {batch_id} with {len(jobs)} jobs")
    
    # Process batch (sequential for testing)
    def progress_callback(batch_id, job_id, status, progress):
        print(f"Progress: {batch_id} - {job_id} - {status} - {progress:.1f}%")
    
    try:
        result = processor.process_batch_sequential(batch_id, progress_callback)
        
        print(f"Batch completed:")
        print(f"  Total jobs: {result.total_jobs}")
        print(f"  Completed: {result.completed_jobs}")
        print(f"  Failed: {result.failed_jobs}")
        print(f"  Processing time: {result.total_processing_time:.2f}s")
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
    
    finally:
        processor.cleanup_batch(batch_id)
        processor.shutdown()
