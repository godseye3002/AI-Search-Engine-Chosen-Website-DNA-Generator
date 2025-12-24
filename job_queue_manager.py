"""
Job Queue Manager for GodsEye Pipeline

Manages job queues between pipeline stages, handles filtering,
and provides batch processing capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from pipeline_models import Job, PipelineRun
from utils.batch_calculator import calculate_batches, create_batches_with_items, BatchInfo


@dataclass
class JobQueue:
    """Represents a queue of jobs for a specific stage"""
    stage: int
    jobs: List[Job]
    name: str
    
    def __len__(self):
        return len(self.jobs)
    
    def add_job(self, job: Job):
        """Add a job to the queue"""
        self.jobs.append(job)
    
    def remove_job(self, job_id: str) -> Optional[Job]:
        """Remove and return a job from the queue"""
        for i, job in enumerate(self.jobs):
            if job.job_id == job_id:
                return self.jobs.pop(i)
        return None
    
    def get_pending_jobs(self) -> List[Job]:
        """Get jobs with pending status for this stage"""
        stage_status_map = {
            1: 'stage_1_status',
            2: 'stage_2_status', 
            3: 'stage_3_status'
        }
        
        status_field = stage_status_map.get(self.stage)
        if not status_field:
            return []
        
        return [job for job in self.jobs if getattr(job, status_field) == 'pending']
    
    def get_completed_jobs(self) -> List[Job]:
        """Get jobs with completed status for this stage"""
        stage_status_map = {
            1: 'stage_1_status',
            2: 'stage_2_status',
            3: 'stage_3_status'
        }
        
        status_field = stage_status_map.get(self.stage)
        if not status_field:
            return []
        
        return [job for job in self.jobs if getattr(job, status_field) == 'completed']
    
    def get_failed_jobs(self) -> List[Job]:
        """Get jobs with failed status for this stage"""
        stage_status_map = {
            1: 'stage_1_status',
            2: 'stage_2_status',
            3: 'stage_3_status'
        }
        
        status_field = stage_status_map.get(self.stage)
        if not status_field:
            return []
        
        return [job for job in self.jobs if getattr(job, status_field) == 'failed']


class JobQueueManager:
    """Manages job queues across pipeline stages"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage-specific parallel limits
        self.max_parallel_stage_1 = config.get('pipeline', {}).get('max_parallel_workers_stage_1', 10)
        self.max_parallel_stage_2 = config.get('pipeline', {}).get('max_parallel_workers_stage_2', 10)
        self.max_parallel_stage_3 = config.get('pipeline', {}).get('max_parallel_workers_stage_3', 5)
        
        self.logger.info("Job Queue Manager initialized")
    
    def create_stage_1_queue(self, run: PipelineRun) -> JobQueue:
        """
        Create queue for Stage 1 from all jobs in the run.
        
        Args:
            run: Pipeline run with jobs
            
        Returns:
            JobQueue with all jobs ready for Stage 1
        """
        all_jobs = list(run.jobs.values())
        
        # Sort by position to maintain order
        all_jobs.sort(key=lambda job: job.position)
        
        queue = JobQueue(
            stage=1,
            jobs=all_jobs,
            name="Stage 1 - Website Classification"
        )
        
        self.logger.info(f"Created Stage 1 queue with {len(queue)} jobs")
        return queue
    
    def create_stage_2_queue(self, run: PipelineRun) -> JobQueue:
        """
        Create queue for Stage 2 from jobs that completed Stage 1 successfully
        and were selected for Stage 2.
        
        Args:
            run: Pipeline run with completed Stage 1 jobs
            
        Returns:
            JobQueue with jobs ready for Stage 2
        """
        eligible_jobs = []
        
        for job in run.jobs.values():
            # Check if Stage 1 completed and job is selected for Stage 2
            if (job.stage_1_status == 'completed' and 
                job.selected_for_stage_2):
                eligible_jobs.append(job)
        
        # Sort by position to maintain order
        eligible_jobs.sort(key=lambda job: job.position)
        
        queue = JobQueue(
            stage=2,
            jobs=eligible_jobs,
            name="Stage 2 - DNA Analysis"
        )
        
        self.logger.info(f"Created Stage 2 queue with {len(queue)} jobs (from {len(run.jobs)} total)")
        return queue
    
    def create_stage_3_queue(self, run: PipelineRun) -> JobQueue:
        """
        Create queue for Stage 3 from jobs that completed Stage 2 successfully.
        
        Args:
            run: Pipeline run with completed Stage 2 jobs
            
        Returns:
            JobQueue with jobs ready for Stage 3
        """
        eligible_jobs = []
        
        for job in run.jobs.values():
            # Check if Stage 2 completed
            if job.stage_2_status == 'completed':
                eligible_jobs.append(job)
        
        # Sort by position to maintain order
        eligible_jobs.sort(key=lambda job: job.position)
        
        queue = JobQueue(
            stage=3,
            jobs=eligible_jobs,
            name="Stage 3 - Final Aggregation"
        )
        
        self.logger.info(f"Created Stage 3 queue with {len(queue)} jobs (from {len(run.jobs)} total)")
        return queue
    
    def create_batches_for_stage(self, queue: JobQueue) -> List[BatchInfo]:
        """
        Create batches for processing a stage queue.
        
        Args:
            queue: JobQueue for a specific stage
            
        Returns:
            List of BatchInfo objects
        """
        max_parallel = self._get_max_parallel_for_stage(queue.stage)
        pending_jobs = queue.get_pending_jobs()
        
        batches = create_batches_with_items(pending_jobs, max_parallel)
        
        self.logger.info(f"Created {len(batches)} batches for {queue.name} "
                         f"(max_parallel={max_parallel}, pending_jobs={len(pending_jobs)})")
        
        return batches
    
    def _get_max_parallel_for_stage(self, stage: int) -> int:
        """Get maximum parallel workers for a stage"""
        if stage == 1:
            return self.max_parallel_stage_1
        elif stage == 2:
            return self.max_parallel_stage_2
        elif stage == 3:
            return self.max_parallel_stage_3
        else:
            return 5  # Default
    
    def filter_stage_1_results(self, run: PipelineRun) -> Tuple[int, int]:
        """
        Filter Stage 1 results to determine which jobs proceed to Stage 2.
        
        Args:
            run: Pipeline run with completed Stage 1 jobs
            
        Returns:
            Tuple of (selected_count, rejected_count)
        """
        selected_count = 0
        rejected_count = 0
        
        for job in run.jobs.values():
            if job.stage_1_status == 'completed':
                # Check classification result
                if job.classification in ['third_party', 'competitor']:
                    # These classifications should proceed to DNA analysis
                    job.selected_for_stage_2 = True
                    selected_count += 1
                elif job.classification == 'special_url':
                    # Special URLs are skipped for DNA analysis
                    job.selected_for_stage_2 = False
                    job.stage_2_status = 'skipped'
                    rejected_count += 1
                else:
                    # Unknown classification, skip for safety
                    job.selected_for_stage_2 = False
                    job.stage_2_status = 'skipped'
                    rejected_count += 1
            else:
                # Failed or not completed jobs don't proceed
                job.selected_for_stage_2 = False
                rejected_count += 1
        
        self.logger.info(f"Stage 1 filtering: {selected_count} selected, {rejected_count} rejected for Stage 2")
        
        return selected_count, rejected_count
    
    def filter_stage_2_results(self, run: PipelineRun) -> Tuple[int, int]:
        """
        Filter Stage 2 results to select jobs for Stage 3.
        
        Args:
            run: Pipeline run with completed Stage 2 jobs
            
        Returns:
            Tuple of (selected_count, rejected_count)
        """
        completed_jobs = []
        failed_jobs = []
        
        for job in run.jobs.values():
            if job.stage_2_status == 'completed':
                completed_jobs.append(job)
            elif job.stage_2_status == 'failed':
                failed_jobs.append(job)
        
        # Filter criteria for Stage 3
        selected_jobs = []
        rejected_jobs = []
        
        for job in completed_jobs:
            # Load DNA analysis results if available
            should_select = self._evaluate_stage_2_job(job)
            
            if should_select:
                job.selected_for_stage_3 = True
                selected_jobs.append(job)
            else:
                rejected_jobs.append(job)
        
        # Update job statuses
        for job in rejected_jobs:
            job.stage_2_status = 'skipped'
        
        selected_count = len(selected_jobs)
        rejected_count = len(rejected_jobs) + len(failed_jobs)
        
        self.logger.info(f"Stage 2 filtering: {selected_count} selected, {rejected_count} rejected for Stage 3")
        
        return selected_count, rejected_count
    
    def _evaluate_stage_2_job(self, job: Job) -> bool:
        """
        Evaluate if a Stage 2 job should proceed to Stage 3.
        
        Args:
            job: Job with completed Stage 2 analysis
            
        Returns:
            True if job should proceed to Stage 3
        """
        # Default selection logic - can be enhanced with DNA analysis quality metrics
        # For now, select all completed Stage 2 jobs
        return job.stage_2_status == 'completed'
    
    def get_queue_summary(self, queue: JobQueue) -> Dict[str, Any]:
        """
        Get summary statistics for a job queue.
        
        Args:
            queue: JobQueue to summarize
            
        Returns:
            Dictionary with queue statistics
        """
        pending = queue.get_pending_jobs()
        completed = queue.get_completed_jobs()
        failed = queue.get_failed_jobs()
        
        # Calculate success rate
        total_processed = len(completed) + len(failed)
        success_rate = len(completed) / total_processed if total_processed > 0 else 0
        
        return {
            'stage': queue.stage,
            'name': queue.name,
            'total_jobs': len(queue.jobs),
            'pending': len(pending),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': success_rate,
            'total_processed': total_processed
        }
    
    def validate_queue_health(self, queue: JobQueue) -> Dict[str, Any]:
        """
        Validate queue health and identify potential issues.
        
        Args:
            queue: JobQueue to validate
            
        Returns:
            Dictionary with health check results
        """
        issues = []
        warnings = []
        
        # Check for jobs stuck in running state
        running_jobs = [job for job in queue.jobs 
                       if getattr(job, f'stage_{queue.stage}_status') == 'running']
        
        if running_jobs:
            warnings.append(f"{len(running_jobs)} jobs stuck in 'running' state")
        
        # Check for high failure rate
        failed_jobs = queue.get_failed_jobs()
        completed_jobs = queue.get_completed_jobs()
        total_processed = len(failed_jobs) + len(completed_jobs)
        
        if total_processed > 0:
            failure_rate = len(failed_jobs) / total_processed
            if failure_rate > 0.3:  # 30% failure rate threshold
                issues.append(f"High failure rate: {failure_rate:.1%}")
        
        # Check for no jobs
        if len(queue.jobs) == 0:
            issues.append("No jobs in queue")
        
        # Check for no pending jobs when queue should have work
        pending_jobs = queue.get_pending_jobs()
        if len(pending_jobs) == 0 and len(completed_jobs) < len(queue.jobs):
            warnings.append("No pending jobs but queue not fully processed")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'summary': self.get_queue_summary(queue)
        }


if __name__ == "__main__":
    # Test job queue manager
    import yaml
    from datetime import datetime
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test queue manager
    manager = JobQueueManager(config)
    
    print("=== Job Queue Manager Test ===")
    
    # Create test run and jobs
    from pipeline_models import PipelineRun, Job
    
    run = PipelineRun(
        run_id="test_run",
        created_at=datetime.now(),
        query="test query",
        total_links=5
    )
    
    # Add test jobs
    for i in range(5):
        job = Job(
            job_id=f"job_{i+1}",
            run_id="test_run",
            source_link={"url": f"http://example.com/{i+1}", "text": f"Link {i+1}"},
            url=f"http://example.com/{i+1}",
            text=f"Link {i+1}",
            position=i+1
        )
        
        # Simulate some completed jobs
        if i < 3:
            job.stage_1_status = 'completed'
            job.classification = 'third_party' if i % 2 == 0 else 'special_url'
        
        run.jobs[job.job_id] = job
    
    # Test queue creation
    stage1_queue = manager.create_stage_1_queue(run)
    print(f"Stage 1 queue: {len(stage1_queue)} jobs")
    
    # Test filtering
    selected, rejected = manager.filter_stage_1_results(run)
    print(f"Filtering results: {selected} selected, {rejected} rejected")
    
    # Test Stage 2 queue
    stage2_queue = manager.create_stage_2_queue(run)
    print(f"Stage 2 queue: {len(stage2_queue)} jobs")
    
    # Test batch creation
    batches = manager.create_batches_for_stage(stage1_queue)
    print(f"Created {len(batches)} batches for Stage 1")
    
    # Test queue summary
    summary = manager.get_queue_summary(stage1_queue)
    print(f"Stage 1 summary: {summary}")
    
    # Test health check
    health = manager.validate_queue_health(stage1_queue)
    print(f"Stage 1 health: {'Healthy' if health['healthy'] else 'Issues found'}")
    if health['issues']:
        print(f"Issues: {health['issues']}")
    if health['warnings']:
        print(f"Warnings: {health['warnings']}")
