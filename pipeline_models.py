"""
Pipeline Data Models

Contains dataclasses for Job and PipelineRun to avoid circular imports.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from utils.batch_calculator import BatchInfo


@dataclass
class Job:
    """Represents a single job (one source link) in the pipeline"""
    job_id: str
    run_id: str
    source_link: Dict[str, Any]  # From ai_response source_links
    url: str
    text: str
    position: int
    
    # Classification result
    classification: Optional[str] = None  # third_party, competitor, special_url
    
    # Stage statuses
    stage_1_status: str = "pending"  # pending, running, completed, failed, skipped
    stage_1_output_path: Optional[str] = None
    stage_1_error: Optional[str] = None
    stage_1_start_time: Optional[datetime] = None
    stage_1_end_time: Optional[datetime] = None
    
    stage_2_status: str = "pending"
    stage_2_output_path: Optional[str] = None
    stage_2_error: Optional[str] = None
    stage_2_start_time: Optional[datetime] = None
    stage_2_end_time: Optional[datetime] = None
    
    stage_3_status: str = "pending"
    stage_3_output_path: Optional[str] = None
    stage_3_error: Optional[str] = None
    stage_3_start_time: Optional[datetime] = None
    stage_3_end_time: Optional[datetime] = None
    
    # Retry tracking
    max_retries: int = 2
    retry_count: int = 0
    
    # Selection for next stage
    selected_for_stage_2: bool = False
    selected_for_stage_3: bool = False
    
    def mark_stage_start(self, stage: int):
        """Mark a stage as started"""
        current_time = datetime.now()
        if stage == 1:
            self.stage_1_status = "running"
            self.stage_1_start_time = current_time
        elif stage == 2:
            self.stage_2_status = "running"
            self.stage_2_start_time = current_time
        elif stage == 3:
            self.stage_3_status = "running"
            self.stage_3_start_time = current_time
    
    def mark_stage_complete(self, stage: int, output_path: Optional[str] = None):
        """Mark a stage as completed"""
        current_time = datetime.now()
        if stage == 1:
            self.stage_1_status = "completed"
            self.stage_1_end_time = current_time
            self.stage_1_output_path = output_path
        elif stage == 2:
            self.stage_2_status = "completed"
            self.stage_2_end_time = current_time
            self.stage_2_output_path = output_path
        elif stage == 3:
            self.stage_3_status = "completed"
            self.stage_3_end_time = current_time
            self.stage_3_output_path = output_path
    
    def mark_stage_failed(self, stage: int, error: str):
        """Mark a stage as failed"""
        current_time = datetime.now()
        if stage == 1:
            self.stage_1_status = "failed"
            self.stage_1_end_time = current_time
            self.stage_1_error = error
        elif stage == 2:
            self.stage_2_status = "failed"
            self.stage_2_end_time = current_time
            self.stage_2_error = error
        elif stage == 3:
            self.stage_3_status = "failed"
            self.stage_3_end_time = current_time
            self.stage_3_error = error
    
    def can_retry(self, stage: int) -> bool:
        """Check if job can be retried for a specific stage"""
        if stage == 1 and self.stage_1_status == "failed":
            return self.retry_count < self.max_retries
        elif stage == 2 and self.stage_2_status == "failed":
            return self.retry_count < self.max_retries
        elif stage == 3 and self.stage_3_status == "failed":
            return self.retry_count < self.max_retries
        return False
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1
    
    def reset_stage_for_retry(self, stage: int):
        """Reset a stage for retry"""
        if stage == 1:
            self.stage_1_status = "pending"
            self.stage_1_start_time = None
            self.stage_1_end_time = None
            self.stage_1_error = None
        elif stage == 2:
            self.stage_2_status = "pending"
            self.stage_2_start_time = None
            self.stage_2_end_time = None
            self.stage_2_error = None
        elif stage == 3:
            self.stage_3_status = "pending"
            self.stage_3_start_time = None
            self.stage_3_end_time = None
            self.stage_3_error = None


@dataclass
class PipelineRun:
    """Represents a complete pipeline run with all jobs"""
    run_id: str
    created_at: datetime
    query: str
    total_links: int
    
    # Jobs in this run
    jobs: Dict[str, Job] = field(default_factory=dict)
    
    # Stage batches
    stage_1_batches: List[BatchInfo] = field(default_factory=list)
    stage_2_batches: List[BatchInfo] = field(default_factory=list)
    stage_3_batches: List[BatchInfo] = field(default_factory=list)
    
    # Run metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "created"  # created, running, completed, failed
    current_stage: int = 1
    
    # Stage timing
    stage_1_start: Optional[datetime] = None
    stage_1_end: Optional[datetime] = None
    stage_2_start: Optional[datetime] = None
    stage_2_end: Optional[datetime] = None
    stage_3_start: Optional[datetime] = None
    stage_3_end: Optional[datetime] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this run"""
        summary = {
            'total': len(self.jobs),
            'stage_1_pending': 0,
            'stage_1_running': 0,
            'stage_1_completed': 0,
            'stage_1_failed': 0,
            'stage_1_skipped': 0,
            'stage_2_pending': 0,
            'stage_2_running': 0,
            'stage_2_completed': 0,
            'stage_2_failed': 0,
            'stage_2_skipped': 0,
            'stage_3_pending': 0,
            'stage_3_running': 0,
            'stage_3_completed': 0,
            'stage_3_failed': 0,
            'stage_3_skipped': 0
        }
        
        for job in self.jobs.values():
            # Stage 1
            summary[f'stage_1_{job.stage_1_status}'] += 1
            # Stage 2
            summary[f'stage_2_{job.stage_2_status}'] += 1
            # Stage 3
            summary[f'stage_3_{job.stage_3_status}'] += 1
        
        return summary
