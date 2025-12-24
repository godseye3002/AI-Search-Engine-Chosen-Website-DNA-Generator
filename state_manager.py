"""
State Manager for GodsEye Pipeline

Handles persistent storage and retrieval of pipeline state,
job tracking, and run management.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from pipeline_models import PipelineRun, Job


class StateManager:
    """Manages pipeline state persistence and retrieval"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Output directories
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        self.state_dir = os.path.join(self.base_output_dir, 'states')
        
        # Create directories
        os.makedirs(self.state_dir, exist_ok=True)
        
        self.logger.info("State Manager initialized")
    
    def save_run_state(self, run: PipelineRun):
        """Save complete run state to file"""
        state_file = os.path.join(self.state_dir, f'run_{run.run_id}.json')
        
        # Convert run to serializable format
        state_data = self._serialize_run(run)
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.debug(f"Saved run state for {run.run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save run state for {run.run_id}: {e}")
            raise
    
    def load_run_state(self, run_id: str) -> Optional[PipelineRun]:
        """Load run state from file"""
        state_file = os.path.join(self.state_dir, f'run_{run_id}.json')
        
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            run = self._deserialize_run(state_data)
            self.logger.debug(f"Loaded run state for {run_id}")
            return run
            
        except Exception as e:
            self.logger.error(f"Failed to load run state for {run_id}: {e}")
            return None
    
    def _serialize_run(self, run: PipelineRun) -> Dict[str, Any]:
        """Convert PipelineRun to serializable dict"""
        return {
            'run_id': run.run_id,
            'created_at': run.created_at.isoformat(),
            'query': run.query,
            'total_links': run.total_links,
            'status': run.status,
            'current_stage': run.current_stage,
            'jobs': {
                job_id: self._serialize_job(job) 
                for job_id, job in run.jobs.items()
            },
            'stage_1_batches': [self._serialize_batch(batch) for batch in run.stage_1_batches],
            'stage_2_batches': [self._serialize_batch(batch) for batch in run.stage_2_batches],
            'stage_3_batches': [self._serialize_batch(batch) for batch in run.stage_3_batches],
            'stage_1_start': run.stage_1_start.isoformat() if run.stage_1_start else None,
            'stage_1_end': run.stage_1_end.isoformat() if run.stage_1_end else None,
            'stage_2_start': run.stage_2_start.isoformat() if run.stage_2_start else None,
            'stage_2_end': run.stage_2_end.isoformat() if run.stage_2_end else None,
            'stage_3_start': run.stage_3_start.isoformat() if run.stage_3_start else None,
            'stage_3_end': run.stage_3_end.isoformat() if run.stage_3_end else None
        }
    
    def _deserialize_run(self, state_data: Dict[str, Any]) -> PipelineRun:
        """Convert dict back to PipelineRun"""
        run = PipelineRun(
            run_id=state_data['run_id'],
            created_at=datetime.fromisoformat(state_data['created_at']),
            query=state_data['query'],
            total_links=state_data['total_links']
        )
        
        run.status = state_data.get('status', 'initialized')
        run.current_stage = state_data.get('current_stage', 1)
        
        # Deserialize jobs
        run.jobs = {
            job_id: self._deserialize_job(job_data)
            for job_id, job_data in state_data['jobs'].items()
        }
        
        # Deserialize batches (will be handled by batch_calculator)
        # For now, just store the raw data
        run.stage_1_batches_raw = state_data.get('stage_1_batches', [])
        run.stage_2_batches_raw = state_data.get('stage_2_batches', [])
        run.stage_3_batches_raw = state_data.get('stage_3_batches', [])
        
        # Deserialize timestamps
        if state_data.get('stage_1_start'):
            run.stage_1_start = datetime.fromisoformat(state_data['stage_1_start'])
        if state_data.get('stage_1_end'):
            run.stage_1_end = datetime.fromisoformat(state_data['stage_1_end'])
        if state_data.get('stage_2_start'):
            run.stage_2_start = datetime.fromisoformat(state_data['stage_2_start'])
        if state_data.get('stage_2_end'):
            run.stage_2_end = datetime.fromisoformat(state_data['stage_2_end'])
        if state_data.get('stage_3_start'):
            run.stage_3_start = datetime.fromisoformat(state_data['stage_3_start'])
        if state_data.get('stage_3_end'):
            run.stage_3_end = datetime.fromisoformat(state_data['stage_3_end'])
        
        return run
    
    def _serialize_job(self, job: Job) -> Dict[str, Any]:
        """Convert Job to serializable dict"""
        return {
            'job_id': job.job_id,
            'run_id': job.run_id,
            'source_link': job.source_link,
            'url': job.url,
            'text': job.text,
            'position': job.position,
            'stage_1_status': job.stage_1_status,
            'stage_1_output_path': job.stage_1_output_path,
            'stage_1_error': job.stage_1_error,
            'stage_1_start_time': job.stage_1_start_time.isoformat() if job.stage_1_start_time else None,
            'stage_1_end_time': job.stage_1_end_time.isoformat() if job.stage_1_end_time else None,
            'stage_2_status': job.stage_2_status,
            'stage_2_output_path': job.stage_2_output_path,
            'stage_2_error': job.stage_2_error,
            'stage_2_start_time': job.stage_2_start_time.isoformat() if job.stage_2_start_time else None,
            'stage_2_end_time': job.stage_2_end_time.isoformat() if job.stage_2_end_time else None,
            'stage_3_status': job.stage_3_status,
            'stage_3_output_path': job.stage_3_output_path,
            'stage_3_error': job.stage_3_error,
            'stage_3_start_time': job.stage_3_start_time.isoformat() if job.stage_3_start_time else None,
            'stage_3_end_time': job.stage_3_end_time.isoformat() if job.stage_3_end_time else None,
            'retry_count': job.retry_count,
            'max_retries': job.max_retries,
            'classification': job.classification,
            'selected_for_stage_2': job.selected_for_stage_2
        }
    
    def _deserialize_job(self, job_data: Dict[str, Any]) -> Job:
        """Convert dict back to Job"""
        job = Job(
            job_id=job_data['job_id'],
            run_id=job_data['run_id'],
            source_link=job_data['source_link'],
            url=job_data['url'],
            text=job_data['text'],
            position=job_data['position'],
            max_retries=job_data.get('max_retries', 2)
        )
        
        job.stage_1_status = job_data.get('stage_1_status', 'pending')
        job.stage_1_output_path = job_data.get('stage_1_output_path')
        job.stage_1_error = job_data.get('stage_1_error')
        job.stage_2_status = job_data.get('stage_2_status', 'pending')
        job.stage_2_output_path = job_data.get('stage_2_output_path')
        job.stage_2_error = job_data.get('stage_2_error')
        job.stage_3_status = job_data.get('stage_3_status', 'pending')
        job.stage_3_output_path = job_data.get('stage_3_output_path')
        job.stage_3_error = job_data.get('stage_3_error')
        job.retry_count = job_data.get('retry_count', 0)
        job.classification = job_data.get('classification')
        job.selected_for_stage_2 = job_data.get('selected_for_stage_2', False)
        
        # Deserialize timestamps
        if job_data.get('stage_1_start_time'):
            job.stage_1_start_time = datetime.fromisoformat(job_data['stage_1_start_time'])
        if job_data.get('stage_1_end_time'):
            job.stage_1_end_time = datetime.fromisoformat(job_data['stage_1_end_time'])
        if job_data.get('stage_2_start_time'):
            job.stage_2_start_time = datetime.fromisoformat(job_data['stage_2_start_time'])
        if job_data.get('stage_2_end_time'):
            job.stage_2_end_time = datetime.fromisoformat(job_data['stage_2_end_time'])
        if job_data.get('stage_3_start_time'):
            job.stage_3_start_time = datetime.fromisoformat(job_data['stage_3_start_time'])
        if job_data.get('stage_3_end_time'):
            job.stage_3_end_time = datetime.fromisoformat(job_data['stage_3_end_time'])
        
        return job
    
    def _serialize_batch(self, batch) -> Dict[str, Any]:
        """Convert BatchInfo to serializable dict"""
        return {
            'batch_id': batch.batch_id,
            'start_idx': batch.start_idx,
            'end_idx': batch.end_idx,
            'size': batch.size,
            'items': batch.items if hasattr(batch, 'items') else []
        }
    
    def list_runs(self) -> List[str]:
        """List all available run IDs"""
        state_files = Path(self.state_dir).glob('run_*.json')
        run_ids = []
        
        for file in state_files:
            run_id = file.stem.replace('run_', '')
            run_ids.append(run_id)
        
        return sorted(run_ids, reverse=True)  # Most recent first
    
    def delete_run_state(self, run_id: str) -> bool:
        """Delete run state file"""
        state_file = os.path.join(self.state_dir, f'run_{run_id}.json')
        
        try:
            if os.path.exists(state_file):
                os.remove(state_file)
                self.logger.info(f"Deleted run state for {run_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete run state for {run_id}: {e}")
            return False
    
    def cleanup_old_states(self, keep_days: int = 7):
        """Clean up old run state files"""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        state_files = Path(self.state_dir).glob('run_*.json')
        deleted_count = 0
        
        for file in state_files:
            if file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete old state file {file}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old run state files")


if __name__ == "__main__":
    # Test state manager
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test state manager
    manager = StateManager(config)
    
    print("=== State Manager Test ===")
    
    # List existing runs
    runs = manager.list_runs()
    print(f"Existing runs: {runs[:5]}...")  # Show first 5
    
    # Test cleanup
    manager.cleanup_old_states(keep_days=30)
    print("Cleanup completed")
