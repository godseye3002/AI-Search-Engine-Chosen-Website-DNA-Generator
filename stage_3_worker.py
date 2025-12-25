"""
Stage 3 Worker - Final Aggregation

Handles final aggregation of DNA analysis results.
Creates comprehensive reports and actionable insights.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from final_aggregation_core import aggregate_pipeline_results
from utils.timeout_handler import execute_with_timeout, TimeoutResult
from pipeline_models import Job


class Stage3Worker:
    """Worker for Stage 3: Final Aggregation"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stage 3 specific settings
        self.timeout_per_job = config.get('stage_3_aggregation', {}).get('timeout_per_job', 300)
        self.output_dir = config.get('stage_3_aggregation', {}).get('output_dir', 'stage_3_results')
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        
        # Create output directory
        self.stage_output_dir = os.path.join(self.base_output_dir, self.output_dir)
        os.makedirs(self.stage_output_dir, exist_ok=True)
        
        self.logger.info("Stage 3 Worker initialized with simplified aggregation")
    
    def process_run(self, run_id: str, query: str, jobs: list) -> Dict[str, Any]:
        """
        Process a complete pipeline run through Stage 3 final aggregation.
        
        Args:
            run_id: Pipeline run identifier
            query: Original search query
            jobs: List of Job objects with completed Stage 2
            
        Returns:
            Dictionary with aggregation results and status
        """
        self.logger.info(f"Processing run {run_id} for Stage 3 final aggregation")
        
        try:
            # Load DNA analysis results from Stage 2
            dna_results = self._load_stage_2_results(jobs)
            
            if not dna_results:
                raise ValueError(f"No DNA analysis results found for run {run_id}")
            
            self.logger.info(f"Loaded {len(dna_results)} DNA analysis results")
            
            # Execute Master Blueprint aggregation with timeout
            result = execute_with_timeout(
                aggregate_pipeline_results,
                args=(run_id, query, dna_results),
                timeout=self.timeout_per_job
            )
            
            # Process result
            if result.status == TimeoutResult.SUCCESS:
                master_blueprint = result.result
                
                # Save aggregation outputs
                output_paths = self._save_run_outputs(run_id, master_blueprint)
                
                # Update job statuses
                for job in jobs:
                    if job.selected_for_stage_3:
                        job.mark_stage_complete(3, output_paths['aggregation_path'])
                
                self.logger.info(f"Run {run_id} completed Stage 3: Final aggregation complete")
                
                return {
                    'status': 'completed',
                    'run_id': run_id,
                    'total_analyzed': len(dna_results),
                    'output_paths': output_paths,
                    'processing_time': None,
                    'error': None
                }
                
            elif result.status == TimeoutResult.TIMEOUT:
                error_msg = f"Stage 3 timeout after {self.timeout_per_job}s"
                self.logger.warning(f"Run {run_id} failed Stage 3: {error_msg}")
                
                # Mark jobs as failed
                for job in jobs:
                    if job.selected_for_stage_3:
                        job.mark_stage_failed(3, error_msg)
                
                return {
                    'status': 'failed',
                    'run_id': run_id,
                    'error': error_msg,
                    'total_analyzed': 0
                }
                
            else:  # ERROR
                error_msg = f"Stage 3 error: {result.error_message}"
                self.logger.error(f"Run {run_id} failed Stage 3: {error_msg}")
                
                # Mark jobs as failed
                for job in jobs:
                    if job.selected_for_stage_3:
                        job.mark_stage_failed(3, error_msg)
                
                return {
                    'status': 'failed',
                    'run_id': run_id,
                    'error': error_msg,
                    'total_analyzed': 0
                }
        
        except Exception as e:
            error_msg = f"Unexpected error in Stage 3: {str(e)}"
            self.logger.error(f"Run {run_id} failed Stage 3: {error_msg}")
            
            # Mark jobs as failed
            for job in jobs:
                if job.selected_for_stage_3:
                    job.mark_stage_failed(3, error_msg)
            
            return {
                'status': 'failed',
                'run_id': run_id,
                'error': error_msg,
                'total_analyzed': 0
            }
    
    def _load_stage_2_results(self, jobs: list) -> list:
        """
        Load DNA analysis results from Stage 2 for selected jobs.
        
        Args:
            jobs: List of Job objects selected for Stage 3
            
        Returns:
            List of DNA analysis result dictionaries
        """
        dna_results = []
        
        for job in jobs:
            if not job.selected_for_stage_3 or job.stage_2_status != 'completed':
                continue
            
            try:
                # Load DNA analysis result
                dna_file = os.path.join(
                    self.base_output_dir, 
                    'stage_2_results', 
                    f'job_{job.job_id}', 
                    'dna_analysis.json'
                )
                
                with open(dna_file, 'r', encoding='utf-8') as f:
                    dna_data = json.load(f)
                
                # Add job metadata
                dna_data['job_id'] = job.job_id
                dna_data['url'] = job.url
                dna_data['classification'] = job.classification
                
                dna_results.append(dna_data)
                
            except FileNotFoundError:
                self.logger.warning(f"DNA analysis file not found for job {job.job_id}")
                continue
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in DNA analysis for job {job.job_id}: {e}")
                continue
        
        return dna_results
    
    def _save_run_outputs(self, run_id: str, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Save final aggregation results to files.
        
        Args:
            run_id: Pipeline run identifier
            result: Aggregation result
            
        Returns:
            Dictionary with file paths
        """
        # Create run-specific directory
        run_dir = os.path.join(self.stage_output_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        file_paths = {}
        
        # Save complete aggregation result
        # Save master blueprint as the main output (simplified approach)
        aggregation_path = os.path.join(run_dir, 'final_aggregation.json')
        
        # The result is now just the master blueprint dictionary
        with open(aggregation_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        file_paths['aggregation_path'] = aggregation_path
        
        self.logger.info(f"Saved master blueprint to {aggregation_path}")
        return file_paths
    
    def get_run_summary(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics for a processed run.
        
        Args:
            run_result: Result from process_run
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'run_id': run_result.get('run_id'),
            'status': run_result.get('status'),
            'total_analyzed': run_result.get('total_analyzed', 0),
            'processing_time': run_result.get('processing_time', 0),
            'error': run_result.get('error'),
            'has_recommendations': bool(run_result.get('output_paths', {}).get('recommendations_path')),
            'has_opportunities': bool(run_result.get('output_paths', {}).get('opportunities_path')),
            'has_report': bool(run_result.get('output_paths', {}).get('report_path'))
        }


if __name__ == "__main__":
    # Test Stage 3 worker
    import yaml
    from datetime import datetime
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test worker
    worker = Stage3Worker(config)
    
    print("=== Stage 3 Worker Test ===")
    
    # Create test jobs
    from pipeline_models import Job
    
    test_jobs = []
    for i in range(3):
        job = Job(
            job_id=f"test_job_{i+1}",
            run_id="test_run",
            source_link={
                "url": f"https://example.com/{i+1}",
                "text": f"Example website {i+1}",
                "related_to": "test search query"
            },
            url=f"https://example.com/{i+1}",
            text=f"Example website {i+1}",
            position=i+1
        )
        
        # Simulate Stage 2 completion
        job.mark_stage_complete(2, f"outputs/stage_2_results/job_{job.job_id}/dna_analysis.json")
        job.selected_for_stage_3 = True
        
        test_jobs.append(job)
    
    # Test processing
    result = worker.process_run("test_run", "test query", test_jobs)
    
    print(f"Run status: {result['status']}")
    print(f"Total analyzed: {result['total_analyzed']}")
    if result['error']:
        print(f"Error: {result['error']}")
    if result.get('output_paths'):
        print(f"Outputs saved to: {result['output_paths']}")
    
    # Test summary
    summary = worker.get_run_summary(result)
    print(f"Run summary: {summary}")
