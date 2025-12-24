"""
Batch Completion Validator

Ensures all stages are complete before proceeding to cross-batch aggregation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from pipeline_models import PipelineRun, Job


@dataclass
class BatchValidationResult:
    """Result of batch validation"""
    batch_id: str
    is_valid: bool
    total_jobs: int
    stage_1_complete: int
    stage_2_complete: int
    stage_3_complete: int
    errors: List[str]
    warnings: List[str]
    validation_time: datetime


@dataclass
class CrossBatchValidationResult:
    """Result of cross-batch validation"""
    total_batches: int
    valid_batches: int
    total_jobs: int
    ready_for_aggregation: bool
    batch_results: List[BatchValidationResult]
    global_errors: List[str]
    validation_time: datetime


class BatchValidator:
    """Validates batch completion and readiness for cross-batch aggregation"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation settings
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        self.batch_results_dir = os.path.join(self.base_output_dir, 'batch_results')
        self.stage_1_dir = os.path.join(self.base_output_dir, 'stage_1_results')
        self.stage_2_dir = os.path.join(self.base_output_dir, 'stage_2_results')
        self.stage_3_dir = os.path.join(self.base_output_dir, 'stage_3_results')
        
        # Validation thresholds
        self.min_stage_1_completion = config.get('validation', {}).get('min_stage_1_completion', 0.95)
        self.min_stage_2_completion = config.get('validation', {}).get('min_stage_2_completion', 0.90)
        self.allow_partial_stage_3 = config.get('validation', {}).get('allow_partial_stage_3', False)
        
        self.logger.info("Batch Validator initialized")
    
    def validate_single_batch(self, batch_id: str) -> BatchValidationResult:
        """
        Validate completion status of a single batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            BatchValidationResult with validation details
        """
        self.logger.info(f"Validating batch {batch_id}")
        
        validation_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Load batch summary
            batch_dir = os.path.join(self.batch_results_dir, f"batch_{batch_id}")
            summary_file = os.path.join(batch_dir, 'batch_summary.json')
            
            if not os.path.exists(summary_file):
                errors.append(f"Batch summary file not found: {summary_file}")
                return BatchValidationResult(
                    batch_id=batch_id,
                    is_valid=False,
                    total_jobs=0,
                    stage_1_complete=0,
                    stage_2_complete=0,
                    stage_3_complete=0,
                    errors=errors,
                    warnings=warnings,
                    validation_time=validation_time
                )
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                batch_summary = json.load(f)
            
            total_jobs = batch_summary.get('total_jobs', 0)
            
            # Validate Stage 1 completion
            stage_1_complete = self._validate_stage_1_completion(batch_id, total_jobs)
            stage_2_complete = self._validate_stage_2_completion(batch_id, stage_1_complete)
            stage_3_complete = self._validate_stage_3_completion(batch_id, stage_2_complete)
            
            # Check completion thresholds
            stage_1_rate = stage_1_complete / total_jobs if total_jobs > 0 else 0
            stage_2_rate = stage_2_complete / total_jobs if total_jobs > 0 else 0
            stage_3_rate = stage_3_complete / total_jobs if total_jobs > 0 else 0
            
            if stage_1_rate < self.min_stage_1_completion:
                errors.append(f"Stage 1 completion rate {stage_1_rate:.2%} below threshold {self.min_stage_1_completion:.2%}")
            
            if stage_2_rate < self.min_stage_2_completion:
                errors.append(f"Stage 2 completion rate {stage_2_rate:.2%} below threshold {self.min_stage_2_completion:.2%}")
            
            if not self.allow_partial_stage_3 and stage_3_rate < 1.0 and stage_2_complete > 0:
                warnings.append(f"Stage 3 not fully completed ({stage_3_rate:.2%}) - may affect aggregation quality")
            
            # Determine overall validity
            is_valid = len(errors) == 0 and stage_1_rate >= self.min_stage_1_completion and stage_2_rate >= self.min_stage_2_completion
            
            self.logger.info(f"Batch {batch_id} validation: {'VALID' if is_valid else 'INVALID'} "
                           f"(S1: {stage_1_complete}/{total_jobs}, S2: {stage_2_complete}/{total_jobs}, S3: {stage_3_complete}/{total_jobs})")
            
            return BatchValidationResult(
                batch_id=batch_id,
                is_valid=is_valid,
                total_jobs=total_jobs,
                stage_1_complete=stage_1_complete,
                stage_2_complete=stage_2_complete,
                stage_3_complete=stage_3_complete,
                errors=errors,
                warnings=warnings,
                validation_time=validation_time
            )
            
        except Exception as e:
            error_msg = f"Validation failed for batch {batch_id}: {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return BatchValidationResult(
                batch_id=batch_id,
                is_valid=False,
                total_jobs=0,
                stage_1_complete=0,
                stage_2_complete=0,
                stage_3_complete=0,
                errors=errors,
                warnings=warnings,
                validation_time=validation_time
            )
    
    def validate_cross_batch_readiness(self, batch_ids: List[str]) -> CrossBatchValidationResult:
        """
        Validate readiness of multiple batches for cross-batch aggregation.
        
        Args:
            batch_ids: List of batch identifiers to validate
            
        Returns:
            CrossBatchValidationResult with overall validation status
        """
        self.logger.info(f"Validating cross-batch readiness for {len(batch_ids)} batches")
        
        validation_time = datetime.now()
        batch_results = []
        global_errors = []
        
        total_jobs = 0
        valid_batches = 0
        
        for batch_id in batch_ids:
            result = self.validate_single_batch(batch_id)
            batch_results.append(result)
            
            total_jobs += result.total_jobs
            if result.is_valid:
                valid_batches += 1
            
            # Collect global errors
            global_errors.extend(result.errors)
        
        # Check overall readiness
        ready_for_aggregation = (
            valid_batches == len(batch_ids) and  # All batches valid
            len(global_errors) == 0 and         # No global errors
            total_jobs > 0                      # Has data to aggregate
        )
        
        self.logger.info(f"Cross-batch validation: {'READY' if ready_for_aggregation else 'NOT READY'} "
                        f"({valid_batches}/{len(batch_ids)} batches valid, {total_jobs} total jobs)")
        
        return CrossBatchValidationResult(
            total_batches=len(batch_ids),
            valid_batches=valid_batches,
            total_jobs=total_jobs,
            ready_for_aggregation=ready_for_aggregation,
            batch_results=batch_results,
            global_errors=global_errors,
            validation_time=validation_time
        )
    
    def _validate_stage_1_completion(self, batch_id: str, expected_jobs: int) -> int:
        """Validate Stage 1 completion for a batch"""
        completed_count = 0
        
        try:
            # Find all Stage 1 result directories for this batch
            batch_pattern = f"batch_{batch_id}"
            
            for job_dir in os.listdir(self.stage_1_dir):
                if job_dir.startswith("job_"):
                    job_dir_path = os.path.join(self.stage_1_dir, job_dir)
                    
                    # Check if this job belongs to the batch
                    classified_file = os.path.join(job_dir_path, 'classified_data.json')
                    if os.path.exists(classified_file):
                        try:
                            with open(classified_file, 'r', encoding='utf-8') as f:
                                job_data = json.load(f)
                            
                            # Simple validation - check if classification exists
                            if job_data.get('classification'):
                                completed_count += 1
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to validate job {job_dir}: {e}")
                            continue
            
            self.logger.debug(f"Stage 1 validation for batch {batch_id}: {completed_count}/{expected_jobs} jobs completed")
            
        except Exception as e:
            self.logger.error(f"Stage 1 validation error for batch {batch_id}: {e}")
        
        return completed_count
    
    def _validate_stage_2_completion(self, batch_id: str, stage_1_jobs: int) -> int:
        """Validate Stage 2 completion for a batch"""
        completed_count = 0
        
        try:
            # Find all Stage 2 result directories
            for job_dir in os.listdir(self.stage_2_dir):
                if job_dir.startswith("job_"):
                    job_dir_path = os.path.join(self.stage_2_dir, job_dir)
                    
                    # Check for DNA analysis output
                    dna_file = os.path.join(job_dir_path, 'dna_analysis.json')
                    profile_file = os.path.join(job_dir_path, 'dna_profile.json')
                    
                    if os.path.exists(dna_file) and os.path.exists(profile_file):
                        try:
                            # Validate DNA analysis structure
                            with open(dna_file, 'r', encoding='utf-8') as f:
                                dna_data = json.load(f)
                            
                            # Check for required fields
                            if (dna_data.get('page_structure') and 
                                dna_data.get('content_characteristics') and
                                dna_data.get('ranking_factors')):
                                completed_count += 1
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to validate Stage 2 for job {job_dir}: {e}")
                            continue
            
            self.logger.debug(f"Stage 2 validation for batch {batch_id}: {completed_count}/{stage_1_jobs} jobs completed")
            
        except Exception as e:
            self.logger.error(f"Stage 2 validation error for batch {batch_id}: {e}")
        
        return completed_count
    
    def _validate_stage_3_completion(self, batch_id: str, stage_2_jobs: int) -> int:
        """Validate Stage 3 completion for a batch"""
        completed_count = 0
        
        try:
            # Look for batch-level Stage 3 results
            batch_dir = os.path.join(self.batch_results_dir, f"batch_{batch_id}")
            aggregation_file = os.path.join(batch_dir, 'final_aggregation.json')
            
            if os.path.exists(aggregation_file):
                try:
                    with open(aggregation_file, 'r', encoding='utf-8') as f:
                        aggregation_data = json.load(f)
                    
                    # Check for required aggregation fields
                    if (aggregation_data.get('aggregated_dna_profile') and
                        aggregation_data.get('competitive_insights') and
                        aggregation_data.get('content_recommendations')):
                        completed_count = stage_2_jobs  # All jobs considered completed
                        
                except Exception as e:
                    self.logger.warning(f"Failed to validate Stage 3 aggregation for batch {batch_id}: {e}")
            else:
                # Check individual run results
                for run_dir in os.listdir(self.stage_3_dir):
                    if run_dir.startswith("run_"):
                        run_dir_path = os.path.join(self.stage_3_dir, run_dir)
                        
                        aggregation_file = os.path.join(run_dir_path, 'final_aggregation.json')
                        if os.path.exists(aggregation_file):
                            completed_count += 1
            
            self.logger.debug(f"Stage 3 validation for batch {batch_id}: {completed_count}/{stage_2_jobs} jobs completed")
            
        except Exception as e:
            self.logger.error(f"Stage 3 validation error for batch {batch_id}: {e}")
        
        return completed_count
    
    def get_batch_completion_summary(self, batch_id: str) -> Dict[str, Any]:
        """
        Get detailed completion summary for a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Dictionary with completion details
        """
        result = self.validate_single_batch(batch_id)
        
        summary = {
            'batch_id': result.batch_id,
            'is_valid': result.is_valid,
            'validation_time': result.validation_time.isoformat(),
            'total_jobs': result.total_jobs,
            'completion_rates': {
                'stage_1': result.stage_1_complete / result.total_jobs if result.total_jobs > 0 else 0,
                'stage_2': result.stage_2_complete / result.total_jobs if result.total_jobs > 0 else 0,
                'stage_3': result.stage_3_complete / result.total_jobs if result.total_jobs > 0 else 0
            },
            'completed_jobs': {
                'stage_1': result.stage_1_complete,
                'stage_2': result.stage_2_complete,
                'stage_3': result.stage_3_complete
            },
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        return summary
    
    def find_incomplete_batches(self) -> List[str]:
        """
        Find all batches that are not ready for aggregation.
        
        Returns:
            List of batch IDs that need attention
        """
        incomplete_batches = []
        
        if not os.path.exists(self.batch_results_dir):
            self.logger.warning("Batch results directory not found")
            return incomplete_batches
        
        # Find all batch directories
        for item in os.listdir(self.batch_results_dir):
            if item.startswith("batch_"):
                batch_id = item.replace("batch_", "")
                
                result = self.validate_single_batch(batch_id)
                if not result.is_valid:
                    incomplete_batches.append(batch_id)
        
        self.logger.info(f"Found {len(incomplete_batches)} incomplete batches")
        return incomplete_batches
    
    def save_validation_report(self, validation_result: CrossBatchValidationResult, 
                             output_file: Optional[str] = None):
        """
        Save validation report to file.
        
        Args:
            validation_result: Cross-batch validation result
            output_file: Optional output file path
        """
        if output_file is None:
            timestamp = validation_result.validation_time.strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.base_output_dir, f'validation_report_{timestamp}.json')
        
        report_data = {
            'validation_time': validation_result.validation_time.isoformat(),
            'summary': {
                'total_batches': validation_result.total_batches,
                'valid_batches': validation_result.valid_batches,
                'total_jobs': validation_result.total_jobs,
                'ready_for_aggregation': validation_result.ready_for_aggregation
            },
            'batch_results': [],
            'global_errors': validation_result.global_errors
        }
        
        for batch_result in validation_result.batch_results:
            batch_data = {
                'batch_id': batch_result.batch_id,
                'is_valid': batch_result.is_valid,
                'total_jobs': batch_result.total_jobs,
                'completed_jobs': {
                    'stage_1': batch_result.stage_1_complete,
                    'stage_2': batch_result.stage_2_complete,
                    'stage_3': batch_result.stage_3_complete
                },
                'errors': batch_result.errors,
                'warnings': batch_result.warnings
            }
            report_data['batch_results'].append(batch_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = BatchValidator(config)
    
    print("=== Batch Validator Test ===")
    
    # Find all batches
    if os.path.exists(validator.batch_results_dir):
        batch_ids = []
        for item in os.listdir(validator.batch_results_dir):
            if item.startswith("batch_"):
                batch_ids.append(item.replace("batch_", ""))
        
        if batch_ids:
            print(f"Found {len(batch_ids)} batches: {batch_ids}")
            
            # Validate cross-batch readiness
            result = validator.validate_cross_batch_readiness(batch_ids)
            
            print(f"Cross-batch validation result:")
            print(f"  Ready for aggregation: {result.ready_for_aggregation}")
            print(f"  Valid batches: {result.valid_batches}/{result.total_batches}")
            print(f"  Total jobs: {result.total_jobs}")
            
            if result.global_errors:
                print("Global errors:")
                for error in result.global_errors:
                    print(f"  - {error}")
            
            # Save validation report
            validator.save_validation_report(result)
            
        else:
            print("No batches found")
    else:
        print("Batch results directory not found")
