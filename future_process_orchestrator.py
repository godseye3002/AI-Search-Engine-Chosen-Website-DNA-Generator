"""
Future Process Orchestrator

Coordinates the complete Phase 5 workflow including validation,
dependency management, and cross-batch aggregation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
import asyncio

from batch_processor import BatchProcessor, BatchResult
from batch_validator import BatchValidator, CrossBatchValidationResult
from cross_batch_aggregator import CrossBatchAggregator, CrossBatchResult
from dependency_manager import DependencyManager, StageStatus


class ProcessState(Enum):
    """Process execution state"""
    IDLE = "idle"
    VALIDATING = "validating"
    PROCESSING = "processing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessMetrics:
    """Process execution metrics"""
    total_batches: int
    total_jobs: int
    processed_batches: int
    processed_jobs: int
    failed_batches: int
    failed_jobs: int
    start_time: datetime
    end_time: Optional[datetime]
    processing_time: float


class FutureProcessOrchestrator:
    """Orchestrates the complete Phase 5 future process"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.batch_processor = BatchProcessor(config, logger)
        self.batch_validator = BatchValidator(config, logger)
        self.cross_batch_aggregator = CrossBatchAggregator(config, logger)
        self.dependency_manager = DependencyManager(config, logger)
        
        # Process state
        self.current_state = ProcessState.IDLE
        self.process_metrics: Optional[ProcessMetrics] = None
        self.active_process_id: Optional[str] = None
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        self.logger.info("Future Process Orchestrator initialized")
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates"""
        self.status_callback = callback
    
    def execute_future_process(self, batch_configs: List[Dict[str, Any]], 
                             validate_before: bool = True,
                             wait_for_dependencies: bool = True) -> CrossBatchResult:
        """
        Execute the complete future process workflow.
        
        Args:
            batch_configs: List of batch configurations
            validate_before: Validate batches before processing
            wait_for_dependencies: Wait for dependencies to be satisfied
            
        Returns:
            CrossBatchResult with final aggregation
        """
        with self._lock:
            if self.current_state != ProcessState.IDLE:
                raise RuntimeError(f"Process already running in state: {self.current_state.value}")
            
            self.current_state = ProcessState.PROCESSING
            self.active_process_id = f"future_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._stop_event.clear()
        
        try:
            self.logger.info(f"Starting future process: {self.active_process_id}")
            
            # Initialize metrics
            total_jobs = sum(len(config.get('queries', [])) for config in batch_configs)
            self.process_metrics = ProcessMetrics(
                total_batches=len(batch_configs),
                total_jobs=total_jobs,
                processed_batches=0,
                processed_jobs=0,
                failed_batches=0,
                failed_jobs=0,
                start_time=datetime.now(),
                end_time=None,
                processing_time=0.0
            )
            
            # Step 1: Create and validate batches
            batch_ids = self._create_and_validate_batches(batch_configs, validate_before)
            
            # Step 2: Wait for dependencies if required
            if wait_for_dependencies:
                self._wait_for_all_dependencies(batch_ids)
            
            # Step 3: Process batches
            processed_batches = self._process_all_batches(batch_ids)
            
            # Step 4: Cross-batch aggregation
            aggregation_result = self._perform_cross_batch_aggregation(processed_batches)
            
            # Update final metrics
            self.process_metrics.end_time = datetime.now()
            self.process_metrics.processing_time = (
                self.process_metrics.end_time - self.process_metrics.start_time
            ).total_seconds()
            
            self.current_state = ProcessState.COMPLETED
            self.logger.info(f"Future process completed: {self.active_process_id}")
            
            return aggregation_result
            
        except Exception as e:
            self.current_state = ProcessState.FAILED
            self.logger.error(f"Future process failed: {e}")
            raise
        finally:
            self.active_process_id = None
    
    def _create_and_validate_batches(self, batch_configs: List[Dict[str, Any]], 
                                   validate_before: bool) -> List[str]:
        """Create and validate batches"""
        self._update_status(ProcessState.VALIDATING)
        
        batch_ids = []
        
        for i, config in enumerate(batch_configs):
            self._update_progress(f"Creating batch {i+1}/{len(batch_configs)}", 
                                (i / len(batch_configs)) * 100)
            
            # Create batch
            batch_id = config.get('batch_id', f"batch_{i+1:03d}")
            queries = config.get('queries', [])
            ai_files = config.get('ai_response_files', ['ai_response.json'] * len(queries))
            
            jobs = self.batch_processor.create_batch(batch_id, queries, ai_files)
            batch_ids.append(batch_id)
            
            self.logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")
        
        # Validate batches if requested
        if validate_before:
            self._update_progress("Validating batches", 50)
            
            validation_result = self.batch_validator.validate_cross_batch_readiness(batch_ids)
            
            if not validation_result.ready_for_aggregation:
                self.logger.warning("Batches not fully ready for aggregation, but proceeding with processing")
            
            self.logger.info(f"Validation completed: {validation_result.valid_batches}/{validation_result.total_batches} batches valid")
        
        return batch_ids
    
    def _wait_for_all_dependencies(self, batch_ids: List[str]):
        """Wait for all stage dependencies to be satisfied"""
        self._update_progress("Waiting for dependencies", 60)
        
        # Check Stage 1 dependencies (should always be ready)
        if not self.dependency_manager.can_start_stage(1, batch_ids):
            raise RuntimeError("Stage 1 dependencies not satisfied")
        
        # Wait for Stage 2 dependencies
        if not self.dependency_manager.wait_for_dependencies(2, batch_ids, timeout=3600):
            raise RuntimeError("Timeout waiting for Stage 2 dependencies")
        
        # Wait for Stage 3 dependencies
        if not self.dependency_manager.wait_for_dependencies(3, batch_ids, timeout=3600):
            raise RuntimeError("Timeout waiting for Stage 3 dependencies")
        
        self.logger.info("All dependencies satisfied")
    
    def _process_all_batches(self, batch_ids: List[str]) -> List[str]:
        """Process all batches"""
        self._update_status(ProcessState.PROCESSING)
        
        processed_batches = []
        
        for i, batch_id in enumerate(batch_ids):
            if self._stop_event.is_set():
                raise RuntimeError("Process stopped by user")
            
            progress = 60 + (i / len(batch_ids)) * 30
            self._update_progress(f"Processing batch {batch_id}", progress)
            
            try:
                # Process batch
                def batch_progress_callback(bid, job_id, status, prog):
                    self._update_progress(f"Batch {bid}: {job_id} - {status}", progress + (prog / len(batch_ids)) * 10)
                
                result = self.batch_processor.process_batch_parallel(batch_id, batch_progress_callback)
                
                if result.failed_jobs > 0:
                    self.process_metrics.failed_batches += 1
                    self.process_metrics.failed_jobs += result.failed_jobs
                    self.logger.warning(f"Batch {batch_id} completed with {result.failed_jobs} failed jobs")
                else:
                    self.process_metrics.processed_batches += 1
                
                self.process_metrics.processed_jobs += result.completed_jobs
                processed_batches.append(batch_id)
                
                self.logger.info(f"Batch {batch_id} processed: {result.completed_jobs} completed, {result.failed_jobs} failed")
                
            except Exception as e:
                self.process_metrics.failed_batches += 1
                self.logger.error(f"Failed to process batch {batch_id}: {e}")
                # Continue with other batches
        
        if not processed_batches:
            raise RuntimeError("No batches processed successfully")
        
        return processed_batches
    
    def _perform_cross_batch_aggregation(self, batch_ids: List[str]) -> CrossBatchResult:
        """Perform cross-batch aggregation"""
        self._update_status(ProcessState.AGGREGATING)
        self._update_progress("Performing cross-batch aggregation", 90)
        
        try:
            result = self.cross_batch_aggregator.aggregate_cross_batch(batch_ids)
            
            self._update_progress("Cross-batch aggregation completed", 95)
            
            self.logger.info(f"Cross-batch aggregation completed: {result.aggregation_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cross-batch aggregation failed: {e}")
            raise
    
    def get_process_status(self) -> Dict[str, Any]:
        """Get current process status"""
        with self._lock:
            status = {
                'state': self.current_state.value,
                'process_id': self.active_process_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.process_metrics:
                status['metrics'] = {
                    'total_batches': self.process_metrics.total_batches,
                    'total_jobs': self.process_metrics.total_jobs,
                    'processed_batches': self.process_metrics.processed_batches,
                    'processed_jobs': self.process_metrics.processed_jobs,
                    'failed_batches': self.process_metrics.failed_batches,
                    'failed_jobs': self.process_metrics.failed_jobs,
                    'start_time': self.process_metrics.start_time.isoformat(),
                    'processing_time': self.process_metrics.processing_time
                }
                
                if self.process_metrics.end_time:
                    status['metrics']['end_time'] = self.process_metrics.end_time.isoformat()
            
            return status
    
    def stop_process(self):
        """Stop the current process"""
        with self._lock:
            if self.current_state in [ProcessState.PROCESSING, ProcessState.AGGREGATING]:
                self._stop_event.set()
                self.current_state = ProcessState.FAILED
                self.logger.info("Process stopped by user")
    
    def _update_status(self, new_state: ProcessState):
        """Update process status"""
        with self._lock:
            self.current_state = new_state
        
        if self.status_callback:
            self.status_callback(new_state.value)
    
    def _update_progress(self, message: str, progress: float):
        """Update progress"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def validate_future_process_readiness(self) -> Dict[str, Any]:
        """Validate readiness for future process execution"""
        readiness_report = {
            'ready': True,
            'checks': {},
            'issues': []
        }
        
        # Check batch processor
        try:
            # Simple check - just ensure it's initialized
            readiness_report['checks']['batch_processor'] = 'ready'
        except Exception as e:
            readiness_report['checks']['batch_processor'] = 'error'
            readiness_report['issues'].append(f"Batch processor error: {e}")
            readiness_report['ready'] = False
        
        # Check batch validator
        try:
            incomplete = self.batch_validator.find_incomplete_batches()
            if incomplete:
                readiness_report['checks']['batch_validator'] = 'warning'
                readiness_report['issues'].append(f"Incomplete batches found: {incomplete}")
            else:
                readiness_report['checks']['batch_validator'] = 'ready'
        except Exception as e:
            readiness_report['checks']['batch_validator'] = 'error'
            readiness_report['issues'].append(f"Batch validator error: {e}")
            readiness_report['ready'] = False
        
        # Check dependency manager
        try:
            # Check Stage 1 (should always be ready)
            stage1_ready = self.dependency_manager.can_start_stage(1)
            readiness_report['checks']['dependency_manager'] = 'ready' if stage1_ready else 'warning'
            
            if not stage1_ready:
                readiness_report['issues'].append("Stage 1 dependencies not satisfied")
        except Exception as e:
            readiness_report['checks']['dependency_manager'] = 'error'
            readiness_report['issues'].append(f"Dependency manager error: {e}")
            readiness_report['ready'] = False
        
        # Check output directories
        base_dir = self.config.get('pipeline', {}).get('base_output_dir', 'outputs')
        required_dirs = ['batch_results', 'cross_batch_results']
        
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    readiness_report['checks'][f'directory_{dir_name}'] = 'created'
                except Exception as e:
                    readiness_report['checks'][f'directory_{dir_name}'] = 'error'
                    readiness_report['issues'].append(f"Cannot create {dir_name} directory: {e}")
                    readiness_report['ready'] = False
            else:
                readiness_report['checks'][f'directory_{dir_name}'] = 'ready'
        
        return readiness_report
    
    def get_process_summary(self) -> Dict[str, Any]:
        """Get summary of completed process"""
        if not self.process_metrics:
            return {'error': 'No process metrics available'}
        
        summary = {
            'process_id': self.active_process_id,
            'final_state': self.current_state.value,
            'metrics': {
                'total_batches': self.process_metrics.total_batches,
                'total_jobs': self.process_metrics.total_jobs,
                'processed_batches': self.process_metrics.processed_batches,
                'processed_jobs': self.process_metrics.processed_jobs,
                'failed_batches': self.process_metrics.failed_batches,
                'failed_jobs': self.process_metrics.failed_jobs,
                'success_rate': (self.process_metrics.processed_jobs / self.process_metrics.total_jobs 
                               if self.process_metrics.total_jobs > 0 else 0),
                'processing_time': self.process_metrics.processing_time
            },
            'start_time': self.process_metrics.start_time.isoformat()
        }
        
        if self.process_metrics.end_time:
            summary['end_time'] = self.process_metrics.end_time.isoformat()
        
        return summary


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = FutureProcessOrchestrator(config)
    
    print("=== Future Process Orchestrator Test ===")
    
    # Test readiness
    readiness = orchestrator.validate_future_process_readiness()
    print(f"Process readiness: {readiness['ready']}")
    
    if readiness['issues']:
        print("Issues found:")
        for issue in readiness['issues']:
            print(f"  - {issue}")
    
    # Example batch configurations
    batch_configs = [
        {
            'batch_id': 'test_batch_1',
            'queries': ['best SEO tools', 'content marketing strategies'],
            'ai_response_files': ['ai_response.json', 'ai_response.json']
        },
        {
            'batch_id': 'test_batch_2', 
            'queries': ['website optimization', 'ranking factors'],
            'ai_response_files': ['ai_response.json', 'ai_response.json']
        }
    ]
    
    # Set up callbacks
    def progress_callback(message, progress):
        print(f"Progress: {progress:.1f}% - {message}")
    
    def status_callback(status):
        print(f"Status: {status}")
    
    orchestrator.set_progress_callback(progress_callback)
    orchestrator.set_status_callback(status_callback)
    
    print("\nStarting future process...")
    
    try:
        result = orchestrator.execute_future_process(batch_configs)
        
        print(f"\nFuture process completed successfully!")
        print(f"Aggregation ID: {result.aggregation_id}")
        print(f"Total Batches: {result.total_batches}")
        print(f"Total Jobs: {result.total_jobs}")
        print(f"Recommendations: {len(result.unified_recommendations)}")
        
        # Get summary
        summary = orchestrator.get_process_summary()
        print(f"\nProcess Summary:")
        print(f"  Success Rate: {summary['metrics']['success_rate']:.2%}")
        print(f"  Processing Time: {summary['metrics']['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"Future process failed: {e}")
    
    # Get final status
    final_status = orchestrator.get_process_status()
    print(f"\nFinal Status: {final_status['state']}")
