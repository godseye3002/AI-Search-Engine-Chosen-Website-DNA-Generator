"""
Dependency Manager

Manages dependencies between pipeline stages and ensures proper execution order.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import threading
import time

from batch_validator import BatchValidator, CrossBatchValidationResult


class StageStatus(Enum):
    """Stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class StageDependency:
    """Dependency between stages"""
    stage: int
    depends_on: List[int]
    completion_threshold: float = 1.0  # 0.0 to 1.0


@dataclass
class DependencyCheck:
    """Result of dependency check"""
    stage: int
    dependencies_met: bool
    completion_rates: Dict[int, float]
    blocking_issues: List[str]
    ready_to_run: bool


class DependencyManager:
    """Manages pipeline stage dependencies"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Dependency configuration
        self.stage_dependencies = self._load_stage_dependencies()
        self.batch_validator = BatchValidator(config, logger)
        
        # Runtime state
        self.stage_status: Dict[int, StageStatus] = {
            1: StageStatus.PENDING,
            2: StageStatus.PENDING,
            3: StageStatus.PENDING
        }
        
        self.batch_status: Dict[str, Dict[int, StageStatus]] = {}
        self.dependency_cache: Dict[str, DependencyCheck] = {}
        self.cache_ttl = config.get('dependency_management', {}).get('cache_ttl', 300)  # 5 minutes
        
        self.logger.info("Dependency Manager initialized")
    
    def _load_stage_dependencies(self) -> List[StageDependency]:
        """Load stage dependency configuration"""
        dependencies = [
            StageDependency(
                stage=1,
                depends_on=[],  # Stage 1 has no dependencies
                completion_threshold=1.0
            ),
            StageDependency(
                stage=2,
                depends_on=[1],  # Stage 2 depends on Stage 1
                completion_threshold=self.config.get('dependency_management', {}).get('stage_2_threshold', 0.95)
            ),
            StageDependency(
                stage=3,
                depends_on=[1, 2],  # Stage 3 depends on Stage 1 and 2
                completion_threshold=self.config.get('dependency_management', {}).get('stage_3_threshold', 1.0)
            )
        ]
        
        return dependencies
    
    def check_stage_dependencies(self, stage: int, batch_ids: Optional[List[str]] = None) -> DependencyCheck:
        """
        Check if dependencies for a stage are met.
        
        Args:
            stage: Stage number to check
            batch_ids: Optional list of batch IDs to check (uses all if None)
            
        Returns:
            DependencyCheck with dependency status
        """
        cache_key = f"stage_{stage}_{hash(tuple(batch_ids or []))}"
        
        # Check cache first
        if cache_key in self.dependency_cache:
            cached_check = self.dependency_cache[cache_key]
            cache_age = (datetime.now() - cached_check.timestamp).total_seconds()
            if cache_age < self.cache_ttl:
                return cached_check
        
        self.logger.info(f"Checking dependencies for Stage {stage}")
        
        # Get stage dependency configuration
        stage_dep = next((dep for dep in self.stage_dependencies if dep.stage == stage), None)
        if not stage_dep:
            return DependencyCheck(
                stage=stage,
                dependencies_met=True,
                completion_rates={},
                blocking_issues=[],
                ready_to_run=True
            )
        
        completion_rates = {}
        blocking_issues = []
        dependencies_met = True
        
        # Check each dependency
        for dep_stage in stage_dep.depends_on:
            if batch_ids:
                # Check specific batches
                completion_rate = self._get_batch_completion_rate(dep_stage, batch_ids)
            else:
                # Check all batches
                completion_rate = self._get_global_completion_rate(dep_stage)
            
            completion_rates[dep_stage] = completion_rate
            
            if completion_rate < stage_dep.completion_threshold:
                dependencies_met = False
                blocking_issues.append(
                    f"Stage {dep_stage} completion rate {completion_rate:.2%} "
                    f"below threshold {stage_dep.completion_threshold:.2%}"
                )
        
        ready_to_run = dependencies_met and len(blocking_issues) == 0
        
        check_result = DependencyCheck(
            stage=stage,
            dependencies_met=dependencies_met,
            completion_rates=completion_rates,
            blocking_issues=blocking_issues,
            ready_to_run=ready_to_run,
            timestamp=datetime.now()
        )
        
        # Cache result
        self.dependency_cache[cache_key] = check_result
        
        self.logger.info(f"Stage {stage} dependency check: {'READY' if ready_to_run else 'BLOCKED'}")
        
        return check_result
    
    def _get_batch_completion_rate(self, stage: int, batch_ids: List[str]) -> float:
        """Get completion rate for specific batches"""
        validation_result = self.batch_validator.validate_cross_batch_readiness(batch_ids)
        
        if stage == 1:
            completed = sum(result.stage_1_complete for result in validation_result.batch_results)
            total = sum(result.total_jobs for result in validation_result.batch_results)
        elif stage == 2:
            completed = sum(result.stage_2_complete for result in validation_result.batch_results)
            total = sum(result.total_jobs for result in validation_result.batch_results)
        elif stage == 3:
            completed = sum(result.stage_3_complete for result in validation_result.batch_results)
            total = sum(result.total_jobs for result in validation_result.batch_results)
        else:
            return 0.0
        
        return completed / total if total > 0 else 0.0
    
    def _get_global_completion_rate(self, stage: int) -> float:
        """Get completion rate for all batches"""
        if not os.path.exists(self.batch_validator.batch_results_dir):
            return 0.0
        
        # Find all batch IDs
        batch_ids = []
        for item in os.listdir(self.batch_validator.batch_results_dir):
            if item.startswith("batch_"):
                batch_ids.append(item.replace("batch_", ""))
        
        if not batch_ids:
            return 0.0
        
        return self._get_batch_completion_rate(stage, batch_ids)
    
    def can_start_stage(self, stage: int, batch_ids: Optional[List[str]] = None) -> bool:
        """
        Check if a stage can start based on dependencies.
        
        Args:
            stage: Stage number to check
            batch_ids: Optional list of batch IDs
            
        Returns:
            True if stage can start, False otherwise
        """
        dependency_check = self.check_stage_dependencies(stage, batch_ids)
        
        # Also check if stage is not already running or completed
        current_status = self.stage_status.get(stage, StageStatus.PENDING)
        
        can_start = (
            dependency_check.ready_to_run and
            current_status in [StageStatus.PENDING, StageStatus.FAILED]
        )
        
        self.logger.info(f"Stage {stage} can start: {can_start}")
        
        return can_start
    
    def set_stage_status(self, stage: int, status: StageStatus, batch_id: Optional[str] = None):
        """
        Set status for a stage.
        
        Args:
            stage: Stage number
            status: New status
            batch_id: Optional batch ID for batch-specific status
        """
        if batch_id:
            if batch_id not in self.batch_status:
                self.batch_status[batch_id] = {}
            self.batch_status[batch_id][stage] = status
            self.logger.debug(f"Batch {batch_id} Stage {stage} status set to {status.value}")
        else:
            self.stage_status[stage] = status
            self.logger.info(f"Stage {stage} status set to {status.value}")
        
        # Clear dependency cache when status changes
        self.dependency_cache.clear()
    
    def get_stage_status(self, stage: int, batch_id: Optional[str] = None) -> StageStatus:
        """
        Get status for a stage.
        
        Args:
            stage: Stage number
            batch_id: Optional batch ID
            
        Returns:
            Current stage status
        """
        if batch_id and batch_id in self.batch_status:
            return self.batch_status[batch_id].get(stage, StageStatus.PENDING)
        else:
            return self.stage_status.get(stage, StageStatus.PENDING)
    
    def wait_for_dependencies(self, stage: int, batch_ids: Optional[List[str]] = None, 
                            timeout: Optional[int] = None, check_interval: int = 30) -> bool:
        """
        Wait for dependencies to be satisfied.
        
        Args:
            stage: Stage number to wait for
            batch_ids: Optional list of batch IDs
            timeout: Maximum time to wait in seconds (None for infinite)
            check_interval: Time between checks in seconds
            
        Returns:
            True if dependencies satisfied, False if timeout
        """
        self.logger.info(f"Waiting for Stage {stage} dependencies")
        
        start_time = time.time()
        
        while True:
            if self.can_start_stage(stage, batch_ids):
                self.logger.info(f"Stage {stage} dependencies satisfied")
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout waiting for Stage {stage} dependencies")
                return False
            
            self.logger.debug(f"Stage {stage} dependencies not ready, waiting {check_interval}s")
            time.sleep(check_interval)
    
    def validate_cross_batch_dependencies(self, batch_ids: List[str]) -> Dict[int, DependencyCheck]:
        """
        Validate dependencies for all stages across multiple batches.
        
        Args:
            batch_ids: List of batch IDs to validate
            
        Returns:
            Dictionary mapping stage numbers to dependency checks
        """
        self.logger.info(f"Validating cross-batch dependencies for {len(batch_ids)} batches")
        
        results = {}
        
        for stage in [1, 2, 3]:
            check = self.check_stage_dependencies(stage, batch_ids)
            results[stage] = check
        
        return results
    
    def get_dependency_report(self, stage: int, batch_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate detailed dependency report for a stage.
        
        Args:
            stage: Stage number
            batch_ids: Optional list of batch IDs
            
        Returns:
            Dependency report dictionary
        """
        dependency_check = self.check_stage_dependencies(stage, batch_ids)
        current_status = self.get_stage_status(stage)
        
        report = {
            'stage': stage,
            'current_status': current_status.value,
            'dependencies_met': dependency_check.dependencies_met,
            'ready_to_run': dependency_check.ready_to_run,
            'completion_rates': dependency_check.completion_rates,
            'blocking_issues': dependency_check.blocking_issues,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add stage dependency configuration
        stage_dep = next((dep for dep in self.stage_dependencies if dep.stage == stage), None)
        if stage_dep:
            report['dependency_config'] = {
                'depends_on': stage_dep.depends_on,
                'completion_threshold': stage_dep.completion_threshold
            }
        
        # Add batch-specific status if batch IDs provided
        if batch_ids:
            report['batch_status'] = {}
            for batch_id in batch_ids:
                batch_status = self.get_stage_status(stage, batch_id)
                report['batch_status'][batch_id] = batch_status.value
        
        return report
    
    def reset_stage_status(self, stage: int, batch_id: Optional[str] = None):
        """
        Reset stage status to PENDING.
        
        Args:
            stage: Stage number to reset
            batch_id: Optional batch ID
        """
        self.set_stage_status(stage, StageStatus.PENDING, batch_id)
        self.dependency_cache.clear()
        
        self.logger.info(f"Reset Stage {stage} status to PENDING")
    
    def clear_dependency_cache(self):
        """Clear dependency cache"""
        self.dependency_cache.clear()
        self.logger.info("Dependency cache cleared")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.basicConfig(level=logging.INFO)
    
    dep_manager = DependencyManager(config)
    
    print("=== Dependency Manager Test ===")
    
    # Test Stage 1 (no dependencies)
    can_start_1 = dep_manager.can_start_stage(1)
    print(f"Stage 1 can start: {can_start_1}")
    
    # Test Stage 2 (depends on Stage 1)
    can_start_2 = dep_manager.can_start_stage(2)
    print(f"Stage 2 can start: {can_start_2}")
    
    # Test Stage 3 (depends on Stage 1 and 2)
    can_start_3 = dep_manager.can_start_stage(3)
    print(f"Stage 3 can start: {can_start_3}")
    
    # Generate dependency reports
    for stage in [1, 2, 3]:
        report = dep_manager.get_dependency_report(stage)
        print(f"\nStage {stage} Report:")
        print(f"  Status: {report['current_status']}")
        print(f"  Ready: {report['ready_to_run']}")
        print(f"  Dependencies Met: {report['dependencies_met']}")
        if report['blocking_issues']:
            print("  Blocking Issues:")
            for issue in report['blocking_issues']:
                print(f"    - {issue}")
