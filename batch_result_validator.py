"""
Batch Result Validator

Validates and quality-checks batch processing results.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from batch_validator import BatchValidationResult


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation check"""
    level: ValidationLevel
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class BatchQualityReport:
    """Quality report for a batch"""
    batch_id: str
    overall_score: float  # 0.0 to 1.0
    validation_results: List[ValidationResult]
    quality_metrics: Dict[str, Any]
    recommendations: List[str]
    validation_time: datetime


class BatchResultValidator:
    """Validates batch processing results for quality and completeness"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation thresholds
        self.quality_thresholds = config.get('validation', {}).get('quality_thresholds', {
            'min_completion_rate': 0.90,
            'min_data_quality_score': 0.70,
            'max_error_rate': 0.10,
            'min_content_coverage': 0.80
        })
        
        # Output directories
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        self.batch_results_dir = os.path.join(self.base_output_dir, 'batch_results')
        self.validation_reports_dir = os.path.join(self.base_output_dir, 'validation_reports')
        
        os.makedirs(self.validation_reports_dir, exist_ok=True)
        
        self.logger.info("Batch Result Validator initialized")
    
    def validate_batch_quality(self, batch_id: str) -> BatchQualityReport:
        """
        Perform comprehensive quality validation for a batch.
        
        Args:
            batch_id: Batch identifier to validate
            
        Returns:
            BatchQualityReport with validation results
        """
        self.logger.info(f"Starting quality validation for batch {batch_id}")
        
        validation_time = datetime.now()
        validation_results = []
        quality_metrics = {}
        recommendations = []
        
        try:
            # Load batch data
            batch_data = self._load_batch_data(batch_id)
            
            # Validate completeness
            completeness_results = self._validate_completeness(batch_id, batch_data)
            validation_results.extend(completeness_results)
            
            # Validate data quality
            quality_results = self._validate_data_quality(batch_id, batch_data)
            validation_results.extend(quality_results)
            
            # Validate consistency
            consistency_results = self._validate_consistency(batch_id, batch_data)
            validation_results.extend(consistency_results)
            
            # Validate performance
            performance_results = self._validate_performance(batch_id, batch_data)
            validation_results.extend(performance_results)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(validation_results, batch_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, quality_metrics)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(quality_metrics)
            
            report = BatchQualityReport(
                batch_id=batch_id,
                overall_score=overall_score,
                validation_results=validation_results,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                validation_time=validation_time
            )
            
            # Save validation report
            self._save_validation_report(report)
            
            self.logger.info(f"Batch {batch_id} quality validation completed: score {overall_score:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Quality validation failed for batch {batch_id}: {e}")
            
            # Return error report
            error_result = ValidationResult(
                level=ValidationLevel.CRITICAL,
                category="system",
                message=f"Validation failed: {str(e)}",
                suggestion="Check batch data and configuration"
            )
            
            return BatchQualityReport(
                batch_id=batch_id,
                overall_score=0.0,
                validation_results=[error_result],
                quality_metrics={},
                recommendations=["Fix validation errors"],
                validation_time=validation_time
            )
    
    def _load_batch_data(self, batch_id: str) -> Dict[str, Any]:
        """Load batch data for validation"""
        batch_dir = os.path.join(self.batch_results_dir, f"batch_{batch_id}")
        
        batch_data = {
            'batch_id': batch_id,
            'batch_dir': batch_dir,
            'summary': {},
            'job_results': [],
            'detailed_results': []
        }
        
        # Load batch summary
        summary_file = os.path.join(batch_dir, 'batch_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                batch_data['summary'] = json.load(f)
        
        # Load job results
        jobs_file = os.path.join(batch_dir, 'job_results.json')
        if os.path.exists(jobs_file):
            with open(jobs_file, 'r', encoding='utf-8') as f:
                batch_data['job_results'] = json.load(f)
        
        # Load detailed results
        detailed_file = os.path.join(batch_dir, 'detailed_results.json')
        if os.path.exists(detailed_file):
            with open(detailed_file, 'r', encoding='utf-8') as f:
                batch_data['detailed_results'] = json.load(f)
        
        return batch_data
    
    def _validate_completeness(self, batch_id: str, batch_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate batch completeness"""
        results = []
        
        summary = batch_data.get('summary', {})
        job_results = batch_data.get('job_results', [])
        
        total_jobs = summary.get('total_jobs', len(job_results))
        completed_jobs = summary.get('completed_jobs', 0)
        failed_jobs = summary.get('failed_jobs', 0)
        
        completion_rate = completed_jobs / total_jobs if total_jobs > 0 else 0
        
        # Check completion rate
        if completion_rate < self.quality_thresholds['min_completion_rate']:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="completeness",
                message=f"Low completion rate: {completion_rate:.2%}",
                details={
                    'completed': completed_jobs,
                    'total': total_jobs,
                    'threshold': self.quality_thresholds['min_completion_rate']
                },
                suggestion="Review failed jobs and retry processing"
            ))
        elif completion_rate < 1.0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="completeness",
                message=f"Some jobs failed: {failed_jobs} of {total_jobs}",
                details={
                    'failed': failed_jobs,
                    'total': total_jobs
                },
                suggestion="Investigate failed jobs for patterns"
            ))
        
        # Check for missing files
        required_files = ['batch_summary.json', 'job_results.json']
        missing_files = []
        
        for file_name in required_files:
            file_path = os.path.join(batch_data['batch_dir'], file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="completeness",
                message=f"Missing required files: {missing_files}",
                details={'missing_files': missing_files},
                suggestion="Regenerate missing output files"
            ))
        
        return results
    
    def _validate_data_quality(self, batch_id: str, batch_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data quality"""
        results = []
        
        detailed_results = batch_data.get('detailed_results', [])
        
        if not detailed_results:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="data_quality",
                message="No detailed results available for quality assessment",
                suggestion="Ensure detailed results are generated during processing"
            ))
            return results
        
        # Validate aggregation results
        quality_scores = []
        content_coverage_scores = []
        
        for result in detailed_results:
            if 'aggregation_results' in result:
                agg_data = result['aggregation_results']
                
                # Check DNA profile quality
                dna_profile = agg_data.get('aggregated_dna_profile', {})
                if dna_profile:
                    # Simple quality score based on data completeness
                    profile_completeness = self._assess_profile_completeness(dna_profile)
                    quality_scores.append(profile_completeness)
                
                # Check content coverage
                recommendations = agg_data.get('content_recommendations', {})
                if recommendations:
                    coverage_score = len(recommendations.get('recommendations', [])) / 10.0  # Normalize to 0-1
                    content_coverage_scores.append(min(coverage_score, 1.0))
        
        # Average quality scores
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_coverage = sum(content_coverage_scores) / len(content_coverage_scores) if content_coverage_scores else 0
        
        # Check quality thresholds
        if avg_quality < self.quality_thresholds['min_data_quality_score']:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="data_quality",
                message=f"Low data quality score: {avg_quality:.2f}",
                details={
                    'average_score': avg_quality,
                    'threshold': self.quality_thresholds['min_data_quality_score']
                },
                suggestion="Review input data quality and processing parameters"
            ))
        
        if avg_coverage < self.quality_thresholds['min_content_coverage']:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="data_quality",
                message=f"Low content coverage: {avg_coverage:.2f}",
                details={
                    'coverage_score': avg_coverage,
                    'threshold': self.quality_thresholds['min_content_coverage']
                },
                suggestion="Expand content analysis scope"
            ))
        
        return results
    
    def _validate_consistency(self, batch_id: str, batch_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data consistency across batch"""
        results = []
        
        job_results = batch_data.get('job_results', [])
        detailed_results = batch_data.get('detailed_results', [])
        
        # Check consistency between job results and detailed results
        if len(job_results) != len(detailed_results):
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="consistency",
                message=f"Job count mismatch: {len(job_results)} jobs vs {len(detailed_results)} detailed results",
                details={
                    'job_results_count': len(job_results),
                    'detailed_results_count': len(detailed_results)
                },
                suggestion="Ensure all jobs have corresponding detailed results"
            ))
        
        # Check for consistent data structures
        if detailed_results:
            first_result = detailed_results[0]
            expected_keys = set(first_result.keys())
            
            inconsistent_results = []
            for i, result in enumerate(detailed_results[1:], 1):
                if set(result.keys()) != expected_keys:
                    inconsistent_results.append(i)
            
            if inconsistent_results:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="consistency",
                    message=f"Inconsistent data structures in {len(inconsistent_results)} results",
                    details={'inconsistent_indices': inconsistent_results},
                    suggestion="Standardize result data structures"
                ))
        
        return results
    
    def _validate_performance(self, batch_id: str, batch_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate batch performance metrics"""
        results = []
        
        summary = batch_data.get('summary', {})
        job_results = batch_data.get('job_results', [])
        
        # Check processing time
        total_time = summary.get('total_processing_time', 0)
        total_jobs = summary.get('total_jobs', len(job_results))
        
        if total_jobs > 0:
            avg_time_per_job = total_time / total_jobs
            
            # Performance warning if too slow
            if avg_time_per_job > 300:  # 5 minutes per job
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="performance",
                    message=f"Slow processing: {avg_time_per_job:.1f}s per job",
                    details={
                        'avg_time_per_job': avg_time_per_job,
                        'total_time': total_time,
                        'total_jobs': total_jobs
                    },
                    suggestion="Optimize processing parameters or increase resources"
                ))
        
        # Check error rate
        failed_jobs = summary.get('failed_jobs', 0)
        error_rate = failed_jobs / total_jobs if total_jobs > 0 else 0
        
        if error_rate > self.quality_thresholds['max_error_rate']:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="performance",
                message=f"High error rate: {error_rate:.2%}",
                details={
                    'error_rate': error_rate,
                    'failed_jobs': failed_jobs,
                    'total_jobs': total_jobs,
                    'threshold': self.quality_thresholds['max_error_rate']
                },
                suggestion="Investigate error patterns and improve error handling"
            ))
        
        return results
    
    def _assess_profile_completeness(self, dna_profile: Dict[str, Any]) -> float:
        """Assess completeness of DNA profile"""
        required_categories = ['page_structure', 'content_characteristics', 'technical_seo', 'ranking_factors']
        present_categories = sum(1 for cat in required_categories if cat in dna_profile and dna_profile[cat])
        
        return present_categories / len(required_categories)
    
    def _calculate_quality_metrics(self, validation_results: List[ValidationResult], 
                                 batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics from validation results"""
        metrics = {}
        
        # Count validation levels
        level_counts = {level.value: 0 for level in ValidationLevel}
        category_counts = {}
        
        for result in validation_results:
            level_counts[result.level.value] += 1
            
            if result.category not in category_counts:
                category_counts[result.category] = 0
            category_counts[result.category] += 1
        
        metrics['validation_counts'] = level_counts
        metrics['category_counts'] = category_counts
        
        # Calculate severity score
        severity_weights = {
            ValidationLevel.CRITICAL.value: 1.0,
            ValidationLevel.ERROR.value: 0.5,
            ValidationLevel.WARNING.value: 0.2,
            ValidationLevel.INFO.value: 0.05
        }
        
        total_severity = sum(
            level_counts[level] * weight 
            for level, weight in severity_weights.items()
        )
        
        max_possible_severity = sum(level_counts.values()) * 1.0
        severity_score = 1.0 - (total_severity / max_possible_severity) if max_possible_severity > 0 else 1.0
        
        metrics['severity_score'] = severity_score
        
        # Batch-specific metrics
        summary = batch_data.get('summary', {})
        metrics['batch_metrics'] = {
            'total_jobs': summary.get('total_jobs', 0),
            'completed_jobs': summary.get('completed_jobs', 0),
            'failed_jobs': summary.get('failed_jobs', 0),
            'completion_rate': summary.get('completed_jobs', 0) / max(summary.get('total_jobs', 1), 1),
            'processing_time': summary.get('total_processing_time', 0)
        }
        
        return metrics
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Group by category
        category_issues = {}
        for result in validation_results:
            if result.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                if result.category not in category_issues:
                    category_issues[result.category] = []
                category_issues[result.category].append(result)
        
        # Generate recommendations for each category with issues
        for category, issues in category_issues.items():
            if category == 'completeness':
                recommendations.append("Improve job completion rate by addressing failed jobs")
            elif category == 'data_quality':
                recommendations.append("Enhance data quality through better input validation")
            elif category == 'consistency':
                recommendations.append("Standardize data structures and processing workflows")
            elif category == 'performance':
                recommendations.append("Optimize processing parameters and resource allocation")
            elif category == 'system':
                recommendations.append("Review system configuration and error handling")
        
        # Add general recommendations
        severity_score = quality_metrics.get('severity_score', 1.0)
        if severity_score < 0.8:
            recommendations.append("Address critical validation issues before proceeding")
        
        completion_rate = quality_metrics.get('batch_metrics', {}).get('completion_rate', 1.0)
        if completion_rate < 0.95:
            recommendations.append("Investigate and resolve job failures to improve reliability")
        
        return recommendations
    
    def _calculate_overall_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        severity_score = quality_metrics.get('severity_score', 1.0)
        completion_rate = quality_metrics.get('batch_metrics', {}).get('completion_rate', 1.0)
        
        # Weighted combination
        overall_score = (severity_score * 0.6) + (completion_rate * 0.4)
        
        return round(overall_score, 3)
    
    def _save_validation_report(self, report: BatchQualityReport):
        """Save validation report to file"""
        timestamp = report.validation_time.strftime('%Y%m%d_%H%M%S')
        filename = f"validation_report_{report.batch_id}_{timestamp}.json"
        filepath = os.path.join(self.validation_reports_dir, filename)
        
        report_data = {
            'batch_id': report.batch_id,
            'overall_score': report.overall_score,
            'validation_time': report.validation_time.isoformat(),
            'quality_metrics': report.quality_metrics,
            'validation_results': [
                {
                    'level': result.level.value,
                    'category': result.category,
                    'message': result.message,
                    'details': result.details,
                    'suggestion': result.suggestion
                }
                for result in report.validation_results
            ],
            'recommendations': report.recommendations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report saved to {filepath}")
    
    def validate_multiple_batches(self, batch_ids: List[str]) -> Dict[str, BatchQualityReport]:
        """Validate multiple batches"""
        reports = {}
        
        for batch_id in batch_ids:
            try:
                report = self.validate_batch_quality(batch_id)
                reports[batch_id] = report
            except Exception as e:
                self.logger.error(f"Failed to validate batch {batch_id}: {e}")
        
        return reports
    
    def get_batch_quality_summary(self, batch_id: str) -> Dict[str, Any]:
        """Get quality summary for a batch"""
        # Find latest validation report
        batch_reports = []
        for filename in os.listdir(self.validation_reports_dir):
            if filename.startswith(f"validation_report_{batch_id}_"):
                batch_reports.append(filename)
        
        if not batch_reports:
            return {'error': 'No validation reports found'}
        
        # Get latest report
        latest_report = sorted(batch_reports)[-1]
        report_path = os.path.join(self.validation_reports_dir, latest_report)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        return {
            'batch_id': report_data['batch_id'],
            'overall_score': report_data['overall_score'],
            'validation_time': report_data['validation_time'],
            'issue_count': len(report_data['validation_results']),
            'recommendations_count': len(report_data['recommendations']),
            'top_issues': report_data['validation_results'][:3]
        }


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.basicConfig(level=logging.INFO)
    
    validator = BatchResultValidator(config)
    
    print("=== Batch Result Validator Test ===")
    
    # Find available batches
    if os.path.exists(validator.batch_results_dir):
        batch_ids = []
        for item in os.listdir(validator.batch_results_dir):
            if item.startswith("batch_"):
                batch_ids.append(item.replace("batch_", ""))
        
        if batch_ids:
            print(f"Found {len(batch_ids)} batches: {batch_ids}")
            
            # Validate first batch
            test_batch = batch_ids[0]
            print(f"\nValidating batch {test_batch}...")
            
            report = validator.validate_batch_quality(test_batch)
            
            print(f"Validation completed:")
            print(f"  Overall Score: {report.overall_score:.3f}")
            print(f"  Validation Results: {len(report.validation_results)}")
            print(f"  Recommendations: {len(report.recommendations)}")
            
            # Show top issues
            critical_issues = [r for r in report.validation_results if r.level == ValidationLevel.CRITICAL]
            error_issues = [r for r in report.validation_results if r.level == ValidationLevel.ERROR]
            
            if critical_issues:
                print(f"\nCritical Issues ({len(critical_issues)}):")
                for issue in critical_issues:
                    print(f"  - {issue.message}")
            
            if error_issues:
                print(f"\nErrors ({len(error_issues)}):")
                for issue in error_issues[:3]:  # Show first 3
                    print(f"  - {issue.message}")
            
            if report.recommendations:
                print(f"\nRecommendations:")
                for rec in report.recommendations[:3]:  # Show first 3
                    print(f"  - {rec}")
            
        else:
            print("No batches found")
    else:
        print("Batch results directory not found")
