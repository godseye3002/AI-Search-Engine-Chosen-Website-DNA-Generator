"""
Cross-Batch Aggregator

Handles aggregation across multiple validated batches.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from batch_validator import BatchValidator, CrossBatchValidationResult


@dataclass
class CrossBatchResult:
    """Result of cross-batch aggregation"""
    aggregation_id: str
    total_batches: int
    total_jobs: int
    aggregated_dna_profile: Dict[str, Any]
    cross_batch_insights: Dict[str, Any]
    unified_recommendations: List[Dict[str, Any]]
    created_at: datetime


class CrossBatchAggregator:
    """Aggregates results across multiple validated batches"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.base_output_dir = config.get('pipeline', {}).get('base_output_dir', 'outputs')
        self.batch_results_dir = os.path.join(self.base_output_dir, 'batch_results')
        self.cross_batch_dir = os.path.join(self.base_output_dir, 'cross_batch_results')
        
        os.makedirs(self.cross_batch_dir, exist_ok=True)
        
        self.validator = BatchValidator(config, logger)
        
        self.logger.info("Cross-Batch Aggregator initialized")
    
    def aggregate_cross_batch(self, batch_ids: List[str]) -> CrossBatchResult:
        """Aggregate results across multiple batches"""
        self.logger.info(f"Starting cross-batch aggregation for {len(batch_ids)} batches")
        
        # Validate batches first
        validation_result = self.validator.validate_cross_batch_readiness(batch_ids)
        
        if not validation_result.ready_for_aggregation:
            raise ValueError("Batches not ready for aggregation")
        
        # Load batch results
        batch_results = self._load_batch_results(batch_ids)
        
        # Aggregate DNA profiles
        aggregated_profile = self._aggregate_dna_profiles(batch_results)
        
        # Generate cross-batch insights
        cross_insights = self._generate_cross_insights(batch_results, aggregated_profile)
        
        # Create unified recommendations
        unified_recs = self._create_unified_recommendations(batch_results, cross_insights)
        
        # Create result
        result = CrossBatchResult(
            aggregation_id=f"cross_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_batches=len(batch_ids),
            total_jobs=validation_result.total_jobs,
            aggregated_dna_profile=aggregated_profile,
            cross_batch_insights=cross_insights,
            unified_recommendations=unified_recs,
            created_at=datetime.now()
        )
        
        # Save results
        self._save_cross_batch_results(result)
        
        self.logger.info(f"Cross-batch aggregation completed: {result.aggregation_id}")
        return result
    
    def _load_batch_results(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        """Load results from multiple batches"""
        batch_results = []
        
        for batch_id in batch_ids:
            batch_dir = os.path.join(self.batch_results_dir, f"batch_{batch_id}")
            detailed_file = os.path.join(batch_dir, 'detailed_results.json')
            
            if os.path.exists(detailed_file):
                with open(detailed_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    batch_results.extend(results)
            else:
                # Load individual run results
                for job_result in self._load_individual_batch_results(batch_id):
                    batch_results.append(job_result)
        
        return batch_results
    
    def _load_individual_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Load results from individual runs in a batch"""
        results = []
        
        batch_dir = os.path.join(self.batch_results_dir, f"batch_{batch_id}")
        jobs_file = os.path.join(batch_dir, 'job_results.json')
        
        if not os.path.exists(jobs_file):
            return results
        
        with open(jobs_file, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        for job in jobs:
            if job.get('run_id') and job.get('status') == 'completed':
                run_dir = os.path.join(self.base_output_dir, 'stage_3_results', f"run_{job['run_id']}")
                aggregation_file = os.path.join(run_dir, 'final_aggregation.json')
                
                if os.path.exists(aggregation_file):
                    with open(aggregation_file, 'r', encoding='utf-8') as f:
                        aggregation_data = json.load(f)
                    
                    results.append({
                        'job_id': job['job_id'],
                        'run_id': job['run_id'],
                        'query': job.get('query', ''),
                        'aggregation_results': aggregation_data
                    })
        
        return results
    
    def _aggregate_dna_profiles(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate DNA profiles across all results"""
        aggregated = {
            'page_structure': {},
            'content_characteristics': {},
            'technical_seo': {},
            'ranking_factors': {},
            'cross_batch_metrics': {
                'total_sources': len(batch_results),
                'aggregation_timestamp': datetime.now().isoformat()
            }
        }
        
        # Collect all profiles
        all_profiles = []
        for result in batch_results:
            if 'aggregation_results' in result:
                profile = result['aggregation_results'].get('aggregated_dna_profile', {})
                all_profiles.append(profile)
        
        if not all_profiles:
            return aggregated
        
        # Aggregate numeric values
        numeric_fields = {
            'page_structure': ['avg_heading_depth', 'content_to_html_ratio'],
            'content_characteristics': ['avg_sentence_length', 'readability_score'],
            'technical_seo': ['page_load_score', 'mobile_friendly_score'],
            'ranking_factors': ['content_quality_score', 'authority_score']
        }
        
        for category, fields in numeric_fields.items():
            for field in fields:
                values = []
                for profile in all_profiles:
                    if field in profile.get(category, {}):
                        try:
                            values.append(float(profile[category][field]))
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    aggregated[category][field] = {
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        # Aggregate categorical data
        categorical_fields = {
            'page_structure': ['heading_structure', 'content_organization'],
            'content_characteristics': ['content_types', 'tone_analysis'],
            'technical_seo': ['meta_tags', 'schema_markup'],
            'ranking_factors': ['ranking_signals', 'competitive_advantages']
        }
        
        for category, fields in categorical_fields.items():
            for field in fields:
                all_values = []
                for profile in all_profiles:
                    if field in profile.get(category, {}):
                        value = profile[category][field]
                        if isinstance(value, list):
                            all_values.extend(value)
                        else:
                            all_values.append(value)
                
                if all_values:
                    # Count frequencies
                    freq = {}
                    for val in all_values:
                        freq[val] = freq.get(val, 0) + 1
                    
                    aggregated[category][field] = {
                        'most_common': max(freq, key=freq.get),
                        'frequency_distribution': freq,
                        'unique_count': len(freq)
                    }
        
        return aggregated
    
    def _generate_cross_insights(self, batch_results: List[Dict[str, Any]], 
                                aggregated_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cross-batch analysis"""
        insights = {
            'performance_patterns': {},
            'content_gaps': [],
            'opportunity_areas': [],
            'competitive_landscape': {},
            'batch_comparison': {}
        }
        
        # Analyze performance patterns
        if 'ranking_factors' in aggregated_profile:
            ranking_factors = aggregated_profile['ranking_factors']
            
            # Find top performing factors
            top_factors = []
            for factor, data in ranking_factors.items():
                if 'average' in data:
                    top_factors.append((factor, data['average']))
            
            top_factors.sort(key=lambda x: x[1], reverse=True)
            insights['performance_patterns']['top_ranking_factors'] = top_factors[:5]
        
        # Identify content gaps
        all_queries = [r.get('query', '') for r in batch_results]
        unique_topics = set()
        for query in all_queries:
            # Simple topic extraction
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    unique_topics.add(word)
        
        insights['content_gaps'] = list(unique_topics)[:10]
        
        # Find opportunity areas
        if 'page_structure' in aggregated_profile:
            structure = aggregated_profile['page_structure']
            if 'content_to_html_ratio' in structure:
                ratio_data = structure['content_to_html_ratio']
                if ratio_data.get('average', 0) < 0.3:
                    insights['opportunity_areas'].append({
                        'area': 'Content Density',
                        'recommendation': 'Increase content-to-HTML ratio for better SEO performance',
                        'current_average': ratio_data.get('average', 0)
                    })
        
        return insights
    
    def _create_unified_recommendations(self, batch_results: List[Dict[str, Any]], 
                                       insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create unified recommendations from cross-batch analysis"""
        recommendations = []
        
        # High-priority recommendations based on aggregated data
        if 'performance_patterns' in insights:
            top_factors = insights['performance_patterns'].get('top_ranking_factors', [])
            if top_factors:
                recommendations.append({
                    'priority': 'high',
                    'category': 'ranking_optimization',
                    'recommendation': f"Focus on optimizing {top_factors[0][0]} as it shows highest correlation with ranking success",
                    'evidence': f"Average score: {top_factors[0][1]:.2f}",
                    'expected_impact': 'High'
                })
        
        # Content recommendations
        if 'content_gaps' in insights:
            gaps = insights['content_gaps'][:5]
            if gaps:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'content_strategy',
                    'recommendation': f"Create content addressing underrepresented topics: {', '.join(gaps[:3])}",
                    'evidence': f"Identified {len(gaps)} content gaps across {len(batch_results)} analyses",
                    'expected_impact': 'Medium'
                })
        
        # Technical recommendations
        if 'opportunity_areas' in insights:
            for opportunity in insights['opportunity_areas']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'technical_optimization',
                    'recommendation': opportunity['recommendation'],
                    'evidence': f"Current performance: {opportunity.get('current_average', 'N/A')}",
                    'expected_impact': 'High'
                })
        
        return recommendations
    
    def _save_cross_batch_results(self, result: CrossBatchResult):
        """Save cross-batch aggregation results"""
        output_dir = os.path.join(self.cross_batch_dir, result.aggregation_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main aggregation result
        result_file = os.path.join(output_dir, 'cross_batch_aggregation.json')
        
        result_data = {
            'aggregation_id': result.aggregation_id,
            'total_batches': result.total_batches,
            'total_jobs': result.total_jobs,
            'created_at': result.created_at.isoformat(),
            'aggregated_dna_profile': result.aggregated_dna_profile,
            'cross_batch_insights': result.cross_batch_insights,
            'unified_recommendations': result.unified_recommendations
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'summary_report.txt')
        
        summary = f"""
Cross-Batch Aggregation Summary
==============================

Aggregation ID: {result.aggregation_id}
Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Scope:
- Total Batches: {result.total_batches}
- Total Jobs: {result.total_jobs}

Key Insights:
"""
        
        for insight_type, data in result.cross_batch_insights.items():
            summary += f"\n{insight_type.replace('_', ' ').title()}:\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    summary += f"  - {key}: {value}\n"
        
        summary += f"\nRecommendations ({len(result.unified_recommendations)}):\n"
        for i, rec in enumerate(result.unified_recommendations, 1):
            summary += f"\n{i}. [{rec['priority'].upper()}] {rec['recommendation']}\n"
            summary += f"   Category: {rec['category']}\n"
            summary += f"   Expected Impact: {rec['expected_impact']}\n"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info(f"Cross-batch results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.basicConfig(level=logging.INFO)
    
    aggregator = CrossBatchAggregator(config)
    
    print("=== Cross-Batch Aggregator Test ===")
    
    # Find available batches
    if os.path.exists(aggregator.batch_results_dir):
        batch_ids = []
        for item in os.listdir(aggregator.batch_results_dir):
            if item.startswith("batch_"):
                batch_ids.append(item.replace("batch_", ""))
        
        if batch_ids:
            print(f"Found {len(batch_ids)} batches: {batch_ids}")
            
            try:
                result = aggregator.aggregate_cross_batch(batch_ids[:2])  # Test with first 2 batches
                
                print(f"Cross-batch aggregation completed:")
                print(f"  Aggregation ID: {result.aggregation_id}")
                print(f"  Total Batches: {result.total_batches}")
                print(f"  Total Jobs: {result.total_jobs}")
                print(f"  Recommendations: {len(result.unified_recommendations)}")
                
            except Exception as e:
                print(f"Aggregation failed: {e}")
        else:
            print("No batches found")
    else:
        print("Batch results directory not found")
