#!/usr/bin/env python3
"""
Test script for Master Blueprint Integration
Verifies that Stage 3 with Master Blueprint works correctly
"""

import os
import json
import sys
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append('.')

from final_aggregation_core import MasterBlueprintAggregator, AggregationResult
from stage_3_worker import Stage3Worker


def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_test_dna_data():
    """Create sample DNA data for testing"""
    return [
        {
            "job_id": "test_job_1",
            "url": "https://example1.com",
            "classification": "competitor",
            "dna_profile": {
                "page_structure": {
                    "html_patterns": ["H1", "H2", "P", "UL"],
                    "semantic_markup": True
                },
                "content_characteristics": {
                    "entity_density": 0.05,
                    "topic_focus": ["AI tools", "productivity"]
                },
                "ranking_factors": {
                    "evidence_quality": 75,
                    "authority_signals": ["backlinks", "citations"]
                }
            }
        },
        {
            "job_id": "test_job_2", 
            "url": "https://example2.com",
            "classification": "third_party",
            "dna_profile": {
                "page_structure": {
                    "html_patterns": ["H1", "H2", "BLOCKQUOTE", "P"],
                    "semantic_markup": True
                },
                "content_characteristics": {
                    "entity_density": 0.07,
                    "topic_focus": ["AI automation", "efficiency"]
                },
                "ranking_factors": {
                    "evidence_quality": 85,
                    "authority_signals": ["expert_quotes", "data_sources"]
                }
            }
        }
    ]


def test_master_blueprint_aggregator():
    """Test the Master Blueprint Aggregator directly"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Master Blueprint Aggregator...")
    
    try:
        # Create aggregator
        aggregator = MasterBlueprintAggregator()
        
        # Test data
        test_data = create_test_dna_data()
        run_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        query = "best AI productivity tools"
        
        # Run aggregation
        logger.info(f"Running aggregation for {len(test_data)} DNA profiles...")
        result = aggregator.aggregate_run_results(run_id, query, test_data)
        
        # Debug: Check if aggregation failed
        if result.error:
            logger.error(f"Aggregation failed with error: {result.error}")
            return False
        
        # Verify results
        assert result.run_id == run_id
        assert result.query == query
        assert result.total_analyzed == len(test_data)
        assert result.master_blueprint is not None
        assert "master_blueprint_id" in result.master_blueprint
        assert "layer_1_structural_framework" in result.master_blueprint
        assert "layer_2_entity_knowledge_base" in result.master_blueprint
        assert result.competitive_insights is not None
        assert len(result.content_recommendations) > 0
        assert result.summary_report is not None
        
        logger.info("✓ Master Blueprint Aggregator test passed!")
        logger.info(f"  - Generated blueprint: {result.master_blueprint.get('master_blueprint_id', 'Unknown')}")
        logger.info(f"  - Structural components: {len(result.master_blueprint.get('layer_1_structural_framework', []))}")
        logger.info(f"  - Entity knowledge: {len(result.master_blueprint.get('layer_2_entity_knowledge_base', {}))}")
        logger.info(f"  - Recommendations: {len(result.content_recommendations)}")
        
        return True
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"✗ Master Blueprint Aggregator test failed: {str(e)}")
        logger.error(f"Full traceback: {error_details}")
        return False


def test_stage_3_worker():
    """Test Stage 3 Worker with Master Blueprint integration"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Stage 3 Worker...")
    
    try:
        # Create worker with test config
        config = {
            "stage_3_aggregation": {
                "timeout_per_job": 300,
                "output_dir": "test_stage_3_results",
                "model_name": "gemini-3-pro-preview"
            },
            "pipeline": {
                "base_output_dir": "test_outputs"
            }
        }
        
        worker = Stage3Worker(config)
        
        # Create mock jobs
        from pipeline_models import Job
        jobs = []
        for i, test_data in enumerate(create_test_dna_data()):
            job = Job(
                job_id=test_data["job_id"],
                run_id="test_run",
                source_link={"url": test_data["url"], "text": "test text"},
                url=test_data["url"],
                text="test text",
                position=i,
                classification=test_data["classification"]
            )
            job.mark_stage_complete(2, "test_dna_analysis.json")  # Mark Stage 2 complete
            job.selected_for_stage_3 = True
            jobs.append(job)
        
        # Create test DNA files
        os.makedirs("test_outputs/stage_2_results", exist_ok=True)
        for job in jobs:
            job_dir = f"test_outputs/stage_2_results/job_{job.job_id}"
            os.makedirs(job_dir, exist_ok=True)
            
            test_data = next(d for d in create_test_dna_data() if d["job_id"] == job.job_id)
            with open(f"{job_dir}/dna_analysis.json", 'w') as f:
                json.dump(test_data, f, indent=2)
        
        # Run Stage 3
        run_id = f"test_worker_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = worker.process_run(run_id, "best AI productivity tools", jobs)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['run_id'] == run_id
        assert result['total_analyzed'] == len(jobs)
        assert 'output_paths' in result
        
        # Check output files exist
        output_paths = result['output_paths']
        assert os.path.exists(output_paths['aggregation_path'])
        assert os.path.exists(output_paths['blueprint_path'])  # New master blueprint file
        assert os.path.exists(output_paths['report_path'])
        
        logger.info("✓ Stage 3 Worker test passed!")
        logger.info(f"  - Status: {result['status']}")
        logger.info(f"  - Total analyzed: {result['total_analyzed']}")
        logger.info(f"  - Blueprint saved: {output_paths['blueprint_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Stage 3 Worker test failed: {str(e)}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    import shutil
    test_dirs = ["test_outputs", "test_stage_3_results"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def main():
    """Main test function"""
    logger = setup_logging()
    logger.info("Starting Master Blueprint Integration Tests")
    logger.info("=" * 50)
    
    success = True
    
    # Test Master Blueprint Aggregator
    if not test_master_blueprint_aggregator():
        success = False
    
    print()
    
    # Test Stage 3 Worker
    if not test_stage_3_worker():
        success = False
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    logger.info("=" * 50)
    if success:
        logger.info("🎉 ALL TESTS PASSED! Master Blueprint integration is working correctly.")
        logger.info("Stage 3 is now enhanced with Master Blueprint methodology.")
    else:
        logger.error("❌ SOME TESTS FAILED! Please check the implementation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
