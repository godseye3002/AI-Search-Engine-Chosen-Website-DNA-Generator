#!/usr/bin/env python3
"""
Run all input files through the complete pipeline
This script processes all JSON files in the inputs directory
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from batch_processor import BatchProcessor

def setup_logging():
    """Setup logging for the run"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'batch_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def find_input_directories():
    """Find all input directories"""
    base_inputs_dir = Path("inputs")
    if not base_inputs_dir.exists():
        raise FileNotFoundError("inputs directory not found")
    
    input_dirs = []
    for item in base_inputs_dir.iterdir():
        if item.is_dir():
            # Check if directory contains JSON files
            json_files = list(item.glob("*.json"))
            if json_files:
                input_dirs.append(str(item))
                print(f"Found input directory: {item} ({len(json_files)} files)")
    
    return input_dirs

def run_batch_for_directory(batch_processor, directory, logger):
    """Run batch processing for a single directory"""
    logger.info(f"Processing directory: {directory}")
    
    try:
        # Create batch from directory and get the batch_id
        batch_id = f"batch_{os.path.basename(directory)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        jobs = batch_processor.create_batch_from_directory(directory)
        
        # The batch_id should match what was created in create_batch_from_directory
        # Let's check what batch_id was actually used
        if jobs:
            # Extract the actual batch_id from the first job's job_id
            actual_batch_id = "_".join(jobs[0].job_id.split('_')[:4])  # Get first 4 parts
            logger.info(f"Using batch_id: {actual_batch_id} with {len(jobs)} jobs")
            batch_id = actual_batch_id
        else:
            logger.error("No jobs created from directory")
            return None
        
        logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")
        
        # Process batch in parallel
        result = batch_processor.process_batch_parallel(batch_id)
        
        logger.info(f"Batch {directory} completed: {result.completed_jobs}/{result.total_jobs} successful")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process directory {directory}: {str(e)}")
        return None

def main():
    """Main function to run all inputs"""
    logger = setup_logging()
    logger.info("Starting batch processing of all input directories")
    
    # Load config
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Create batch processor
    try:
        batch_processor = BatchProcessor(config, logger)
        logger.info("Batch processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize batch processor: {e}")
        return 1
    
    # Find all input directories
    input_dirs = find_input_directories()
    if not input_dirs:
        logger.error("No input directories with JSON files found")
        return 1
    
    logger.info(f"Found {len(input_dirs)} input directories to process")
    
    # Process each directory
    total_results = []
    for directory in input_dirs:
        logger.info(f"\n{'='*60}")
        result = run_batch_for_directory(batch_processor, directory, logger)
        if result:
            total_results.append({
                'directory': directory,
                'result': result
            })
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    
    total_jobs = sum(r['result'].total_jobs for r in total_results)
    total_completed = sum(r['result'].completed_jobs for r in total_results)
    total_failed = sum(r['result'].failed_jobs for r in total_results)
    
    logger.info(f"Total directories processed: {len(total_results)}")
    logger.info(f"Total jobs: {total_jobs}")
    logger.info(f"Completed: {total_completed}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success rate: {total_completed/total_jobs*100:.1f}%" if total_jobs > 0 else "N/A")
    
    # Cleanup
    batch_processor.shutdown()
    logger.info("Batch processor shutdown complete")
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
