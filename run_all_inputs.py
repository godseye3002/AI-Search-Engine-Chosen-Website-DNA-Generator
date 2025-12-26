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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append('.')

from pipeline_orchestrator import PipelineOrchestrator

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


def load_and_merge_ai_responses(directory: str) -> Dict[str, Any]:
    json_files = sorted([str(p) for p in Path(directory).glob('*.json')])
    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")

    merged_source_links = []
    queries = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            ai_response = json.load(f)

        q = ai_response.get('query')
        if q:
            queries.append(q)

        links = ai_response.get('source_links') or []
        if isinstance(links, list):
            merged_source_links.extend(links)

    unique_queries = []
    for q in queries:
        if q not in unique_queries:
            unique_queries.append(q)

    query_context = " | ".join(unique_queries) if unique_queries else os.path.basename(directory)

    return {
        'query': query_context,
        'source_links': merged_source_links,
        'batch_queries': unique_queries,
        'input_directory': directory,
        'input_files': json_files,
    }

def run_pipeline_for_directory(directory: str, logger: logging.Logger, max_workers: int) -> Dict[str, Any]:
    logger.info(f"Processing directory: {directory}")

    ai_response = load_and_merge_ai_responses(directory)
    run_id = os.path.basename(directory)

    orchestrator = PipelineOrchestrator()
    executed_run_id = orchestrator.run_pipeline_from_ai_response(ai_response, run_id_override=run_id)

    final_output_path = os.path.join(
        'outputs',
        'stage_3_results',
        f"run_{executed_run_id}",
        'final_aggregation.json'
    )

    return {
        'directory': directory,
        'run_id': executed_run_id,
        'final_output_path': final_output_path,
        'total_source_links': len(ai_response.get('source_links', [])),
        'unique_queries': ai_response.get('batch_queries', []),
    }

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
    
    max_workers = min(
        len(find_input_directories()),
        config.get('batch_processing', {}).get('max_concurrent_runs', 2)
    )
    
    # Find all input directories
    input_dirs = find_input_directories()
    if not input_dirs:
        logger.error("No input directories with JSON files found")
        return 1
    
    logger.info(f"Found {len(input_dirs)} input directories to process")
    
    total_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_pipeline_for_directory, directory, logger, max_workers): directory
            for directory in input_dirs
        }

        for future in as_completed(futures):
            directory = futures[future]
            try:
                result = future.result()
                total_results.append(result)
                logger.info(f"Final output for {directory}: {result['final_output_path']}")
            except Exception as e:
                logger.error(f"Failed to process directory {directory}: {e}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    
    total_runs = len(total_results)
    total_failed = max(0, len(input_dirs) - total_runs)
    
    logger.info(f"Total directories processed: {total_runs}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success rate: {total_runs/len(input_dirs)*100:.1f}%" if len(input_dirs) > 0 else "N/A")

    for result in sorted(total_results, key=lambda r: r['directory']):
        logger.info(f"Directory: {result['directory']}")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"Final output: {result['final_output_path']}")

    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
