"""
Database-Driven Pipeline Orchestrator

Integrates with Supabase for dual-source data workflow:
- Fetches input from Supabase tables
- Processes through DNA pipeline
- Saves results back to Supabase
"""

import os
import json
import uuid
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from utils.batch_calculator import calculate_batches, create_batches_with_items, BatchInfo
from utils.timeout_handler import execute_with_timeout, ExecutionResult, TimeoutResult
from pipeline_models import Job, PipelineRun
from job_queue_manager import JobQueueManager
from stage_1_worker import Stage1Worker
from stage_2_worker import Stage2Worker
from stage_3_worker import Stage3Worker
from database.supabase_manager import SupabaseDataManager, DataSource, ProductAnalysisRecord, DNAAnalysisRecord
from utils.env_utils import get_log_level


class DatabasePipelineOrchestrator:
    """Database-driven pipeline orchestrator for Supabase integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.queue_manager = JobQueueManager(self.config, self.logger)
        self.stage_1_worker = Stage1Worker(self.config, self.logger)
        self.stage_2_worker = Stage2Worker(self.config, self.logger)
        self.stage_3_worker = Stage3Worker(self.config, self.logger)
        
        # Database manager
        self.db_manager = SupabaseDataManager()
        
        # Active runs
        self.active_runs: Dict[str, PipelineRun] = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', get_log_level())
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _generate_input_hash(self, input_rows: List[Any]) -> str:
        """
        Generate a hash from input row IDs to detect data changes
        
        Args:
            input_rows: List of input database rows
            
        Returns:
            SHA-256 hex digest of sorted input IDs
        """
        if not input_rows:
            # Handle empty input
            return hashlib.sha256(b"").hexdigest()
        
        # Extract IDs from rows (assuming they have 'id' field)
        try:
            ids = [str(row.id) for row in input_rows]
        except AttributeError:
            # Fallback if rows don't have 'id' field
            ids = [str(row) for row in input_rows]
        
        # Sort IDs to ensure consistent hash regardless of order
        sorted_ids = sorted(ids)
        
        # Combine IDs into a single string
        combined_ids = ",".join(sorted_ids)
        
        # Generate SHA-256 hash
        hash_digest = hashlib.sha256(combined_ids.encode('utf-8')).hexdigest()
        
        self.logger.debug(f"Generated input hash from {len(sorted_ids)} rows: {hash_digest[:12]}...")
        return hash_digest

    def _build_ai_response_from_input_rows(self, input_rows: List[Any]) -> Dict[str, Any]:
        if not input_rows:
            return {
                'query': '',
                'source_links': []
            }

        base_response: Dict[str, Any] = {}
        merged_source_links: List[Any] = []

        for row in input_rows:
            converted = self._convert_to_ai_response(getattr(row, 'raw_serp_results', None))
            if not base_response:
                base_response = converted if isinstance(converted, dict) else {}

            links = []
            if isinstance(converted, dict):
                links = converted.get('source_links') or []
            if isinstance(links, list) and links:
                merged_source_links.extend(links)

        if not base_response:
            base_response = {
                'query': '',
                'source_links': []
            }

        # Deduplicate links while preserving order
        unique_links: List[Any] = []
        seen: set = set()
        for link in merged_source_links:
            if isinstance(link, dict):
                key = link.get('raw_url') or link.get('url') or json.dumps(link, sort_keys=True)
            else:
                key = str(link)

            if key in seen:
                continue
            seen.add(key)
            unique_links.append(link)

        base_response['source_links'] = unique_links

        # Prefer a query if base_response does not have one
        if not base_response.get('query'):
            for row in input_rows:
                converted = self._convert_to_ai_response(getattr(row, 'raw_serp_results', None))
                if isinstance(converted, dict) and converted.get('query'):
                    base_response['query'] = converted.get('query')
                    break

        self.logger.info(
            f"Built ai_response from {len(input_rows)} input rows: source_links_count={len(unique_links)}"
        )
        return base_response
    
    def _create_directories(self):
        """Create necessary output directories"""
        base_dir = self.config.get('pipeline', {}).get('base_output_dir', 'outputs')
        directories = [
            'stage_1_results',
            'stage_2_results', 
            'stage_3_results'
        ]
        
        for directory in directories:
            full_path = os.path.join(base_dir, directory)
            os.makedirs(full_path, exist_ok=True)
    
    def process_product_from_database(self, data_source: DataSource, product_id: str) -> Dict[str, Any]:
        """
        Process a single product from database through the complete pipeline
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            product_id: Product identifier
            
        Returns:
            Processing result with status and details
        """
        self.logger.info(f"Starting database pipeline processing for product {product_id} from {data_source.value} source")
        
        try:
            # Fetch all input rows for the product first
            input_rows = self.db_manager.fetch_all_input_rows_for_product(data_source, product_id)

            if not input_rows:
                raise ValueError(f"Product {product_id} not found in {data_source.value} table")
            
            # Generate hash from input rows
            current_input_hash = self._generate_input_hash(input_rows)
            self.logger.info(f"Generated input hash for product {product_id}: {current_input_hash[:12]}...")
            
            # Check if analysis already exists and compare data freshness
            should_skip, existing_analysis = self.db_manager.check_existing_analysis(data_source, product_id, current_input_hash)
            
            if should_skip:
                if existing_analysis:
                    self.logger.info(f"Product {product_id} data is unchanged, skipping processing")
                    return {
                        "status": "skipped",
                        "product_id": product_id,
                        "data_source": data_source.value,
                        "existing_analysis_id": existing_analysis.id,
                        "message": "Skipped - Up to date"
                    }
                else:
                    # This shouldn't happen with the new logic, but keeping for safety
                    self.logger.info(f"Product {product_id} has no existing analysis, but should_skip is True")
                    pass
            elif existing_analysis:
                # Data has changed, log this and continue with reprocessing
                self.logger.info(f"Product {product_id} data has changed, reprocessing (existing analysis ID: {existing_analysis.id})")
                is_reprocessing = True
            else:
                # New product
                self.logger.info(f"Product {product_id} has no existing analysis, processing for the first time")
                is_reprocessing = False
            
            # Build ai_response from ALL input rows (many-to-one)
            ai_response = self._build_ai_response_from_input_rows(input_rows)
            
            # Check if source_links is empty - if so, skip processing
            if not ai_response.get('source_links'):
                self.logger.info(f"Skipping product {product_id} - no source_links found")
                return {
                    "status": "skipped",
                    "product_id": product_id,
                    "data_source": data_source.value,
                    "reason": "No source_links found in raw_serp_results"
                }
            
            # Generate stable run_id
            run_id = f"dna_{data_source.value}_{product_id}_{int(time.time())}"
            
            # Run pipeline
            final_result = self.run_pipeline_from_ai_response(ai_response, run_id_override=run_id)

            if final_result.get('status') != 'completed':
                raise RuntimeError(final_result.get('error') or "Pipeline did not complete")

            # In-memory pipeline: Stage 3 returns the master blueprint directly
            master_blueprint = final_result.get('master_blueprint') or {}
            if not isinstance(master_blueprint, dict):
                self.logger.warning("Master blueprint is not a dict; using empty dict")
                master_blueprint = {}
            
            # Save results to database
            dna_record = DNAAnalysisRecord(
                product_id=product_id,
                run_id=run_id,
                dna_blueprint=master_blueprint,
                status="completed",
                input_data_hash=current_input_hash
            )
            
            saved_record = self.db_manager.save_dna_analysis(data_source, dna_record)

            try:
                from success_email_sender import send_success_email_to_user
                send_success_email_to_user(
                    product_id=product_id,
                    data_source=data_source.value,
                    analysis_id=getattr(saved_record, "id", None),
                    run_id=run_id,
                )
            except Exception as email_err:
                self.logger.warning(
                    f"Success email send failed for product {product_id} ({data_source.value}): {email_err}"
                )
            
            self.logger.info(f"Successfully processed product {product_id} with analysis ID {saved_record.id}")
            
            # Determine the appropriate message
            if 'is_reprocessing' in locals() and is_reprocessing:
                message = "Processed - Data Updated"
            else:
                message = "Processed - New Data"
            
            return {
                "status": "completed",
                "product_id": product_id,
                "data_source": data_source.value,
                "run_id": run_id,
                "analysis_id": saved_record.id,
                "final_output_path": None,
                "message": message
            }
            
        except Exception as e:
            self.logger.error(f"Error processing product {product_id}: {str(e)}")
            try:
                from error_email_sender import send_ai_error_email
                send_ai_error_email(
                    error=e,
                    error_context="Database orchestrator failed while processing product",
                    metadata={
                        "product_id": product_id,
                        "data_source": data_source.value if hasattr(data_source, 'value') else str(data_source),
                        "run_id": locals().get('run_id'),
                        "stage": "database_pipeline_orchestrator.process_product_from_database",
                    },
                )
            except Exception as email_err:
                self.logger.error(f"Failed to send error email for product {product_id}: {email_err}")
            
            # Update status if we have a record
            try:
                if 'dna_record' in locals():
                    if getattr(dna_record, 'id', None) is not None:
                        self.db_manager.update_dna_analysis_status(data_source, dna_record.id, "failed")
                    else:
                        self.logger.warning(
                            f"Skipping status update because dna_record.id is None for product {product_id}"
                        )
            except:
                pass
            
            return {
                "status": "failed",
                "product_id": product_id,
                "data_source": data_source.value,
                "error": str(e),
                "message": "Processing failed"
            }
    
    def process_batch_from_database(self, data_source: DataSource, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of products from database
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            limit: Maximum number of products to process
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Starting batch processing from {data_source.value} source")
        
        # Fetch pending products
        products = self.db_manager.fetch_pending_products(data_source, limit)
        if not products:
            self.logger.info(f"No pending products found for {data_source.value}")
            return []
        
        self.logger.info(f"Found {len(products)} products to process")
        
        results = []
        for product in products:
            result = self.process_product_from_database(data_source, product.product_id)
            results.append(result)
        
        # Log batch summary
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        skipped = sum(1 for r in results if r["status"] == "already_exists")
        
        self.logger.info(f"Batch processing complete: {completed} completed, {failed} failed, {skipped} skipped")
        
        return results
    
    def _convert_to_ai_response(self, raw_serp_results: Any) -> Dict[str, Any]:
        """
        Convert raw SERP results to AI response format
        
        Args:
            raw_serp_results: Raw SERP data from database (could be dict, list, or None)
            
        Returns:
            Formatted AI response
        """
        # Handle different data types
        if raw_serp_results is None:
            self.logger.warning("raw_serp_results is None, creating empty response")
            return {
                'query': '',
                'source_links': []
            }
        
        if isinstance(raw_serp_results, list):
            self.logger.info(f"raw_serp_results is a list with {len(raw_serp_results)} items")
            # If it's a list, assume it's source_links directly
            return {
                'query': '',
                'source_links': raw_serp_results
            }
        
        if isinstance(raw_serp_results, dict):
            """Use the full raw SERP object as the ai_response.

            We only validate that it has non-empty source_links; otherwise we
            return it unchanged so downstream stages (especially Stage 2 DNA
            analysis) can see the complete context: query, ai_overview_text,
            structure_type, location, etc.
            """
            # Ensure source_links exists and is a list
            source_links = raw_serp_results.get('source_links') or []

            # If there are no links, this product should be skipped
            if not source_links:
                query = raw_serp_results.get('query', raw_serp_results.get('original_query', ''))
                self.logger.warning("source_links is empty, this product should be skipped")
                return {
                    'query': query,
                    'source_links': []
                }

            self.logger.info(
                f"Converted raw_serp_results: query='{raw_serp_results.get('query', raw_serp_results.get('original_query', ''))}', "
                f"source_links_count={len(source_links)}"
            )

            # Return the full raw object so Stage 2 gets full ai_response context
            return raw_serp_results
        
        # Fallback for unexpected types
        self.logger.warning(f"raw_serp_results has unexpected type: {type(raw_serp_results)}")
        return {
            'query': '',
            'source_links': []
        }
    
    def run_pipeline_from_ai_response(self, ai_response: Dict[str, Any], run_id_override: str = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from a pre-loaded AI response
        
        Args:
            ai_response: AI response with query and source_links
            run_id_override: Optional run ID override
            
        Returns:
            Final aggregation result
        """
        # Concurrency-safe, fully in-memory execution.
        # All per-request state lives in local variables inside this function.
        run_id = run_id_override or f"{self.config.get('pipeline', {}).get('run_id_prefix', 'gods_eye_run')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        query = ai_response.get('query', '')
        source_links = ai_response.get('source_links', [])

        run = PipelineRun(
            run_id=run_id,
            created_at=datetime.now(),
            query=query,
            total_links=len(source_links),
        )

        for i, link in enumerate(source_links):
            job_id = f"{run_id}_job_{i+1:03d}"
            job = Job(
                job_id=job_id,
                run_id=run_id,
                source_link=link,
                url=link.get('url', ''),
                text=link.get('text', ''),
                position=link.get('position', i + 1),
                max_retries=self.config.get('pipeline', {}).get('max_retries', 2),
            )
            run.jobs[job_id] = job

        # Stage 1
        stage1_queue = self.queue_manager.create_stage_1_queue(run)
        stage1_batches = self.queue_manager.create_batches_for_stage(stage1_queue)
        for batch in stage1_batches:
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    batch_jobs.append(run.jobs[job_ref])
            processed = self.stage_1_worker.process_batch(batch_jobs)
            for job in processed:
                run.jobs[job.job_id] = job

        self.queue_manager.filter_stage_1_results(run)

        # Stage 2
        stage2_queue = self.queue_manager.create_stage_2_queue(run)
        stage2_batches = self.queue_manager.create_batches_for_stage(stage2_queue)
        for batch in stage2_batches:
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    batch_jobs.append(run.jobs[job_ref])
            processed = self.stage_2_worker.process_batch(batch_jobs, ai_response)
            for job in processed:
                run.jobs[job.job_id] = job

        self.queue_manager.filter_stage_2_results(run)

        # Stage 3
        stage_3_jobs = []
        for job in run.jobs.values():
            if job.selected_for_stage_3 and job.stage_2_status == 'completed':
                stage_3_jobs.append(job)

        if not stage_3_jobs:
            return {
                'status': 'failed',
                'run_id': run_id,
                'error': 'No jobs selected for Stage 3',
                'total_analyzed': 0,
            }

        return self.stage_3_worker.process_run(run_id, query, stage_3_jobs)
    
    def create_run(self, ai_response: Dict[str, Any], run_id_override: str = None) -> str:
        """Create a new pipeline run"""
        # Use provided run_id_override or generate new one
        if run_id_override:
            run_id = run_id_override
        else:
            run_id = f"{self.config.get('pipeline', {}).get('run_id_prefix', 'gods_eye_run')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Extract query and source links
        query = ai_response.get('query', '')
        source_links = ai_response.get('source_links', [])
        
        # Create pipeline run
        run = PipelineRun(
            run_id=run_id,
            created_at=datetime.now(),
            query=query,
            total_links=len(source_links)
        )
        
        # Create jobs for each source link
        for i, link in enumerate(source_links):
            job_id = f"{run_id}_job_{i+1:03d}"
            job = Job(
                job_id=job_id,
                run_id=run_id,
                source_link=link,
                url=link.get('url', ''),
                text=link.get('text', ''),
                position=link.get('position', i+1),
                max_retries=self.config.get('pipeline', {}).get('max_retries', 2)
            )
            run.jobs[job_id] = job
        
        # Calculate initial batches for Stage 1
        max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_1', 10)
        run.stage_1_batches = calculate_batches(len(source_links), max_parallel)
        
        self.active_runs[run_id] = run
        return run_id
    
    def _execute_stage_1(self, run_id: str):
        """Execute Stage 1: Website Classification"""
        run = self.active_runs[run_id]
        self.logger.info(f"Executing Stage 1 for run {run_id}")
        
        # Create Stage 1 queue
        stage1_queue = self.queue_manager.create_stage_1_queue(run)
        
        # Create batches for Stage 1
        batches = self.queue_manager.create_batches_for_stage(stage1_queue)
        
        # Process each batch
        stage_1_results = []
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing Stage 1 batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch.items)} jobs)")
            
            # Extract jobs for this batch
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    # Handle case where batch contains job_ids instead of job objects
                    batch_jobs.append(run.jobs[job_ref])
            
            # Process batch
            batch_result = self.stage_1_worker.process_batch(batch_jobs)
            stage_1_results.extend(batch_result)
            
            # Update run with processed jobs
            for job in batch_result:
                run.jobs[job.job_id] = job
        
        run.stage_1_results = stage_1_results
        self.logger.info(f"Stage 1 completed for run {run_id}")
        
        # Filter Stage 1 results to determine which jobs proceed to Stage 2
        self._filter_stage_1_results(run_id)
    
    def _filter_stage_1_results(self, run_id: str):
        """
        Filter Stage 1 results to determine which jobs proceed to Stage 2.
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Filtering Stage 1 results for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Use queue manager to filter results
        selected_count, rejected_count = self.queue_manager.filter_stage_1_results(run)
        
        # Calculate Stage 2 batches for selected jobs
        if selected_count > 0:
            max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_2', 10)
            run.stage_2_batches = calculate_batches(selected_count, max_parallel)
            self.logger.info(f"Calculated {len(run.stage_2_batches)} Stage 2 batches for {selected_count} selected jobs")
    
    def _execute_stage_2(self, run_id: str) -> List[Dict[str, Any]]:
        """Execute Stage 2: DNA Analysis"""
        run = self.active_runs[run_id]
        self.logger.info(f"Executing Stage 2 for run {run_id}")
        
        # Get AI response for this run
        ai_response = self._ai_responses.get(run_id, {})
        
        # Create Stage 2 queue and batches
        stage2_queue = self.queue_manager.create_stage_2_queue(run)
        batches = self.queue_manager.create_batches_for_stage(stage2_queue)
        
        # Process each batch
        stage_2_results = []
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing Stage 2 batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch.items)} jobs)")
            
            # Extract jobs for this batch
            batch_jobs = []
            for job_ref in batch.items:
                if hasattr(job_ref, 'job_id'):
                    batch_jobs.append(job_ref)
                else:
                    # Handle case where batch contains job_ids instead of job objects
                    batch_jobs.append(run.jobs[job_ref])
            
            # Process batch
            batch_result = self.stage_2_worker.process_batch(batch_jobs, ai_response)
            stage_2_results.extend(batch_result)
            
            # Update run with processed jobs
            for job in batch_result:
                run.jobs[job.job_id] = job
        
        run.stage_2_results = stage_2_results
        self.logger.info(f"Stage 2 completed for run {run_id}")
        
        # Filter Stage 2 results to determine which jobs proceed to Stage 3
        self._filter_stage_2_results(run_id)
        
        return stage_2_results
    
    def _filter_stage_2_results(self, run_id: str):
        """
        Filter Stage 2 results to determine which jobs proceed to Stage 3.
        
        Args:
            run_id: Pipeline run ID
        """
        self.logger.info(f"Filtering Stage 2 results for run {run_id}")
        
        run = self.active_runs[run_id]
        
        # Use queue manager to filter results
        selected_count, rejected_count = self.queue_manager.filter_stage_2_results(run)
        
        # Calculate Stage 3 batches for selected jobs
        if selected_count > 0:
            max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_3', 5)
            run.stage_3_batches = calculate_batches(selected_count, max_parallel)
            self.logger.info(f"Calculated {len(run.stage_3_batches)} Stage 3 batches for {selected_count} selected jobs")
    
    def _execute_stage_3(self, run_id: str) -> Dict[str, Any]:
        """Execute Stage 3: Final Aggregation"""
        run = self.active_runs[run_id]
        self.logger.info(f"Executing Stage 3 for run {run_id}")
        
        # Get jobs selected for Stage 3
        stage_3_jobs = []
        for job in run.jobs.values():
            if job.selected_for_stage_3 and job.stage_2_status == 'completed':
                stage_3_jobs.append(job)
        
        if not stage_3_jobs:
            self.logger.info(f"No jobs selected for Stage 3 in run {run_id}")
            return {}
        
        self.logger.info(f"Processing {len(stage_3_jobs)} jobs for final aggregation")
        
        # Get AI response for this run
        ai_response = self._ai_responses.get(run_id, {})
        query = ai_response.get('query', '')
        
        # Process final aggregation using stage 3 worker
        try:
            # Stage3Worker expects (run_id, query, jobs)
            final_result = self.stage_3_worker.process_run(run_id, query, stage_3_jobs)
            self.logger.info(f"Stage 3 completed for run {run_id}")
            return final_result
        except Exception as e:
            self.logger.error(f"Run {run_id} failed Stage 3: {str(e)}")
            raise
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        db_stats = self.db_manager.get_processing_statistics()
        
        pipeline_stats = {
            "active_runs": len(self.active_runs),
            "completed_runs": len([r for r in self.active_runs.values() if r.completed_at]),
            "database_stats": db_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return pipeline_stats
