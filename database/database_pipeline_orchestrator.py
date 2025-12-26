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
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from utils.batch_calculator import calculate_batches, create_batches_with_items, BatchInfo
from utils.timeout_handler import execute_with_timeout, ExecutionResult, TimeoutResult
from pipeline_models import Job, PipelineRun
from state_manager import StateManager
from job_queue_manager import JobQueueManager
from stage_1_worker import Stage1Worker
from stage_2_worker import Stage2Worker
from stage_3_worker import Stage3Worker
from database.supabase_manager import SupabaseDataManager, DataSource, ProductAnalysisRecord, DNAAnalysisRecord


class DatabasePipelineOrchestrator:
    """Database-driven pipeline orchestrator for Supabase integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self.state_manager = StateManager(self.config, self.logger)
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
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
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
            # Check if analysis already exists
            existing_analysis = self.db_manager.check_existing_analysis(data_source, product_id)
            if existing_analysis and existing_analysis.status == "completed":
                self.logger.info(f"Product {product_id} already has completed analysis")
                return {
                    "status": "already_exists",
                    "product_id": product_id,
                    "data_source": data_source.value,
                    "existing_analysis_id": existing_analysis.id,
                    "message": "Analysis already completed"
                }
            
            # Fetch product data from database
            product_record = self.db_manager.fetch_product_by_id(data_source, product_id)
            if not product_record:
                raise ValueError(f"Product {product_id} not found in {data_source.value} table")
            
            # Convert raw_serp_results to AI response format
            ai_response = self._convert_to_ai_response(product_record.raw_serp_results)
            
            # Generate stable run_id
            run_id = f"dna_{data_source.value}_{product_id}_{int(time.time())}"
            
            # Run pipeline
            final_result = self.run_pipeline_from_ai_response(ai_response, run_id_override=run_id)
            
            # Save results to database
            dna_record = DNAAnalysisRecord(
                product_id=product_id,
                run_id=run_id,
                dna_blueprint=final_result,
                status="completed"
            )
            
            saved_record = self.db_manager.save_dna_analysis(data_source, dna_record)
            
            self.logger.info(f"Successfully processed product {product_id} with analysis ID {saved_record.id}")
            
            return {
                "status": "completed",
                "product_id": product_id,
                "data_source": data_source.value,
                "run_id": run_id,
                "analysis_id": saved_record.id,
                "final_output_path": f"outputs/stage_3_results/run_{run_id}/final_aggregation.json"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing product {product_id}: {str(e)}")
            
            # Update status if we have a record
            try:
                if 'dna_record' in locals():
                    self.db_manager.update_dna_analysis_status(data_source, dna_record.id, "failed")
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
    
    def _convert_to_ai_response(self, raw_serp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw SERP results to AI response format
        
        Args:
            raw_serp_results: Raw SERP data from database
            
        Returns:
            Formatted AI response
        """
        # Extract query and source links from raw_serp_results
        query = raw_serp_results.get('query', '')
        source_links = raw_serp_results.get('source_links', [])
        
        return {
            'query': query,
            'source_links': source_links
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
        # Create run with override
        run_id = self.create_run(ai_response, run_id_override)
        
        # Store AI response for this run
        if hasattr(self, '_ai_responses'):
            self._ai_responses[run_id] = ai_response
        else:
            self._ai_responses = {run_id: ai_response}
        
        # Execute all stages
        self._execute_stage_1(run_id)
        stage_1_results = self._execute_stage_2(run_id)
        final_result = self._execute_stage_3(run_id, stage_1_results)
        
        return final_result
    
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
        
        # Process batches in parallel
        stage_1_results = []
        for batch_info in run.stage_1_batches:
            batch_jobs = [run.jobs[job_id] for job_id in batch_info.job_ids]
            batch_result = self.stage_1_worker.process_batch(batch_jobs, run_id)
            stage_1_results.extend(batch_result.get('results', []))
        
        run.stage_1_results = stage_1_results
        self.logger.info(f"Stage 1 completed for run {run_id}")
    
    def _execute_stage_2(self, run_id: str) -> List[Dict[str, Any]]:
        """Execute Stage 2: DNA Analysis"""
        run = self.active_runs[run_id]
        self.logger.info(f"Executing Stage 2 for run {run_id}")
        
        # Calculate Stage 2 batches
        max_parallel = self.config.get('pipeline', {}).get('max_parallel_workers_stage_2', 5)
        run.stage_2_batches = calculate_batches(len(run.stage_1_results), max_parallel)
        
        # Process batches in parallel
        stage_2_results = []
        for batch_info in run.stage_2_batches:
            batch_results = [run.stage_1_results[i] for i in batch_info.job_ids]
            batch_result = self.stage_2_worker.process_batch(batch_results, run_id)
            stage_2_results.extend(batch_result.get('results', []))
        
        run.stage_2_results = stage_2_results
        self.logger.info(f"Stage 2 completed for run {run_id}")
        
        return stage_2_results
    
    def _execute_stage_3(self, run_id: str, stage_2_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Stage 3: Final Aggregation"""
        run = self.active_runs[run_id]
        self.logger.info(f"Executing Stage 3 for run {run_id}")
        
        # Get AI response for this run
        ai_response = self._ai_responses.get(run_id, {})
        query = ai_response.get('query', '')
        
        # Process final aggregation
        final_result = self.stage_3_worker.process_run(run_id, query, run.jobs)
        
        run.stage_3_result = final_result
        run.completed_at = datetime.now()
        
        self.logger.info(f"Stage 3 completed for run {run_id}")
        return final_result.get('result', {})
    
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
