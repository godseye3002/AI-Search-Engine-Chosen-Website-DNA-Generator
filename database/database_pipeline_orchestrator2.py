"""
Database Pipeline Orchestrator
Coordinates the DNA analysis pipeline using database-driven approach with data freshness checks.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime

from .supabase_manager import SupabaseDataManager, DataSource
from stage_1_worker import Stage1Worker
from stage_2_worker import Stage2Worker  
from stage_3_worker import Stage3Worker

logger = logging.getLogger(__name__)

class DatabasePipelineOrchestrator:
    """Orchestrates the DNA analysis pipeline with database integration"""
    
    def __init__(self):
        """Initialize orchestrator with database manager and workers"""
        self.db_manager = SupabaseDataManager()
        
        # Default configuration for workers
        config = {
            'stage_1_classification': {
                'timeout_per_link': 30,
                'output_dir': 'stage_1_results'
            },
            'stage_2_dna_analysis': {
                'timeout_per_chunk': 60,
                'output_dir': 'stage_2_results'
            },
            'stage_3_final_aggregation': {
                'timeout': 45,
                'output_dir': 'stage_3_results'
            },
            'pipeline': {
                'base_output_dir': 'outputs'
            }
        }
        
        self.stage1_worker = Stage1Worker(config)
        self.stage2_worker = Stage2Worker(config)
        self.stage3_worker = Stage3Worker(config)
        
        logger.info("Database Pipeline Orchestrator initialized")
    
    def _generate_input_hash(self, input_rows: List[Dict[str, Any]]) -> str:
        """
        Generate hash from input row IDs for data freshness checking.
        
        Args:
            input_rows: List of input row dictionaries
            
        Returns:
            SHA-256 hash of sorted input row IDs
        """
        try:
            # Extract IDs and sort them for consistent hashing
            ids = [str(row.get('id', '')) for row in input_rows if row.get('id')]
            sorted_ids = sorted(ids)
            combined_ids = ",".join(sorted_ids)
            
            # Generate SHA-256 hash
            hash_object = hashlib.sha256(combined_ids.encode('utf-8'))
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.exception("Failed to generate input hash")
            raise
    
    def process_product_from_database(self, data_source: DataSource, product_id: str) -> Dict[str, Any]:
        """
        Process a single product from database with data freshness checking.
        
        Args:
            data_source: Data source (GOOGLE or PERPLEXITY)
            product_id: Product ID to process
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"Starting processing for product {product_id} from {data_source.value}")
            
            # Step 1: Fetch input rows
            input_rows = self.db_manager.fetch_input_rows(data_source, product_id)
            if not input_rows:
                return {
                    "status": "failed",
                    "product_id": product_id,
                    "data_source": data_source.value,
                    "error": "No input data found"
                }
            
            # Step 2: Generate input hash for freshness checking
            input_hash = self._generate_input_hash(input_rows)
            logger.info(f"Generated input hash: {input_hash[:16]}...")
            
            # Step 3: Check if analysis already exists and is up-to-date
            should_skip, existing_record = self.db_manager.check_existing_analysis(
                data_source, product_id, input_hash
            )
            
            if should_skip:
                return {
                    "status": "skipped",
                    "product_id": product_id,
                    "data_source": data_source.value,
                    "message": "Analysis already exists and is up-to-date",
                    "analysis_id": existing_record.get('id'),
                    "run_id": existing_record.get('run_id')
                }
            
            # Step 4: Combine all input rows for processing (many-to-one relationship)
            combined_input = self._combine_input_rows(input_rows, data_source)
            
            # Step 5: Run the pipeline stages
            pipeline_result = self._run_pipeline_stages(combined_input, data_source)
            
            if pipeline_result["status"] != "completed":
                return pipeline_result
            
            # Step 6: Save results with input hash
            analysis_data = {
                "product_id": product_id,
                "data_source": data_source.value,
                "run_id": pipeline_result["run_id"],
                "input_data_hash": input_hash,
                "dna_blueprint": pipeline_result.get("dna_blueprint"),
                "processing_metadata": {
                    "input_rows_count": len(input_rows),
                    "processing_time": pipeline_result.get("processing_time"),
                    "stage_results": pipeline_result.get("stage_results", {})
                }
            }
            
            saved_record = self.db_manager.save_dna_analysis(data_source, analysis_data)
            
            return {
                "status": "completed",
                "product_id": product_id,
                "data_source": data_source.value,
                "run_id": pipeline_result["run_id"],
                "analysis_id": saved_record["id"],
                "input_data_hash": input_hash,
                "dna_blueprint": pipeline_result.get("dna_blueprint"),
                "processing_time": pipeline_result.get("processing_time")
            }
            
        except Exception as e:
            logger.exception(f"Failed to process product {product_id}")
            return {
                "status": "failed",
                "product_id": product_id,
                "data_source": data_source.value,
                "error": str(e)
            }
    
    def _combine_input_rows(self, input_rows: List[Dict[str, Any]], data_source: DataSource) -> Dict[str, Any]:
        """
        Combine multiple input rows into a single input for processing.
        
        Args:
            input_rows: List of input rows
            data_source: Data source
            
        Returns:
            Combined input dictionary
        """
        try:
            if not input_rows:
                raise ValueError("No input rows to combine")
            
            # Use the first row as base and merge others
            combined = input_rows[0].copy()
            
            # Combine source links from all rows
            all_source_links = []
            for row in input_rows:
                raw_serp = row.get('raw_serp_results', {})
                if isinstance(raw_serp, str):
                    try:
                        raw_serp = json.loads(raw_serp)
                    except json.JSONDecodeError:
                        raw_serp = {}
                
                if isinstance(raw_serp, dict):
                    source_links = raw_serp.get('source_links', [])
                    if isinstance(source_links, list):
                        all_source_links.extend(source_links)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_links = []
            for link in all_source_links:
                if isinstance(link, dict):
                    key = link.get('raw_url') or link.get('url') or json.dumps(link, sort_keys=True)
                else:
                    key = str(link)
                
                if key not in seen:
                    seen.add(key)
                    unique_links.append(link)
            
            # Update combined input with unique links
            if 'raw_serp_results' not in combined:
                combined['raw_serp_results'] = {}
            
            if isinstance(combined['raw_serp_results'], str):
                try:
                    combined['raw_serp_results'] = json.loads(combined['raw_serp_results'])
                except json.JSONDecodeError:
                    combined['raw_serp_results'] = {}
            
            combined['raw_serp_results']['source_links'] = unique_links
            combined['combined_source_links_count'] = len(unique_links)
            combined['original_input_rows_count'] = len(input_rows)
            
            logger.info(f"Combined {len(input_rows)} input rows with {len(unique_links)} unique source links")
            return combined
            
        except Exception as e:
            logger.exception("Failed to combine input rows")
            raise
    
    def _run_pipeline_stages(self, input_data: Dict[str, Any], data_source: DataSource) -> Dict[str, Any]:
        """
        Run all pipeline stages on the input data.
        
        Args:
            input_data: Combined input data
            data_source: Data source
            
        Returns:
            Pipeline result dictionary
        """
        try:
            start_time = datetime.now()
            run_id = f"dna_{data_source.value}_{input_data.get('product_id', 'unknown')}_{int(start_time.timestamp())}"
            
            logger.info(f"Starting pipeline stages for run {run_id}")
            
            stage_results = {}
            
            # Stage 1: Classification
            logger.info("Running Stage 1: Classification")
            stage1_result = self.stage1_worker.process(input_data)
            stage_results["stage1"] = stage1_result
            
            if stage1_result.get("status") != "completed":
                return {
                    "status": "failed",
                    "run_id": run_id,
                    "error": f"Stage 1 failed: {stage1_result.get('error', 'Unknown error')}",
                    "stage_results": stage_results
                }
            
            # Stage 2: DNA Analysis
            logger.info("Running Stage 2: DNA Analysis")
            stage2_result = self.stage2_worker.process(input_data, stage1_result)
            stage_results["stage2"] = stage2_result
            
            if stage2_result.get("status") != "completed":
                return {
                    "status": "failed", 
                    "run_id": run_id,
                    "error": f"Stage 2 failed: {stage2_result.get('error', 'Unknown error')}",
                    "stage_results": stage_results
                }
            
            # Stage 3: Final Aggregation
            logger.info("Running Stage 3: Final Aggregation")
            stage3_result = self.stage3_worker.process(input_data, stage1_result, stage2_result)
            stage_results["stage3"] = stage3_result
            
            if stage3_result.get("status") != "completed":
                return {
                    "status": "failed",
                    "run_id": run_id, 
                    "error": f"Stage 3 failed: {stage3_result.get('error', 'Unknown error')}",
                    "stage_results": stage_results
                }
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Pipeline completed successfully for run {run_id} in {processing_time:.2f}s")
            
            return {
                "status": "completed",
                "run_id": run_id,
                "processing_time": processing_time,
                "dna_blueprint": stage3_result.get("dna_blueprint"),
                "stage_results": stage_results
            }
            
        except Exception as e:
            logger.exception("Pipeline execution failed")
            return {
                "status": "failed",
                "error": str(e),
                "stage_results": stage_results if 'stage_results' in locals() else {}
            }
    
    def process_batch_from_database(self, data_source: DataSource, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple products from database.
        
        Args:
            data_source: Data source to process
            limit: Maximum number of products to process
            
        Returns:
            List of processing results
        """
        try:
            logger.info(f"Starting batch processing for {data_source.value}")
            
            # Get pending products
            pending_products = self.db_manager.fetch_pending_products(data_source, limit)
            
            if not pending_products:
                logger.info(f"No pending products found for {data_source.value}")
                return []
            
            results = []
            
            for product_data in pending_products:
                product_id = product_data["product_id"]
                result = self.process_product_from_database(data_source, product_id)
                results.append(result)
            
            completed = sum(1 for r in results if r["status"] == "completed")
            failed = sum(1 for r in results if r["status"] == "failed")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            
            logger.info(f"Batch processing complete for {data_source.value}: {completed} completed, {failed} failed, {skipped} skipped")
            
            return results
            
        except Exception as e:
            logger.exception(f"Batch processing failed for {data_source.value}")
            return []
