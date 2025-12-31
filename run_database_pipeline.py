"""
Database-Driven DNA Pipeline Processor

Replaces file-based processing with Supabase database integration.
Supports dual-source workflow for Google and Perplexity data.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add database module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))

from database.supabase_manager import SupabaseDataManager, DataSource
from database.database_pipeline_orchestrator import DatabasePipelineOrchestrator


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [MAIN] - %(message)s'
    )
    return logging.getLogger(__name__)


def process_single_product(data_source: str, product_id: str) -> Dict[str, Any]:
    """Process a single product from database"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate data source
        source_enum = DataSource(data_source.lower())
        
        # Initialize orchestrator
        orchestrator = DatabasePipelineOrchestrator()
        
        # Process product
        result = orchestrator.process_product_from_database(source_enum, product_id)
        
        logger.info(f"Completed processing for product {product_id}: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process product {product_id}: {str(e)}")
        return {
            "status": "failed",
            "product_id": product_id,
            "data_source": data_source,
            "error": str(e)
        }


def process_batch_products(data_source: str, limit: Optional[int] = None, parallel: bool = True) -> List[Dict[str, Any]]:
    """Process multiple products from database"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate data source
        source_enum = DataSource(data_source.lower())
        
        # Initialize orchestrator
        orchestrator = DatabasePipelineOrchestrator()
        
        if parallel:
            # Process products in parallel
            products = orchestrator.db_manager.fetch_pending_products(source_enum, limit)
            if not products:
                logger.info(f"No pending products found for {data_source}")
                return []
            
            logger.info(f"Processing {len(products)} products in parallel")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        orchestrator.process_product_from_database, 
                        source_enum, 
                        product.product_id
                    ): product.product_id 
                    for product in products
                }
                
                results = []
                for future in as_completed(futures):
                    product_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process product {product_id}: {str(e)}")
                        results.append({
                            "status": "failed",
                            "product_id": product_id,
                            "data_source": data_source,
                            "error": str(e)
                        })
            
            return results
        else:
            # Process sequentially
            return orchestrator.process_batch_from_database(source_enum, limit)
            
    except Exception as e:
        logger.error(f"Failed to process batch for {data_source}: {str(e)}")
        return []


def process_all_sources(limit_per_source: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Process products from both Google and Perplexity sources"""
    logger = logging.getLogger(__name__)
    
    results = {
        "google": [],
        "perplexity": []
    }
    
    # Process both sources in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_batch_products, "google", limit_per_source): "google",
            executor.submit(process_batch_products, "perplexity", limit_per_source): "perplexity"
        }
        
        for future in as_completed(futures):
            source = futures[future]
            try:
                source_results = future.result()
                results[source] = source_results
                logger.info(f"Completed processing for {source}: {len(source_results)} products")
            except Exception as e:
                logger.error(f"Failed to process {source}: {str(e)}")
                results[source] = []
    
    return results


def show_statistics():
    """Display processing statistics"""
    logger = logging.getLogger(__name__)
    
    try:
        db_manager = SupabaseDataManager()
        stats = db_manager.get_processing_statistics()
        
        print("\n" + "="*60)
        print("DNA PIPELINE PROCESSING STATISTICS")
        print("="*60)
        
        for source in ["google", "perplexity"]:
            source_stats = stats[source]
            print(f"\n{source.upper()} Source:")
            print(f"  Input Records: {source_stats['input_count']}")
            print(f"  Output Records: {source_stats['output_count']}")
            print(f"  Pending Records: {source_stats['pending_count']}")
        
        print(f"\nLast Updated: {stats['timestamp']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Database-Driven DNA Pipeline Processor")
    parser.add_argument("--source", choices=["google", "perplexity", "all"], default="all",
                       help="Data source to process")
    parser.add_argument("--product-id", type=str, help="Process specific product ID")
    parser.add_argument("--limit", type=int, help="Limit number of products to process")
    parser.add_argument("--parallel", action="store_true", default=True, help="Process in parallel")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        if args.stats:
            show_statistics()
            return
        
        if args.product_id:
            # Process single product
            if not args.source or args.source == "all":
                logger.error("Must specify --source when using --product-id")
                return
            
            result = process_single_product(args.source, args.product_id)
            print(json.dumps(result, indent=2))
            
        elif args.source == "all":
            # Process all sources
            results = process_all_sources(args.limit)
            
            # Print summary
            total_processed = sum(len(results[source]) for source in results)
            completed = sum(
                sum(1 for r in results[source] if r["status"] == "completed") 
                for source in results
            )
            failed = sum(
                sum(1 for r in results[source] if r["status"] == "failed") 
                for source in results
            )
            
            logger.info(f"Processing complete: {completed} completed, {failed} failed, {total_processed} total")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"database_processing_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
        else:
            # Process single source
            results = process_batch_products(args.source, args.limit, args.parallel)
            
            completed = sum(1 for r in results if r["status"] == "completed")
            failed = sum(1 for r in results if r["status"] == "failed")
            
            logger.info(f"Processing complete for {args.source}: {completed} completed, {failed} failed")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"database_processing_{args.source}_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
