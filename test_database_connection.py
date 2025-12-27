"""
Database Connection Test Script

Tests Supabase connectivity and basic operations for the DNA Pipeline.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add database module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))

from database.supabase_manager import SupabaseDataManager, DataSource


def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [TEST] - %(message)s'
    )
    return logging.getLogger(__name__)


def test_database_connection():
    """Test basic database connection"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing Supabase connection...")
        
        # Initialize database manager
        db_manager = SupabaseDataManager()
        logger.info("✓ Supabase client initialized successfully")
        
        # Test connection by getting statistics
        stats = db_manager.get_processing_statistics()
        logger.info("✓ Database connection successful")
        
        return True, db_manager
        
    except Exception as e:
        logger.error(f"✗ Database connection failed: {str(e)}")
        return False, None


def test_table_access(db_manager):
    """Test access to required tables"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing table access...")
        
        # Test Google input table
        google_stats = db_manager.client.table("product_analysis_google").select("id", count="exact").execute()
        logger.info(f"✓ Google input table accessible: {google_stats.count} records")
        
        # Test Perplexity input table
        perplexity_stats = db_manager.client.table("product_analysis_perplexity").select("id", count="exact").execute()
        logger.info(f"✓ Perplexity input table accessible: {perplexity_stats.count} records")
        
        # Test Google output table
        google_dna_stats = db_manager.client.table("product_analysis_dna_google").select("id", count="exact").execute()
        logger.info(f"✓ Google DNA output table accessible: {google_dna_stats.count} records")
        
        # Test Perplexity output table
        perplexity_dna_stats = db_manager.client.table("product_analysis_dna_perplexity").select("id", count="exact").execute()
        logger.info(f"✓ Perplexity DNA output table accessible: {perplexity_dna_stats.count} records")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Table access failed: {str(e)}")
        return False


def test_data_fetching(db_manager):
    """Test data fetching functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing data fetching...")
        
        # Test fetching from Google source
        google_products = db_manager.fetch_pending_products(DataSource.GOOGLE, limit=1)
        if google_products:
            product = google_products[0]
            logger.info(f"✓ Successfully fetched Google product: {product.product_id}")
            logger.info(f"  - Raw SERP results type: {type(product.raw_serp_results)}")
            if isinstance(product.raw_serp_results, list):
                logger.info(f"  - Raw SERP results length: {len(product.raw_serp_results)}")
            elif isinstance(product.raw_serp_results, dict):
                logger.info(f"  - Raw SERP results keys: {list(product.raw_serp_results.keys())}")
        else:
            logger.info("ℹ No pending Google products found")
        
        # Test fetching from Perplexity source
        perplexity_products = db_manager.fetch_pending_products(DataSource.PERPLEXITY, limit=1)
        if perplexity_products:
            product = perplexity_products[0]
            logger.info(f"✓ Successfully fetched Perplexity product: {product.product_id}")
            logger.info(f"  - Raw SERP results type: {type(product.raw_serp_results)}")
            if isinstance(product.raw_serp_results, list):
                logger.info(f"  - Raw SERP results length: {len(product.raw_serp_results)}")
            elif isinstance(product.raw_serp_results, dict):
                logger.info(f"  - Raw SERP results keys: {list(product.raw_serp_results.keys())}")
        else:
            logger.info("ℹ No pending Perplexity products found")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data fetching failed: {str(e)}")
        return False


def test_sample_product_processing():
    """Test processing a sample product"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing sample product processing...")
        
        # Import the database orchestrator
        from database.database_pipeline_orchestrator import DatabasePipelineOrchestrator
        
        # Initialize orchestrator
        orchestrator = DatabasePipelineOrchestrator()
        logger.info("✓ Database orchestrator initialized")
        
        # Find a sample product to process
        db_manager = orchestrator.db_manager
        
        # Try Google first
        google_products = db_manager.fetch_pending_products(DataSource.GOOGLE, limit=1)
        if google_products:
            sample_product = google_products[0]
            logger.info(f"Testing with Google product: {sample_product.product_id}")
            
            # Check if already processed
            existing = db_manager.check_existing_analysis(DataSource.GOOGLE, sample_product.product_id)
            if existing:
                logger.info(f"ℹ Product already processed with status: {existing.status}")
                return True
            
            # Process the product (but don't actually run full pipeline for testing)
            logger.info("✓ Sample product setup successful")
            logger.info("ℹ Skipping full pipeline execution for test")
            return True
        
        # Try Perplexity if no Google products
        perplexity_products = db_manager.fetch_pending_products(DataSource.PERPLEXITY, limit=1)
        if perplexity_products:
            sample_product = perplexity_products[0]
            logger.info(f"Testing with Perplexity product: {sample_product.product_id}")
            
            # Check if already processed
            existing = db_manager.check_existing_analysis(DataSource.PERPLEXITY, sample_product.product_id)
            if existing:
                logger.info(f"ℹ Product already processed with status: {existing.status}")
                return True
            
            logger.info("✓ Sample product setup successful")
            logger.info("ℹ Skipping full pipeline execution for test")
            return True
        
        logger.info("ℹ No sample products found for testing")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sample product processing test failed: {str(e)}")
        return False


def display_full_statistics(db_manager):
    """Display comprehensive database statistics"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Getting comprehensive statistics...")
        
        stats = db_manager.get_processing_statistics()
        
        print("\n" + "="*60)
        print("DNA PIPELINE DATABASE STATISTICS")
        print("="*60)
        
        for source in ["google", "perplexity"]:
            source_stats = stats[source]
            print(f"\n{source.upper()} Source:")
            print(f"  Input Records: {source_stats['input_count']}")
            print(f"  Output Records: {source_stats['output_count']}")
            print(f"  Pending Records: {source_stats['pending_count']}")
        
        print(f"\nLast Updated: {stats['timestamp']}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Statistics display failed: {str(e)}")
        return False


def main():
    """Main test function"""
    logger = setup_logging()
    
    print("DNA Pipeline Database Connection Test")
    print("="*50)
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    connected, db_manager = test_database_connection()
    if not connected:
        print("❌ Database connection failed. Please check your Supabase credentials.")
        return False
    
    # Test 2: Table access
    print("\n2. Testing table access...")
    if not test_table_access(db_manager):
        print("❌ Table access failed. Please check table permissions.")
        return False
    
    # Test 3: Data fetching
    print("\n3. Testing data fetching...")
    if not test_data_fetching(db_manager):
        print("❌ Data fetching failed.")
        return False
    
    # Test 4: Sample product processing
    print("\n4. Testing sample product processing...")
    if not test_sample_product_processing():
        print("❌ Sample product processing failed.")
        return False
    
    # Test 5: Display statistics
    print("\n5. Displaying database statistics...")
    if not display_full_statistics(db_manager):
        print("❌ Statistics display failed.")
        return False
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("Database integration is ready for use.")
    print("="*50)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        sys.exit(1)
