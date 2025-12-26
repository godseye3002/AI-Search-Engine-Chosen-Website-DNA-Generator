"""
Supabase Data Access Layer for DNA Pipeline

Handles dual-source data workflow:
- Google Flow: product_analysis_google -> product_analysis_dna_google
- Perplexity Flow: product_analysis_perplexity -> product_analysis_dna_perplexity
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from supabase import create_client, Client
from dotenv import load_dotenv
from utils.env_utils import is_production_mode

load_dotenv()

# Configure logging
log_level = logging.INFO if not is_production_mode() else logging.ERROR
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [DATABASE] - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source enumeration"""
    GOOGLE = "google"
    PERPLEXITY = "perplexity"


@dataclass
class ProductAnalysisRecord:
    """Represents a product analysis record from input tables"""
    id: int
    product_id: str
    raw_serp_results: Dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class DNAAnalysisRecord:
    """Represents a DNA analysis record for output tables"""
    product_id: str
    run_id: str
    dna_blueprint: Dict[str, Any]
    status: str = "completed"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SupabaseDataManager:
    """Manages data operations with Supabase for the DNA Pipeline"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Table mappings
        self.input_tables = {
            DataSource.GOOGLE: "product_analysis_google",
            DataSource.PERPLEXITY: "product_analysis_perplexity"
        }
        
        self.output_tables = {
            DataSource.GOOGLE: "product_analysis_dna_google",
            DataSource.PERPLEXITY: "product_analysis_dna_perplexity"
        }
    
    def fetch_pending_products(self, data_source: DataSource, limit: Optional[int] = None) -> List[ProductAnalysisRecord]:
        """
        Fetch products that need DNA analysis from the specified source
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            limit: Maximum number of records to fetch
            
        Returns:
            List of ProductAnalysisRecord objects
        """
        table_name = self.input_tables[data_source]
        logger.info(f"Fetching pending products from {table_name}")
        
        try:
            query = self.client.table(table_name).select("id, product_id, raw_serp_results, created_at, updated_at")
            
            # Add limit if specified
            if limit:
                query = query.limit(limit)
ently(limit)
YC
            
            # Order by created_at to process oldest first
            query = query.order("created_at", desc=False)
            
            response = query.execute()
            
            if response.data:
                records = []
                for item in response.data:
                    record = ProductAnalysisRecord(
                        id=item['id'],
                        product_id=item['product_id'],
                        raw_serp_results=item['raw_serp_results'] if isinstance(item['raw_serp_results'], dict) else json.loads(item['raw_serp_results']),
                        created_at=item.get('created_at'),
                        updated_at=item.get('updated_at')
                    )
                    records.append(record)
                
                logger.info(f"Fetched {len(records)} products from {table_name}")
                return records
            else:
                logger.info(f"No pending products found in {table_name}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching products from {table_name}: {str(e)}")
            raise
    
    def fetch_product_by_id(self, data_source: DataSource, product_id: str) -> Optional[ProductAnalysisRecord]:
        """
        Fetch a specific product by ID
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            product_id: Product identifier
            
        Returns:
            ProductAnalysisRecord or None if not found
        """
        table_name = self.input_tables[data_source]
        logger.info(f"Fetching product {product_id} from {table_name}")
        
        try:
            response = self.client.table(table_name).select(
                "id, product_id, raw_serp_results, created_at, updated_at"
            ).eq("product_id", product_id).execute()
            
            if response.data and len(response.data) > 0:
                item = response.data[0]
                record = ProductAnalysisRecord(
                    id=item['id'],
                    product_id=item['product_id'],
                    raw_serp_results=item['raw_serp_results'] if isinstance(item['raw_serp_results'], dict) else json.loads(item['raw_serp_results']),
                    created_at=item.get('created_at'),
                    updated_at=item.get('updated_at')
                )
                logger.info(f"Found product {product_id} in {table_name}")
                return record
            else:
                logger.warning(f"Product {product_id} not found in {table_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching product {product_id} from {table_name}: {str(e)}")
            raise
    
    def save_dna_analysis(self, data_source: DataSource, record: DNAAnalysisRecord) -> DNAAnalysisRecord:
        """
        Save DNA analysis results to the appropriate output table
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            record: DNAAnalysisRecord to save
            
        Returns:
            Saved DNAAnalysisRecord with ID
        """
        table_name = self.output_tables[data_source]
        logger.info(f"Saving DNA analysis for product {record.product_id} to {table_name}")
        
        try:
            # Prepare data for insertion
            data = {
                "product_id": record.product_id,
                "run_id": record.run_id,
                "dna_blueprint": record.dna_blueprint,
                "status": record.status
            }
            
            response = self.client.table(table_name).insert(data).execute()
            
            if response.data and len(response.data) > 0:
                saved_item = response.data[0]
                saved_record = DNAAnalysisRecord(
                    id=saved_item['id'],
                    product_id=saved_item['product_id'],
                    run_id=saved_item['run_id'],
                    dna_blueprint=saved_item['dna_blueprint'],
                    status=saved_item['status'],
                    created_at=saved_item.get('created_at'),
                    updated_at=saved_item.get('updated_at')
                )
                logger.info(f"Saved DNA analysis for product {record.product_id} with ID {saved_record.id}")
                return saved_record
            else:
                raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving DNA analysis for product {record.product_id}: {str(e)}")
            raise
    
    def update_dna_analysis_status(self, data_source: DataSource, record_id: int, status: str) -> bool:
        """
        Update the status of a DNA analysis record
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            record_id: ID of the record to update
            status: New status value
            
        Returns:
            True if successful, False otherwise
        """
        table_name = self.output_tables[data_source]
        logger.info(f"Updating status to '{status}' for record {record_id} in {table_name}")
        
        try:
            response = self.client.table(table_name).update(
                {"status": status, "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", record_id).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Updated status for record {record_id} to '{status}'")
                return True
            else:
                logger.warning(f"No record found with ID {record_id} in {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating status for record {record_id}: {str(e)}")
            return False
    
    def check_existing_analysis(self, data_source: DataSource, product_id: str) -> Optional[DNAAnalysisRecord]:
        """
        Check if DNA analysis already exists for a product
        
        Args:
            data_source: GOOGLE or PERPLEXITY
            product_id: Product identifier
            
        Returns:
            Existing DNAAnalysisRecord or None
        """
        table_name = self.output_tables[data_source]
        logger.info(f"Checking existing analysis for product {product_id} in {table_name}")
        
        try:
            response = self.client.table(table_name).select(
                "id, product_id, run_id, dna_blueprint, status, created_at, updated_at"
            ).eq("product_id", product_id).execute()
            
            if response.data and len(response.data) > 0:
                item = response.data[0]
                record = DNAAnalysisRecord(
                    id=item['id'],
                    product_id=item['product_id'],
                    run_id=item['run_id'],
                    dna_blueprint=item['dna_blueprint'],
                    status=item['status'],
                    created_at=item.get('created_at'),
                    updated_at=item.get('updated_at')
                )
                logger.info(f"Found existing analysis for product {product_id} with status '{record.status}'")
                return record
            else:
                logger.info(f"No existing analysis found for product {product_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking existing analysis for product {product_id}: {str(e)}")
            return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics for monitoring
        
        Returns:
            Dictionary with statistics for both data sources
        """
        stats = {
            "google": {"input_count": 0, "output_count": 0, "pending_count": 0},
            "perplexity": {"input_count": 0, "output_count": 0, "pending_count": 0},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            for source in DataSource:
                source_name = source.value.lower()
                
                # Count input records
                input_response = self.client.table(self.input_tables[source]).select("id", count="exact").execute()
                stats[source_name]["input_count"] = input_response.count or 0
                
                # Count output records
                output_response = self.client.table(self.output_tables[source]).select("id", count="exact").execute()
                stats[source_name]["output_count"] = output_response.count or 0
                
                # Calculate pending
                stats[source_name]["pending_count"] = (
                    stats[source_name]["input_count"] - stats[source_name]["output_count"]
                )
            
            logger.info("Retrieved processing statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {str(e)}")
            return stats


# Global instance for application use
db_manager = SupabaseDataManager()
