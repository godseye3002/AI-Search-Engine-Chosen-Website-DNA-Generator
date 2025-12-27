# Database-Driven DNA Pipeline Usage Guide

## Overview

The DNA Pipeline now supports dual-source database workflow using Supabase. Instead of processing local files, the pipeline fetches data from Supabase tables and saves results back to the database.

## Database Schema

### Input Tables
- `product_analysis_google`: Contains raw SERP results from Google
- `product_analysis_perplexity`: Contains raw SERP results from Perplexity

### Output Tables  
- `product_analysis_dna_google`: Stores DNA analysis results for Google data
- `product_analysis_dna_perplexity`: Stores DNA analysis results for Perplexity data

## Quick Start

### 1. Test Database Connection
```bash
python test_database_connection.py
```

### 2. Process All Pending Products
```bash
# Process all sources (Google + Perplexity)
python run_database_pipeline.py

# Process only Google source
python run_database_pipeline.py --source google

# Process only Perplexity source  
python run_database_pipeline.py --source perplexity
```

### 3. Process Specific Product
```bash
python run_database_pipeline.py --source google --product-id "your-product-id"
```

### 4. Limit Processing
```bash
# Process maximum 5 products from each source
python run_database_pipeline.py --limit 5
```

### 5. View Statistics
```bash
python run_database_pipeline.py --stats
```

## Command Line Options

- `--source`: Choose data source (google, perplexity, all)
- `--product-id`: Process specific product ID
- `--limit`: Limit number of products to process
- `--parallel`: Enable parallel processing (default: true)
- `--stats`: Show processing statistics

## API Usage

### Direct Database Operations

```python
from database.supabase_manager import SupabaseDataManager, DataSource

# Initialize database manager
db_manager = SupabaseDataManager()

# Fetch pending products
google_products = db_manager.fetch_pending_products(DataSource.GOOGLE, limit=10)
perplexity_products = db_manager.fetch_pending_products(DataSource.PERPLEXITY, limit=10)

# Check existing analysis
existing = db_manager.check_existing_analysis(DataSource.GOOGLE, "product-id")

# Save DNA analysis results
from database.supabase_manager import DNAAnalysisRecord
dna_record = DNAAnalysisRecord(
    product_id="product-id",
    run_id="run-id", 
    dna_blueprint={"your": "dna-data"},
    status="completed"
)
saved = db_manager.save_dna_analysis(DataSource.GOOGLE, dna_record)
```

### Pipeline Processing

```python
from database.database_pipeline_orchestrator import DatabasePipelineOrchestrator, DataSource

# Initialize orchestrator
orchestrator = DatabasePipelineOrchestrator()

# Process single product
result = orchestrator.process_product_from_database(DataSource.GOOGLE, "product-id")

# Process batch of products
results = orchestrator.process_batch_from_database(DataSource.PERPLEXITY, limit=5)
```

## Data Flow

1. **Input**: Raw SERP results are fetched from `product_analysis_google` or `product_analysis_perplexity`
2. **Processing**: Data flows through the 3-stage DNA pipeline (Classification → DNA Analysis → Final Aggregation)
3. **Output**: Final DNA blueprints are saved to `product_analysis_dna_google` or `product_analysis_dna_perplexity`

## Error Handling

The pipeline includes comprehensive error handling:
- Database connection errors are logged and retried
- Failed products are marked with status "failed" in the database
- Processing continues with other products if one fails
- Detailed logs are written for debugging

## Monitoring

Use the statistics command to monitor pipeline progress:
```bash
python run_database_pipeline.py --stats
```

This shows:
- Number of input records per source
- Number of completed DNA analyses
- Number of pending records to process

## Configuration

Ensure your `.env` file contains:
```
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
GEMINI_API_KEY=your-gemini-api-key
```

## Notes

- The pipeline automatically skips products that have already been processed
- Each run generates a unique `run_id` for tracking
- Results are saved as JSON in the `dna_blueprint` field
- The pipeline supports parallel processing for better performance
