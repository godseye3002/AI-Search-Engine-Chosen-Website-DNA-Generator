"""
FastAPI Server for GodsEye DNA Pipeline
Provides REST endpoints to trigger pipeline runs via serverless deployment
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from database.database_pipeline_orchestrator import DatabasePipelineOrchestrator
from database.supabase_manager import DataSource
from utils.env_utils import get_log_level

os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=get_log_level(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request models
class ProcessRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to process")
    source: str = Field(..., description="Data source: google or perplexity")

class BatchProcessRequest(BaseModel):
    source: str = Field(..., description="Data source: google or perplexity")
    limit: Optional[int] = Field(10, description="Maximum number of products to process")

class StatusRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to check")

# Response models
class ProcessResponse(BaseModel):
    status: str
    product_id: str
    data_source: str
    run_id: Optional[str] = None
    analysis_id: Optional[str] = None
    final_output_path: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

class StatusResponse(BaseModel):
    product_id: str
    data_source: str
    status: str
    analysis_id: Optional[str] = None
    dna_blueprint: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    try:
        yield
    finally:
        logger.info("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="GodsEye DNA Pipeline API",
    description="Serverless API to run DNA analysis pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return FileResponse('static/serverless_index.html')

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "failed",
            "error": "Internal server error",
            "details": str(exc)
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/process", response_model=ProcessResponse)
async def process_product(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a single product through the DNA pipeline
    
    - **product_id**: Product identifier
    - **source**: Data source (google or perplexity)
    """
    def _run_pipeline_background(ds: DataSource, product_id: str) -> None:
        orchestrator = DatabasePipelineOrchestrator()
        orchestrator.process_product_from_database(ds, product_id)
    try:
        # Validate source
        try:
            data_source = DataSource(request.source.lower())
        except ValueError:
            return ProcessResponse(
                status="failed",
                product_id=request.product_id,
                data_source=request.source,
                error=f"Invalid source '{request.source}'. Must be 'google' or 'perplexity'",
                message="Validation failed"
            )
        
        logger.info(f"Received process request for product {request.product_id} from {data_source.value}")

        # Quick freshness check inline (fast) to preserve current UX:
        # if up-to-date, return skipped immediately; otherwise run full pipeline in background.
        try:
            orchestrator = DatabasePipelineOrchestrator()
            input_rows = orchestrator.db_manager.fetch_all_input_rows_for_product(data_source, request.product_id)
            if not input_rows:
                return ProcessResponse(
                    status="failed",
                    product_id=request.product_id,
                    data_source=request.source,
                    error=f"Product {request.product_id} not found in {data_source.value} table",
                    message="Validation failed",
                )

            current_input_hash = orchestrator._generate_input_hash(input_rows)
            should_skip, existing_analysis = orchestrator.db_manager.check_existing_analysis(
                data_source, request.product_id, current_input_hash
            )
            if should_skip and existing_analysis:
                return ProcessResponse(
                    status="skipped",
                    product_id=request.product_id,
                    data_source=data_source.value,
                    analysis_id=existing_analysis.id,
                    final_output_path=None,
                    message="Skipped - Up to date",
                )
        except Exception:
            # If freshness check fails, still allow background processing attempt.
            pass

        background_tasks.add_task(_run_pipeline_background, data_source, request.product_id)
        return ProcessResponse(
            status="accepted",
            product_id=request.product_id,
            data_source=data_source.value,
            final_output_path=None,
            message="Processing started in background. Check /status for updates.",
        )
        
    except Exception as e:
        logger.error(f"Error processing product {request.product_id}: {str(e)}", exc_info=True)
        try:
            from error_email_sender import send_ai_error_email
            send_ai_error_email(
                error=e,
                error_context="API /process request failed",
                metadata={
                    "product_id": request.product_id,
                    "data_source": request.source,
                    "stage": "serverless_api./process",
                },
            )
        except Exception as email_err:
            logger.error(f"Failed to send error email for API /process: {email_err}")
        return ProcessResponse(
            status="failed",
            product_id=request.product_id,
            data_source=request.source,
            error=str(e),
            message="Processing failed"
        )

@app.post("/process-batch", response_model=List[ProcessResponse])
async def process_batch(request: BatchProcessRequest):
    """
    Process a batch of products through the DNA pipeline
    
    - **source**: Data source (google or perplexity)
    - **limit**: Maximum number of products to process (default: 10)
    """
    try:
        # Validate source
        try:
            data_source = DataSource(request.source.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source '{request.source}'. Must be 'google' or 'perplexity'"
            )
        
        logger.info(f"Received batch process request for {data_source.value} with limit {request.limit}")
        
        # Instantiate DatabasePipelineOrchestrator per request
        orchestrator = DatabasePipelineOrchestrator()
        
        # Run batch pipeline
        results = orchestrator.process_batch_from_database(data_source, request.limit)
        
        return [ProcessResponse(**result) for result in results]
        
    except Exception as e:
        logger.error(f"Error processing batch for {request.source}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@app.post("/status", response_model=StatusResponse)
async def check_status(request: StatusRequest):
    """
    Check the status and results of a processed product
    
    - **product_id**: Product identifier to check
    """
    try:
        # Check both sources
        results = []
        for source in [DataSource.GOOGLE, DataSource.PERPLEXITY]:
            try:
                # Instantiate DatabasePipelineOrchestrator per request
                orchestrator = DatabasePipelineOrchestrator()
                should_skip, existing = orchestrator.db_manager.check_existing_analysis(source, request.product_id)
                if existing:
                    results.append({
                        "data_source": source.value,
                        "status": existing.status,
                        "analysis_id": existing.id,
                        "dna_blueprint": existing.dna_blueprint,
                        "created_at": existing.created_at,
                        "updated_at": existing.updated_at
                    })
            except Exception as e:
                logger.warning(f"Error checking {source.value} for product {request.product_id}: {e}")
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis found for product {request.product_id}"
            )
        
        # Return first match (should typically be one)
        result = results[0]
        return StatusResponse(
            product_id=request.product_id,
            data_source=result["data_source"],
            status=result["status"],
            analysis_id=result["analysis_id"],
            dna_blueprint=result["dna_blueprint"],
            created_at=result["created_at"],
            updated_at=result["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking status for product {request.product_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@app.get("/sources")
async def get_sources():
    """Get available data sources"""
    return {
        "sources": [
            {"value": "google", "label": "Google"},
            {"value": "perplexity", "label": "Perplexity"}
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    try:
        orchestrator = DatabasePipelineOrchestrator()
        stats = orchestrator.get_pipeline_statistics()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

# Mount static files at the end to avoid conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serverless_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
