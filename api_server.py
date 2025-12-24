"""
API Server for GodsEye Pipeline

Provides REST endpoints for pipeline control, monitoring, and results access.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from pipeline_orchestrator import PipelineOrchestrator
from pipeline_models import PipelineRun, Job
from realtime_tracker import connection_manager, realtime_tracker, progress_monitor


# Pydantic models for API
class PipelineRequest(BaseModel):
    ai_response_file: str = "ai_response.json"
    query: Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    current_stage: int
    total_jobs: int
    job_summary: Dict[str, Any]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    url: str
    classification: Optional[str]
    stage_1_status: str
    stage_2_status: str
    stage_3_status: str
    selected_for_stage_2: bool
    selected_for_stage_3: bool


# Initialize FastAPI app
app = FastAPI(
    title="GodsEye Pipeline API",
    description="API for GodsEye content analysis pipeline",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator: Optional[PipelineOrchestrator] = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    try:
        orchestrator = PipelineOrchestrator()
        progress_monitor.orchestrator = orchestrator
        progress_monitor.start_monitoring()
        logger.info("API server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    progress_monitor.stop_monitoring()
    logger.info("API server shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GodsEye Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_runs": len(orchestrator.active_runs) if orchestrator else 0
    }


@app.post("/pipeline/run", response_model=Dict[str, str])
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Start pipeline in background
        def run_pipeline():
            try:
                run_id = orchestrator.run_pipeline(request.ai_response_file)
                logger.info(f"Pipeline run {run_id} completed")
            except Exception as e:
                logger.error(f"Pipeline run failed: {e}")
        
        background_tasks.add_task(run_pipeline)
        
        # Create run first to get ID
        ai_response = orchestrator.load_ai_response(request.ai_response_file)
        run_id = orchestrator.create_run(ai_response)
        
        return {
            "run_id": run_id,
            "status": "started",
            "message": "Pipeline execution started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/runs")
async def list_runs():
    """List all pipeline runs"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    runs = []
    for run_id, run in orchestrator.active_runs.items():
        runs.append({
            "run_id": run_id,
            "status": run.status,
            "current_stage": run.current_stage,
            "total_jobs": len(run.jobs),
            "created_at": run.created_at.isoformat(),
            "query": run.query
        })
    
    return {"runs": runs}


@app.get("/pipeline/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get detailed status of a specific run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    status = orchestrator.get_run_status(run_id)
    
    return RunStatusResponse(
        run_id=run_id,
        status=status["status"],
        current_stage=status["current_stage"],
        total_jobs=status["total_links"],
        job_summary=status["job_summary"],
        created_at=status["created_at"],
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None
    )


@app.get("/pipeline/runs/{run_id}/jobs")
async def get_run_jobs(run_id: str):
    """Get all jobs for a specific run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    jobs = []
    
    for job_id, job in run.jobs.items():
        jobs.append(JobStatusResponse(
            job_id=job_id,
            run_id=run_id,
            url=job.url,
            classification=job.classification,
            stage_1_status=job.stage_1_status,
            stage_2_status=job.stage_2_status,
            stage_3_status=job.stage_3_status,
            selected_for_stage_2=job.selected_for_stage_2,
            selected_for_stage_3=job.selected_for_stage_3
        ))
    
    return {"jobs": jobs}


@app.get("/pipeline/runs/{run_id}/jobs/{job_id}")
async def get_job_details(run_id: str, job_id: str):
    """Get detailed information about a specific job"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    if job_id not in run.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = run.jobs[job_id]
    
    return {
        "job_id": job_id,
        "run_id": run_id,
        "url": job.url,
        "text": job.text,
        "position": job.position,
        "classification": job.classification,
        "source_link": job.source_link,
        "stage_1_status": job.stage_1_status,
        "stage_1_output_path": job.stage_1_output_path,
        "stage_2_status": job.stage_2_status,
        "stage_2_output_path": job.stage_2_output_path,
        "stage_3_status": job.stage_3_status,
        "stage_3_output_path": job.stage_3_output_path,
        "selected_for_stage_2": job.selected_for_stage_2,
        "selected_for_stage_3": job.selected_for_stage_3,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries,
        "created_at": job.created_at.isoformat(),
        "stage_times": {
            "stage_1": {
                "start": job.stage_1_start_time.isoformat() if job.stage_1_start_time else None,
                "end": job.stage_1_end_time.isoformat() if job.stage_1_end_time else None,
                "duration": (job.stage_1_end_time - job.stage_1_start_time).total_seconds() 
                           if job.stage_1_start_time and job.stage_1_end_time else None
            },
            "stage_2": {
                "start": job.stage_2_start_time.isoformat() if job.stage_2_start_time else None,
                "end": job.stage_2_end_time.isoformat() if job.stage_2_end_time else None,
                "duration": (job.stage_2_end_time - job.stage_2_start_time).total_seconds() 
                           if job.stage_2_start_time and job.stage_2_end_time else None
            },
            "stage_3": {
                "start": job.stage_3_start_time.isoformat() if job.stage_3_start_time else None,
                "end": job.stage_3_end_time.isoformat() if job.stage_3_end_time else None,
                "duration": (job.stage_3_end_time - job.stage_3_start_time).total_seconds() 
                           if job.stage_3_start_time and job.stage_3_end_time else None
            }
        }
    }


@app.get("/pipeline/runs/{run_id}/results")
async def get_run_results(run_id: str):
    """Get final results for a completed run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Run not completed yet")
    
    # Load final aggregation results
    results_dir = os.path.join("outputs", "stage_3_results", f"run_{run_id}")
    aggregation_file = os.path.join(results_dir, "final_aggregation.json")
    
    if not os.path.exists(aggregation_file):
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        with open(aggregation_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {e}")


@app.get("/pipeline/runs/{run_id}/results/summary")
async def get_run_summary(run_id: str):
    """Get summary report for a completed run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Run not completed yet")
    
    # Load summary report
    results_dir = os.path.join("outputs", "stage_3_results", f"run_{run_id}")
    summary_file = os.path.join(results_dir, "summary_report.txt")
    
    if not os.path.exists(summary_file):
        raise HTTPException(status_code=404, detail="Summary not found")
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        return {"summary": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load summary: {e}")


@app.get("/pipeline/runs/{run_id}/results/recommendations")
async def get_run_recommendations(run_id: str):
    """Get content recommendations for a completed run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Run not completed yet")
    
    # Load recommendations
    results_dir = os.path.join("outputs", "stage_3_results", f"run_{run_id}")
    recommendations_file = os.path.join(results_dir, "content_recommendations.json")
    
    if not os.path.exists(recommendations_file):
        raise HTTPException(status_code=404, detail="Recommendations not found")
    
    try:
        with open(recommendations_file, 'r', encoding='utf-8') as f:
            recommendations = json.load(f)
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load recommendations: {e}")


@app.get("/pipeline/runs/{run_id}/results/opportunities")
async def get_run_opportunities(run_id: str):
    """Get ranking opportunities for a completed run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = orchestrator.active_runs[run_id]
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Run not completed yet")
    
    # Load opportunities
    results_dir = os.path.join("outputs", "stage_3_results", f"run_{run_id}")
    opportunities_file = os.path.join(results_dir, "ranking_opportunities.json")
    
    if not os.path.exists(opportunities_file):
        raise HTTPException(status_code=404, detail="Opportunities not found")
    
    try:
        with open(opportunities_file, 'r', encoding='utf-8') as f:
            opportunities = json.load(f)
        
        return {"opportunities": opportunities}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load opportunities: {e}")


@app.delete("/pipeline/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a pipeline run"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if run_id not in orchestrator.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Remove from active runs
    del orchestrator.active_runs[run_id]
    
    return {"message": f"Run {run_id} deleted successfully"}


@app.get("/stats")
async def get_pipeline_stats():
    """Get overall pipeline statistics"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    total_runs = len(orchestrator.active_runs)
    completed_runs = sum(1 for run in orchestrator.active_runs.values() if run.status == "completed")
    running_runs = sum(1 for run in orchestrator.active_runs.values() if run.status == "running")
    failed_runs = sum(1 for run in orchestrator.active_runs.values() if run.status == "failed")
    
    total_jobs = sum(len(run.jobs) for run in orchestrator.active_runs.values())
    
    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "running_runs": running_runs,
        "failed_runs": failed_runs,
        "total_jobs": total_jobs,
        "success_rate": completed_runs / total_runs if total_runs > 0 else 0
    }


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket)
    await realtime_tracker.subscribe_to_run(websocket, run_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        await realtime_tracker.unsubscribe_from_run(websocket, run_id)


@app.websocket("/ws")
async def websocket_global_endpoint(websocket: WebSocket):
    """Global WebSocket endpoint for all updates"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Global echo: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# Serve static files (for potential future dashboard)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
