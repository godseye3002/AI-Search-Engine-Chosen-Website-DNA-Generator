"""
Real-time Progress Tracker

Provides WebSocket-based real-time updates for pipeline progress.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass
import threading
import time


@dataclass
class ProgressUpdate:
    """Progress update message"""
    type: str  # 'run_status', 'job_status', 'stage_progress', 'error'
    run_id: str
    timestamp: str
    data: Dict[str, Any]


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Failed to send to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.discard(conn)


class RealtimeTracker:
    """Real-time progress tracking system"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        self._subscribers = {}  # run_id -> set of connections
        self.last_update = {}  # run_id -> last update timestamp
    
    async def subscribe_to_run(self, websocket: WebSocket, run_id: str):
        """Subscribe a WebSocket to updates for a specific run"""
        if run_id not in self._subscribers:
            self._subscribers[run_id] = set()
        self._subscribers[run_id].add(websocket)
        
        # Send current status immediately
        await self.send_current_status(websocket, run_id)
    
    async def unsubscribe_from_run(self, websocket: WebSocket, run_id: str):
        """Unsubscribe a WebSocket from updates for a specific run"""
        if run_id in self._subscribers:
            self._subscribers[run_id].discard(websocket)
            if not self._subscribers[run_id]:
                del self._subscribers[run_id]
    
    async def send_current_status(self, websocket: WebSocket, run_id: str):
        """Send current status of a run to a WebSocket"""
        try:
            # This would need to be implemented by the orchestrator
            # For now, send a placeholder
            status_update = ProgressUpdate(
                type="run_status",
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={"status": "loading", "message": "Loading current status..."}
            )
            await websocket.send_text(json.dumps(status_update.__dict__))
        except Exception as e:
            self.logger.error(f"Failed to send current status: {e}")
    
    async def notify_run_status_change(self, run_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Notify subscribers about run status change"""
        update = ProgressUpdate(
            type="run_status",
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "status": status,
                "details": details or {}
            }
        )
        
        await self._broadcast_to_run_subscribers(run_id, update)
    
    async def notify_stage_progress(self, run_id: str, stage: int, progress: Dict[str, Any]):
        """Notify subscribers about stage progress"""
        update = ProgressUpdate(
            type="stage_progress",
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "stage": stage,
                "progress": progress
            }
        )
        
        await self._broadcast_to_run_subscribers(run_id, update)
    
    async def notify_job_status_change(self, run_id: str, job_id: str, stage: int, status: str, details: Optional[Dict[str, Any]] = None):
        """Notify subscribers about job status change"""
        update = ProgressUpdate(
            type="job_status",
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "job_id": job_id,
                "stage": stage,
                "status": status,
                "details": details or {}
            }
        )
        
        await self._broadcast_to_run_subscribers(run_id, update)
    
    async def notify_error(self, run_id: str, error: str, context: Optional[Dict[str, Any]] = None):
        """Notify subscribers about an error"""
        update = ProgressUpdate(
            type="error",
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "error": error,
                "context": context or {}
            }
        )
        
        await self._broadcast_to_run_subscribers(run_id, update)
    
    async def _broadcast_to_run_subscribers(self, run_id: str, update: ProgressUpdate):
        """Broadcast update to all subscribers of a specific run"""
        if run_id not in self._subscribers:
            return
        
        message = json.dumps(update.__dict__)
        disconnected = set()
        
        for websocket in self._subscribers[run_id]:
            try:
                await websocket.send_text(message)
            except Exception as e:
                self.logger.error(f"Failed to send update to subscriber: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for conn in disconnected:
            self._subscribers[run_id].discard(conn)
        
        # Clean up if no subscribers left
        if not self._subscribers[run_id]:
            del self._subscribers[run_id]
        
        self.last_update[run_id] = datetime.now().isoformat()


class ProgressMonitor:
    """Monitors pipeline progress and sends real-time updates"""
    
    def __init__(self, tracker: RealtimeTracker, orchestrator):
        self.tracker = tracker
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Progress monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Progress monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self._check_for_updates()
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _check_for_updates(self):
        """Check for pipeline updates"""
        if not self.orchestrator:
            return
        
        # This is a simplified version - in practice, you'd track changes more efficiently
        for run_id, run in self.orchestrator.active_runs.items():
            # Check if run status changed
            current_status = run.status
            # You'd need to track previous status to detect changes
            
            # Check stage progress
            if run.current_stage == 1:
                self._check_stage_1_progress(run_id, run)
            elif run.current_stage == 2:
                self._check_stage_2_progress(run_id, run)
            elif run.current_stage == 3:
                self._check_stage_3_progress(run_id, run)
    
    def _check_stage_1_progress(self, run_id: str, run):
        """Check Stage 1 progress"""
        jobs = list(run.jobs.values())
        
        completed = sum(1 for job in jobs if job.stage_1_status == 'completed')
        failed = sum(1 for job in jobs if job.stage_1_status == 'failed')
        running = sum(1 for job in jobs if job.stage_1_status == 'running')
        total = len(jobs)
        
        progress = {
            "completed": completed,
            "failed": failed,
            "running": running,
            "total": total,
            "percentage": (completed + failed) / total * 100 if total > 0 else 0
        }
        
        # Send update (in practice, you'd check if this changed from last time)
        asyncio.create_task(self.tracker.notify_stage_progress(run_id, 1, progress))
    
    def _check_stage_2_progress(self, run_id: str, run):
        """Check Stage 2 progress"""
        stage_2_jobs = [job for job in run.jobs.values() if job.selected_for_stage_2]
        
        completed = sum(1 for job in stage_2_jobs if job.stage_2_status == 'completed')
        failed = sum(1 for job in stage_2_jobs if job.stage_2_status == 'failed')
        running = sum(1 for job in stage_2_jobs if job.stage_2_status == 'running')
        total = len(stage_2_jobs)
        
        progress = {
            "completed": completed,
            "failed": failed,
            "running": running,
            "total": total,
            "percentage": (completed + failed) / total * 100 if total > 0 else 0
        }
        
        asyncio.create_task(self.tracker.notify_stage_progress(run_id, 2, progress))
    
    def _check_stage_3_progress(self, run_id: str, run):
        """Check Stage 3 progress"""
        stage_3_jobs = [job for job in run.jobs.values() if job.selected_for_stage_3]
        
        completed = sum(1 for job in stage_3_jobs if job.stage_3_status == 'completed')
        failed = sum(1 for job in stage_3_jobs if job.stage_3_status == 'failed')
        total = len(stage_3_jobs)
        
        progress = {
            "completed": completed,
            "failed": failed,
            "total": total,
            "percentage": completed / total * 100 if total > 0 else 0
        }
        
        asyncio.create_task(self.tracker.notify_stage_progress(run_id, 3, progress))


# Global instances
connection_manager = ConnectionManager()
realtime_tracker = RealtimeTracker(connection_manager)
progress_monitor = ProgressMonitor(realtime_tracker, None)  # Will be set later
