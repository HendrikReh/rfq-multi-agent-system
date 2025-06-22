"""
FastAPI application for the RFQ Multi-Agent System.

This module provides a production-ready web service interface for the
RFQ processing system, featuring:
- RESTful API endpoints for RFQ processing
- Agent management and monitoring
- Health checks and observability
- Authentication and authorization
- Rate limiting and error handling
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our system components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rfq_system.core.models.rfq import RFQProcessingResult, RFQRequirements
from rfq_system.core.models.customer import CustomerProfile, CustomerIntent
from rfq_system.core.interfaces.agent import AgentHealthStatus
from rfq_system.orchestration.coordinators.parallel import ParallelCoordinator

# Import routers
from .routers import rfq, agents, health, admin
from .middleware.logging import LoggingMiddleware
from .middleware.error_handling import ErrorHandlingMiddleware
from .dependencies.auth import get_current_user


# Application state
app_state = {
    "coordinator": None,
    "agents": {},
    "startup_time": None,
    "request_count": 0,
    "error_count": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logging.info("Starting RFQ Multi-Agent System API...")
    app_state["startup_time"] = time.time()
    
    # Initialize coordinator
    app_state["coordinator"] = ParallelCoordinator(
        max_concurrent_tasks=10,
        enable_health_monitoring=True
    )
    
    # TODO: Initialize agents here
    # app_state["agents"] = await initialize_agents()
    
    logging.info("RFQ API startup complete")
    
    yield
    
    # Shutdown
    logging.info("Shutting down RFQ API...")
    
    # Cancel any running tasks
    if app_state["coordinator"]:
        await app_state["coordinator"].cancel_all_tasks()
    
    # TODO: Cleanup agents
    
    logging.info("RFQ API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RFQ Multi-Agent System API",
    description="Production-ready API for AI-powered Request for Quote processing",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(rfq.router, prefix="/api/v1/rfq", tags=["RFQ Processing"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agent Management"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])


# Root endpoint
@app.get("/", summary="API Root", description="Get basic API information")
async def root():
    """Root endpoint providing API information."""
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    return {
        "service": "RFQ Multi-Agent System API",
        "version": "0.2.0",
        "status": "operational",
        "uptime_seconds": uptime,
        "docs_url": "/docs",
        "health_url": "/health",
        "features": [
            "Multi-agent RFQ processing",
            "Parallel agent coordination", 
            "Real-time health monitoring",
            "Production observability",
            "RESTful API interface"
        ]
    }


# Metrics endpoint
@app.get("/metrics", summary="System Metrics", description="Get system performance metrics")
async def get_metrics():
    """Get system performance metrics."""
    coordinator_stats = {}
    if app_state["coordinator"]:
        coordinator_stats = app_state["coordinator"].get_execution_stats()
    
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    return {
        "system": {
            "uptime_seconds": uptime,
            "total_requests": app_state["request_count"],
            "total_errors": app_state["error_count"],
            "error_rate": app_state["error_count"] / max(app_state["request_count"], 1),
            "active_agents": len(app_state["agents"])
        },
        "coordinator": coordinator_stats,
        "agents": {
            agent_id: {
                "status": "active",  # TODO: Get real status
                "requests": 0,  # TODO: Get real metrics
                "avg_response_time": 0.0
            }
            for agent_id in app_state["agents"]
        }
    }


# Request models
class RFQProcessRequest(BaseModel):
    """Request model for RFQ processing."""
    customer_request: str = Field(description="Raw customer request text")
    customer_profile: Optional[CustomerProfile] = Field(None, description="Customer profile if available")
    priority: str = Field(default="medium", description="Processing priority")
    execution_mode: str = Field(default="parallel", description="Execution mode (sequential, parallel)")
    timeout_seconds: float = Field(default=30.0, description="Processing timeout")


class ProcessingStatus(BaseModel):
    """Processing status response."""
    request_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    estimated_completion: Optional[float] = None
    current_stage: str = ""
    message: str = ""


# Main processing endpoint
@app.post(
    "/api/v1/process",
    response_model=RFQProcessingResult,
    summary="Process RFQ",
    description="Process a customer RFQ request using the multi-agent system"
)
async def process_rfq(
    request: RFQProcessRequest,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user)
):
    """
    Process an RFQ request using the multi-agent system.
    
    This endpoint orchestrates the entire RFQ processing workflow,
    including requirements extraction, analysis, and quote generation.
    """
    app_state["request_count"] += 1
    
    try:
        # TODO: Implement actual RFQ processing
        # This is a placeholder implementation
        
        # Create a mock result for now
        result = RFQProcessingResult(
            rfq_id=f"rfq_{int(time.time())}",
            status="completed",
            requirements=RFQRequirements(
                product_type="Software Development",
                quantity=1,
                completeness="partial",
                confidence_score=0.8
            ),
            interaction_decision={
                "should_ask_questions": True,
                "should_generate_quote": False,
                "next_action": "gather_requirements",
                "reasoning": "Need more technical details",
                "confidence_level": 3
            },
            processing_time_ms=1500.0,
            agents_involved=["rfq_parser", "customer_intent", "interaction_decision"],
            execution_mode=request.execution_mode,
            confidence_score=0.75
        )
        
        return result
        
    except Exception as e:
        app_state["error_count"] += 1
        logging.error(f"RFQ processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Status endpoint for long-running requests
@app.get(
    "/api/v1/status/{request_id}",
    response_model=ProcessingStatus,
    summary="Get Processing Status",
    description="Get the status of a long-running RFQ processing request"
)
async def get_processing_status(request_id: str):
    """Get the status of a processing request."""
    # TODO: Implement actual status tracking
    return ProcessingStatus(
        request_id=request_id,
        status="completed",
        progress=1.0,
        current_stage="completed",
        message="Processing completed successfully"
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    app_state["error_count"] += 1
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    app_state["error_count"] += 1
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


def start_server():
    """Start the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_server() 