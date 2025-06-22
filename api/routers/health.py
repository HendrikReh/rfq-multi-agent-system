"""
Health check endpoints for the RFQ API.

Provides comprehensive health monitoring including:
- Basic health status
- Agent health monitoring  
- System resource monitoring
- Database connectivity
"""

import time
import psutil
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str  # healthy, degraded, unhealthy
    timestamp: float
    uptime_seconds: float
    version: str = "0.2.0"


class DetailedHealthStatus(BaseModel):
    """Detailed health status with system metrics."""
    status: str
    timestamp: float
    uptime_seconds: float
    version: str = "0.2.0"
    system: Dict
    agents: Dict
    database: Dict
    dependencies: Dict


@router.get("/", response_model=HealthStatus, summary="Basic Health Check")
async def health_check():
    """Basic health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        uptime_seconds=time.time() - (time.time() - 100),  # Mock uptime
        version="0.2.0"
    )


@router.get("/detailed", response_model=DetailedHealthStatus, summary="Detailed Health Check")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_status = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3),
        "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "degraded"
    }
    
    # Agent health (mock for now)
    agents_status = {
        "rfq_parser": {"status": "healthy", "response_time_ms": 150},
        "customer_intent": {"status": "healthy", "response_time_ms": 200},
        "pricing_strategy": {"status": "healthy", "response_time_ms": 180},
        "total_agents": 3,
        "healthy_agents": 3,
        "status": "healthy"
    }
    
    # Database health (mock for now)
    database_status = {
        "connected": True,
        "response_time_ms": 25,
        "status": "healthy"
    }
    
    # Dependencies health
    dependencies_status = {
        "openai_api": {"status": "healthy", "response_time_ms": 300},
        "anthropic_api": {"status": "healthy", "response_time_ms": 250},
        "status": "healthy"
    }
    
    # Overall status
    overall_status = "healthy"
    if (system_status["status"] != "healthy" or 
        agents_status["status"] != "healthy" or
        database_status["status"] != "healthy" or
        dependencies_status["status"] != "healthy"):
        overall_status = "degraded"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=time.time(),
        uptime_seconds=time.time() - (time.time() - 100),  # Mock uptime
        version="0.2.0",
        system=system_status,
        agents=agents_status,
        database=database_status,
        dependencies=dependencies_status
    )


@router.get("/agents", summary="Agent Health Status")
async def agent_health():
    """Get health status of all agents."""
    # TODO: Implement actual agent health monitoring
    return {
        "agents": {
            "rfq_parser": {
                "status": "healthy",
                "last_heartbeat": time.time(),
                "response_time_ms": 150,
                "error_count": 0,
                "total_requests": 100
            },
            "customer_intent": {
                "status": "healthy", 
                "last_heartbeat": time.time(),
                "response_time_ms": 200,
                "error_count": 1,
                "total_requests": 95
            },
            "pricing_strategy": {
                "status": "healthy",
                "last_heartbeat": time.time(),
                "response_time_ms": 180,
                "error_count": 0,
                "total_requests": 80
            }
        },
        "summary": {
            "total": 3,
            "healthy": 3,
            "degraded": 0,
            "unhealthy": 0
        }
    }


@router.get("/ready", summary="Readiness Check")
async def readiness_check():
    """Check if the service is ready to accept requests."""
    # TODO: Implement actual readiness checks
    # - All agents initialized
    # - Database connections established
    # - External dependencies available
    
    return {
        "ready": True,
        "checks": {
            "agents_initialized": True,
            "database_connected": True,
            "external_apis_available": True
        }
    }


@router.get("/live", summary="Liveness Check")
async def liveness_check():
    """Check if the service is alive (for Kubernetes)."""
    return {"alive": True, "timestamp": time.time()} 