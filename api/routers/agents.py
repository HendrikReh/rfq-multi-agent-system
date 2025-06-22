"""Agent management endpoints."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_agents():
    """List all agents."""
    return {"message": "Agent management - coming soon"}

@router.get("/{agent_id}/health")
async def get_agent_health(agent_id: str):
    """Get agent health status."""
    return {"agent_id": agent_id, "status": "healthy"} 