"""Admin endpoints."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def admin_dashboard():
    """Admin dashboard."""
    return {"message": "Admin endpoints - coming soon"} 