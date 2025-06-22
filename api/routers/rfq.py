"""RFQ processing endpoints."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_rfqs():
    """List RFQ requests."""
    return {"message": "RFQ endpoints - coming soon"}

@router.post("/")
async def create_rfq():
    """Create new RFQ."""
    return {"message": "Create RFQ - coming soon"} 