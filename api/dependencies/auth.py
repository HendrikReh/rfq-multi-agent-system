"""Authentication dependencies for the API."""

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Get current authenticated user."""
    # TODO: Implement actual authentication
    # For now, return a mock user
    return {"user_id": "test_user", "permissions": ["read", "write"]} 