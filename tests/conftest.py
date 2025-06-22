"""Test configuration and fixtures."""
import pytest
from pydantic_ai import models

# Disable model requests globally for tests
models.ALLOW_MODEL_REQUESTS = False

@pytest.fixture(autouse=True)
def disable_model_requests():
    """Ensure model requests are disabled for all tests."""
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = original
