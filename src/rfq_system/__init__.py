"""
RFQ Multi-Agent System

A production-ready multi-agent system for Request for Quote (RFQ) processing
built with PydanticAI, featuring FastAPI integration and MCP support.

This system follows modern multi-agent architecture patterns inspired by
Anthropic's research on multi-agent systems, providing:

- Modular agent architecture with standardized interfaces
- Orchestration patterns (sequential, parallel, graph-based)
- Production observability and monitoring
- FastAPI web service integration
- Model Context Protocol (MCP) support
- Comprehensive testing framework

Key Components:
- Core agents for RFQ processing workflow
- Specialized agents for domain-specific tasks
- Orchestration framework for multi-agent coordination
- Integration layer for external services
- Monitoring and observability infrastructure
"""

__version__ = "0.2.0"
__author__ = "RFQ System Team"
__email__ = "team@rfq-system.com"

# Core exports
from .core.interfaces.agent import BaseAgent, AgentCapability, AgentStatus
from .core.models.rfq import RFQRequirements, RFQProcessingResult
from .core.models.customer import CustomerIntent, CustomerSentiment
from .orchestration.coordinators.sequential import SequentialCoordinator
from .orchestration.coordinators.parallel import ParallelCoordinator

__all__ = [
    # Core interfaces
    "BaseAgent",
    "AgentCapability", 
    "AgentStatus",
    # Core models
    "RFQRequirements",
    "RFQProcessingResult",
    "CustomerIntent",
    "CustomerSentiment",
    # Orchestration
    "SequentialCoordinator",
    "ParallelCoordinator",
] 