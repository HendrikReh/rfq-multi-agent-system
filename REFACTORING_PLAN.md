# Multi-Agent RFQ System - Refactoring Plan

## 🎯 Overview

This document outlines a comprehensive refactoring plan to restructure the RFQ system following modern multi-agent architecture best practices, inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system).

## 📋 Current Issues & Improvements Needed

### Current Structure Problems:
1. **Flat agent organization** - All agents in single directory
2. **Mixed concerns** - Business logic, orchestration, and infrastructure mixed
3. **No clear separation** between core agents, specialized agents, and orchestration
4. **Limited extensibility** for MCP Server/Client integration
5. **No standardized agent interfaces** or communication patterns
6. **Monolithic integration framework** - Hard to test and extend

### Target Architecture Benefits:
- **Modular design** with clear separation of concerns
- **Standardized agent interfaces** for consistency
- **Extensible orchestration** supporting multiple coordination patterns
- **MCP-ready architecture** for future integration
- **FastAPI-ready structure** for web service deployment
- **Production-ready observability** and monitoring

---

## 🏗️ Proposed New Structure

```
rfc-pydanticai-openai/
├── pyproject.toml                    # Updated project config
├── README.md                         # Updated documentation
├── CLAUDE.md                         # Technical documentation
├── .env.example                      # Environment template
├── 
├── src/                              # Main source code
│   └── rfq_system/                   # Main package
│       ├── __init__.py
│       ├── config/                   # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py           # Pydantic settings
│       │   ├── models.py             # Model configurations
│       │   └── logging.py            # Logging configuration
│       │
│       ├── core/                     # Core domain models & interfaces
│       │   ├── __init__.py
│       │   ├── models/               # Pydantic models
│       │   │   ├── __init__.py
│       │   │   ├── rfq.py           # RFQ-related models
│       │   │   ├── customer.py       # Customer-related models
│       │   │   ├── business.py       # Business logic models
│       │   │   └── system.py         # System models
│       │   ├── interfaces/           # Agent interfaces & protocols
│       │   │   ├── __init__.py
│       │   │   ├── agent.py          # Base agent interface
│       │   │   ├── orchestrator.py   # Orchestrator interface
│       │   │   └── tools.py          # Tool interfaces
│       │   └── exceptions.py         # Custom exceptions
│       │
│       ├── agents/                   # Agent implementations
│       │   ├── __init__.py
│       │   ├── base/                 # Base agent classes
│       │   │   ├── __init__.py
│       │   │   ├── agent.py          # Abstract base agent
│       │   │   ├── specialized.py    # Specialized agent base
│       │   │   └── delegating.py     # Delegating agent base
│       │   │
│       │   ├── core/                 # Core business agents
│       │   │   ├── __init__.py
│       │   │   ├── rfq_parser.py
│       │   │   ├── customer_intent.py
│       │   │   ├── conversation_state.py
│       │   │   ├── interaction_decision.py
│       │   │   ├── question_generation.py
│       │   │   ├── pricing_strategy.py
│       │   │   └── customer_response.py
│       │   │
│       │   ├── specialized/          # Domain-specific agents
│       │   │   ├── __init__.py
│       │   │   ├── competitive_intelligence.py
│       │   │   ├── risk_assessment.py
│       │   │   ├── contract_terms.py
│       │   │   └── proposal_writer.py
│       │   │
│       │   └── evaluation/           # Evaluation & monitoring
│       │       ├── __init__.py
│       │       ├── performance_monitor.py
│       │       └── quality_assessor.py
│       │
│       ├── orchestration/            # Multi-agent coordination
│       │   ├── __init__.py
│       │   ├── coordinators/         # Different coordination patterns
│       │   │   ├── __init__.py
│       │   │   ├── sequential.py     # Sequential execution
│       │   │   ├── parallel.py       # Parallel execution
│       │   │   ├── graph_based.py    # Graph-based state machine
│       │   │   └── adaptive.py       # Adaptive coordination
│       │   ├── strategies/           # Orchestration strategies
│       │   │   ├── __init__.py
│       │   │   ├── delegation.py     # Agent delegation patterns
│       │   │   ├── handoff.py        # Agent handoff patterns
│       │   │   └── consensus.py      # Multi-agent consensus
│       │   └── memory/               # Shared memory & context
│       │       ├── __init__.py
│       │       ├── context_manager.py
│       │       └── conversation_memory.py
│       │
│       ├── tools/                    # Agent tools & utilities
│       │   ├── __init__.py
│       │   ├── search/               # Search tools
│       │   │   ├── __init__.py
│       │   │   └── web_search.py
│       │   ├── analysis/             # Analysis tools
│       │   │   ├── __init__.py
│       │   │   ├── market_analysis.py
│       │   │   └── risk_analysis.py
│       │   └── generation/           # Content generation tools
│       │       ├── __init__.py
│       │       ├── document_generator.py
│       │       └── quote_generator.py
│       │
│       ├── integrations/             # External service integrations
│       │   ├── __init__.py
│       │   ├── mcp/                  # Model Context Protocol
│       │   │   ├── __init__.py
│       │   │   ├── server.py         # MCP Server implementation
│       │   │   ├── client.py         # MCP Client implementation
│       │   │   └── tools.py          # MCP tool definitions
│       │   ├── api/                  # External API integrations
│       │   │   ├── __init__.py
│       │   │   ├── openai_client.py
│       │   │   └── anthropic_client.py
│       │   └── storage/              # Data persistence
│       │       ├── __init__.py
│       │       ├── file_storage.py
│       │       └── database.py
│       │
│       ├── monitoring/               # Observability & monitoring
│       │   ├── __init__.py
│       │   ├── health/               # Health monitoring
│       │   │   ├── __init__.py
│       │   │   ├── agent_health.py
│       │   │   └── system_health.py
│       │   ├── metrics/              # Performance metrics
│       │   │   ├── __init__.py
│       │   │   ├── performance.py
│       │   │   └── usage.py
│       │   └── tracing/              # Distributed tracing
│       │       ├── __init__.py
│       │       └── tracer.py
│       │
│       └── utils/                    # Shared utilities
│           ├── __init__.py
│           ├── logging.py
│           ├── validation.py
│           └── helpers.py
│
├── api/                              # FastAPI web service
│   ├── __init__.py
│   ├── main.py                       # FastAPI application
│   ├── routers/                      # API route handlers
│   │   ├── __init__.py
│   │   ├── rfq.py                    # RFQ processing endpoints
│   │   ├── agents.py                 # Agent management endpoints
│   │   ├── health.py                 # Health check endpoints
│   │   └── admin.py                  # Admin endpoints
│   ├── middleware/                   # Custom middleware
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── logging.py
│   │   └── error_handling.py
│   └── dependencies/                 # FastAPI dependencies
│       ├── __init__.py
│       ├── auth.py
│       └── database.py
│
├── mcp_server/                       # MCP Server implementation
│   ├── __init__.py
│   ├── server.py                     # Main MCP server
│   ├── tools/                        # MCP tool implementations
│   │   ├── __init__.py
│   │   ├── rfq_tools.py
│   │   └── agent_tools.py
│   └── resources/                    # MCP resources
│       ├── __init__.py
│       └── rfq_resources.py
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   ├── unit/                         # Unit tests
│   │   ├── __init__.py
│   │   ├── test_agents/              # Agent tests
│   │   │   ├── __init__.py
│   │   │   ├── test_core_agents.py
│   │   │   ├── test_specialized_agents.py
│   │   │   └── test_orchestration.py
│   │   ├── test_models/              # Model tests
│   │   │   ├── __init__.py
│   │   │   └── test_data_models.py
│   │   └── test_utils/               # Utility tests
│   │       ├── __init__.py
│   │       └── test_helpers.py
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_integration.py
│   │   ├── test_mcp_integration.py
│   │   └── test_multi_agent_workflows.py
│   ├── performance/                  # Performance tests
│   │   ├── __init__.py
│   │   ├── test_load.py
│   │   └── test_scalability.py
│   └── fixtures/                     # Test fixtures
│       ├── __init__.py
│       ├── sample_data.py
│       └── mock_agents.py
│
├── scripts/                          # Utility scripts
│   ├── setup.py                      # Environment setup
│   ├── migrate.py                    # Migration script
│   ├── deploy.py                     # Deployment script
│   └── benchmark.py                  # Performance benchmarking
│
├── docs/                             # Documentation
│   ├── architecture.md              # Architecture overview
│   ├── agents.md                     # Agent documentation
│   ├── api.md                        # API documentation
│   ├── mcp.md                        # MCP integration guide
│   └── deployment.md                # Deployment guide
│
└── examples/                         # Usage examples
    ├── __init__.py
    ├── basic_usage.py                # Basic RFQ processing
    ├── advanced_orchestration.py     # Advanced multi-agent patterns
    ├── mcp_integration.py            # MCP usage examples
    └── api_client.py                 # API client examples
```

---

## 🔧 Key Architectural Improvements

### 1. **Standardized Agent Interface**
```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, List, Optional
from pydantic_ai import Agent

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

class BaseAgent(ABC, Generic[T, R]):
    """Base interface for all agents in the system."""
    
    @abstractmethod
    async def process(self, input_data: T, context: Dict[str, Any]) -> R:
        """Process input and return result."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        pass
```

### 2. **Orchestration Framework**
Following Anthropic's pattern with lead agent + subagents:
- **Coordinators**: Different coordination patterns (sequential, parallel, graph-based)
- **Strategies**: Delegation, handoff, and consensus patterns
- **Memory**: Shared context and conversation memory

### 3. **MCP Integration Ready**
- Dedicated MCP server/client implementations
- Tool definitions compatible with MCP protocol
- Resource management for MCP resources

### 4. **FastAPI Integration**
- Clean API layer separate from business logic
- Proper middleware for auth, logging, error handling
- RESTful endpoints for RFQ processing and agent management

### 5. **Production Observability**
- Health monitoring for individual agents and system
- Performance metrics and usage tracking
- Distributed tracing for multi-agent workflows

---

## 📋 Migration Strategy

### Phase 1: Core Refactoring (Week 1-2)
1. Create new directory structure
2. Extract and refactor core models
3. Implement base agent interfaces
4. Migrate core agents to new structure

### Phase 2: Orchestration Enhancement (Week 3-4)
1. Implement orchestration framework
2. Migrate existing orchestrators
3. Add parallel execution patterns
4. Implement memory management

### Phase 3: Integration Layer (Week 5-6)
1. Build FastAPI service layer
2. Implement MCP server/client
3. Add monitoring and observability
4. Create comprehensive test suite

### Phase 4: Production Readiness (Week 7-8)
1. Performance optimization
2. Security hardening
3. Documentation completion
4. Deployment automation

---

## 🧪 Testing Strategy

### Test Categories:
1. **Unit Tests**: Individual agent functionality
2. **Integration Tests**: Multi-agent workflows
3. **Performance Tests**: Load and scalability
4. **Contract Tests**: API and MCP interface compliance

### Test Infrastructure:
- Pytest with async support
- TestModel for agent testing
- Mock MCP server for integration tests
- Performance benchmarking suite

---

## 📚 Documentation Updates

### Required Documentation:
1. **Architecture Guide**: System design and patterns
2. **Agent Development Guide**: How to create new agents
3. **API Reference**: FastAPI endpoint documentation
4. **MCP Integration Guide**: Using MCP server/client
5. **Deployment Guide**: Production deployment

---

This refactoring plan creates a production-ready, extensible multi-agent system that follows modern architectural patterns and anticipates future requirements for MCP and FastAPI integration. 