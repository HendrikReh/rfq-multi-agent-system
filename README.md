# RFQ Multi-Agent System

A production-ready multi-agent system for Request for Quote (RFQ) processing built with [PydanticAI](https://ai.pydantic.dev/), featuring FastAPI integration and MCP support.

## Overview

This system demonstrates modern multi-agent architecture patterns inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system), providing:

- **Domain specific Agents** for comprehensive RFQ processing
- **Modular Architecture** with clear separation of concerns
- **Production Orchestration** with parallel execution and health monitoring
- **Comprehensive Testing** with unit, integration, and performance tests
- **Logfire Observability** with complete LLM conversation tracing and performance monitoring

## Architecture

### Core Components

```
src/rfq_system/
├── core/                         # Core domain models & interfaces
├── agents/                       # Agent implementations
│   ├── base/                     # Base agent classes
│   ├── core/                     # Core business agents
│   ├── specialized/              # Domain-specific agents
│   └── evaluation/               # Evaluation & monitoring
├── orchestration/                # Multi-agent coordination
├── tools/                        # Agent tools & utilities
├── integrations/                 # External service integrations
├── monitoring/                   # Observability & monitoring
└── utils/                        # Shared utilities
```

### Agent Ecosystem

#### Core Processing Agents (9 agents)
- **RFQParser** - Requirements extraction and validation
- **ConversationStateAgent** - State management and tracking
- **CustomerIntentAgent** - Sentiment analysis and buying readiness
- **InteractionDecisionAgent** - Strategic workflow decisions
- **QuestionGenerationAgent** - Context-aware clarifying questions
- **PricingStrategyAgent** - Intelligent pricing strategies
- **EvaluationIntelligenceAgent** - Performance monitoring
- **CustomerResponseAgent** - Customer simulation and testing
- **RFQOrchestrator** - Core workflow coordination

#### Specialized Domain Agents (4 agents)
- **CompetitiveIntelligenceAgent** - Market positioning and win probability
- **RiskAssessmentAgent** - 10-point risk scoring across 5 categories
- **ContractTermsAgent** - Legal terms and compliance management
- **ProposalWriterAgent** - Professional document generation

#### Evaluation & Quality Assurance
- **BestOfNSelector** - Multiple candidate generation with LLM judge evaluation
- **LLM Judge System** - Structured scoring across accuracy, completeness, relevance, clarity
- **Confidence Scoring** - Score distribution analysis and quality metrics

