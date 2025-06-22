# RFQ Multi-Agent System

A production-ready multi-agent system for Request for Quote (RFQ) processing built with [PydanticAI](https://ai.pydantic.dev/), featuring FastAPI integration and MCP support.

## 🚀 Overview

This system demonstrates modern multi-agent architecture patterns inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system), providing:

- **17+ Specialized Agents** for comprehensive RFQ processing
- **Modular Architecture** with clear separation of concerns
- **Production Orchestration** with parallel execution and health monitoring
- **FastAPI Web Service** with complete REST API
- **MCP Integration** ready for Model Context Protocol
- **Comprehensive Testing** with unit, integration, and performance tests
- **🔥 Logfire Observability** with complete LLM conversation tracing and performance monitoring

## 🏗️ Architecture

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

api/                              # FastAPI web service
mcp_server/                       # MCP Server implementation
tests/                            # Comprehensive test suite
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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rfc-pydanticai-openai

# Install with uv (recommended)
uv install

# Or with pip
pip install -e .
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # optional

# Set up Logfire observability (optional but recommended)
uv run logfire auth
uv run logfire projects use  # or create new project
```

### Basic Usage

```python
from rfq_system import RFQProcessingResult
from rfq_system.orchestration.coordinators.parallel import ParallelCoordinator

# Initialize the system
coordinator = ParallelCoordinator()

# Process an RFQ
result = await coordinator.process_rfq(
    customer_request="We need a custom CRM system for 100 users",
    execution_mode="parallel"
)

print(f"RFQ Status: {result.status}")
print(f"Confidence: {result.confidence_score}")
```

### Best-of-N Selection Usage

```python
from rfq_system.agents.evaluation.best_of_n_selector import BestOfNSelector
from rfq_system.agents.proposal_writer_agent import ProposalWriterAgent

# Initialize Best-of-N selector
selector = BestOfNSelector()
proposal_agent = ProposalWriterAgent()

# Generate multiple candidates and select the best
result = await selector.generate_best_of_n(
    target_agent=proposal_agent,
    prompt="Create a comprehensive RFQ proposal for enterprise CRM",
    context={"budget_range": "$100k-500k", "timeline": "6 months"},
    n=5,  # Generate 5 candidates
    evaluation_criteria={
        "accuracy": 0.3,      # Technical accuracy weight
        "completeness": 0.3,  # Comprehensive coverage
        "relevance": 0.2,     # Customer relevance
        "clarity": 0.2        # Communication clarity
    }
)

print(f"Best candidate score: {result.best_score}")
print(f"Confidence: {result.confidence}")
print(f"Selected proposal: {result.best_candidate}")
```

### FastAPI Web Service

```bash
# Start the API server
uv run -m api.main

# Or using the CLI
rfq-api

# Access the API documentation
open http://localhost:8000/docs
```

### MCP Server

```bash
# Start the MCP server
uv run -m mcp_server.server

# Or using the CLI
rfq-mcp-server
```

## 📊 Multi-Agent Orchestration

### Execution Modes

#### Parallel Execution
```python
from rfq_system.orchestration.coordinators.parallel import ParallelCoordinator, ParallelTask

coordinator = ParallelCoordinator(max_concurrent_tasks=10)

# Create parallel tasks
tasks = [
    ParallelTask("competitive_analysis", competitive_agent, requirements, context),
    ParallelTask("risk_assessment", risk_agent, requirements, context),
    ParallelTask("contract_terms", contract_agent, requirements, context)
]

# Execute in parallel
results = await coordinator.execute_parallel_tasks(tasks)
```

#### Sequential Coordination
```python
from rfq_system.orchestration.coordinators.sequential import SequentialCoordinator

coordinator = SequentialCoordinator()
result = await coordinator.process_workflow(requirements, agents)
```

#### Graph-Based Control Flow
```python
from rfq_system.orchestration.coordinators.graph_based import GraphBasedCoordinator

coordinator = GraphBasedCoordinator()
await coordinator.execute_state_machine(initial_state, state_graph)
```

## 🔥 Logfire Observability

### Complete LLM Conversation Tracing

This system includes comprehensive [Pydantic Logfire](https://logfire.pydantic.dev/) integration for production-grade observability of your multi-agent system:

#### Features
- **LLM Conversation Tracing**: See every message exchanged between agents and models
- **Performance Monitoring**: Track response times, token usage, and costs
- **Error Tracking**: Comprehensive error logging with context
- **Agent Delegation Visibility**: Trace how agents call other agents
- **Best-of-N Selection Monitoring**: Detailed evaluation process tracking
- **Custom Spans and Metrics**: Business-specific observability

#### Setup

```bash
# Install with Logfire support (already included)
uv add 'pydantic-ai[logfire]'

# Authenticate with Logfire
uv run logfire auth

# Create or use existing project
uv run logfire projects new  # or 'logfire projects use'
```

#### Usage

The system automatically instruments all PydanticAI agents when Logfire is configured:

```python
import logfire
from pydantic_ai import Agent

# Configure Logfire (automatically done in our system)
logfire.configure()

# Instrument all PydanticAI agents for detailed conversation logging
logfire.instrument_pydantic_ai(event_mode='logs')
Agent.instrument_all()

# All agent interactions are now traced!
result = await coordinator.process_rfq("Build a CRM system")
```

#### What You'll See in Logfire Dashboard

1. **Agent Execution Spans**
   - Individual agent processing times
   - Parallel execution visualization
   - Error handling and retries

2. **LLM Conversations**
   - System prompts sent to models
   - User messages from agents
   - Model responses with token counts
   - Tool calls and responses

3. **Best-of-N Selection Process**
   - Candidate generation timing
   - LLM judge evaluation details
   - Selection reasoning and confidence scores

4. **Business Metrics**
   - RFQ processing success rates
   - Customer intent analysis results
   - Risk assessment scores
   - Proposal generation quality

#### Example Logfire Integration

```python
# The demo script includes comprehensive Logfire tracing
from examples.demo_real_llm_evaluation import main

# Run with Logfire tracing enabled
await main()  # Automatically traces all LLM interactions

# Check your dashboard at: https://logfire.pydantic.dev/your-project
```

#### Data Privacy & Scrubbing

Logfire automatically scrubs sensitive data:
- API keys and authentication tokens
- Personal identifiable information (PII)
- Credit card numbers and sensitive patterns

The `reasoning` field in evaluation results may show `[Scrubbed due to 'auth']` - this is normal and protects sensitive LLM-generated content while preserving all numerical metrics.

#### Dashboard URL
Access your traces at: `https://logfire.pydantic.dev/your-project-name`

## 🔧 Configuration

### Agent Configuration
```python
from rfq_system.core.interfaces.agent import BaseAgent, AgentCapability

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="custom_agent",
            capabilities=[AgentCapability.PROCESS_RFQ]
        )
    
    async def process(self, input_data, context):
        # Custom processing logic
        return result
```

### Model Configuration
```bash
# Environment variables for model selection
export RFQ_PARSER_MODEL="gpt-4o"
export CUSTOMER_INTENT_MODEL="gpt-4o-mini"
export PRICING_STRATEGY_MODEL="claude-3-sonnet"
```

## 🧪 Testing

### Run All Tests
```bash
# Using pytest
pytest

# Using the test runner
uv run run_all_tests.py

# Performance tests (asyncio only, no trio dependency)
uv run pytest tests/performance/ -m "not slow"
```

### Test Categories
- **Unit Tests**: Individual agent functionality
- **Integration Tests**: Multi-agent workflows
- **Performance Tests**: Load and scalability testing
- **Contract Tests**: API and MCP interface compliance
- **Evaluation Tests**: Best-of-N selector and LLM judge testing
- **Real LLM Tests**: Production testing with actual API calls and cost management

### Best-of-N Selector Testing

The system includes comprehensive testing for the Best-of-N selector and LLM judge functionality:

#### Quick Testing
```bash
# Run all Best-of-N tests
OPENAI_API_KEY=test-key uv run pytest tests/unit/test_best_of_n_selector.py -v

# Standalone evaluation script
OPENAI_API_KEY=test-key uv run python tests/evaluation/test_best_of_n_simple.py

# Interactive demo with real models
OPENAI_API_KEY=your-real-key uv run python examples/demo_best_of_n_selection.py

# Real LLM evaluation tests (uses actual API calls)
OPENAI_API_KEY=your-real-key uv run python tests/evaluation/test_best_of_n_real_llm.py

# Interactive real LLM evaluation demo
OPENAI_API_KEY=your-real-key uv run python examples/demo_real_llm_evaluation.py

# Run with pytest (marked as slow test)
OPENAI_API_KEY=your-real-key uv run pytest tests/evaluation/test_best_of_n_real_llm.py -v -s -m slow
```

#### Real LLM Evaluation Features
- **PydanticEvals Integration**: Official evaluation framework with structured datasets
- **Multiple Test Scenarios**: Enterprise CRM ($100k-300k), Startup MVP ($25k-75k), Healthcare Compliance
- **Custom Quality Evaluators**: BestOfNQualityEvaluator and ProposalQualityEvaluator
- **LLM Judge Assessment**: Real LLM evaluation with domain-specific rubrics
- **Cost Management**: Uses gpt-4o-mini, limits parallel calls, provides cost warnings
- **Safety Features**: API key validation, user confirmation prompts, automatic skipping
- **Comprehensive Output**: Detailed analysis with scores, reasoning, and proposal metrics
- **Accurate Timing**: Custom timing measurements (PydanticEvals duration display disabled due to bug)

#### Testing Features
- ✅ **Multiple candidate generation** with parallel execution
- ✅ **LLM judge evaluation** with structured scoring (accuracy, completeness, relevance, clarity)
- ✅ **Custom evaluation criteria** with configurable weights
- ✅ **Confidence scoring** based on score distribution
- ✅ **Performance testing** (3000+ candidates/second with TestModel)
- ✅ **Error handling and timeouts** with graceful degradation
- ✅ **Real LLM evaluation** with actual API calls using PydanticEvals framework
- ✅ **Production testing** with cost management and safety features
- ✅ **Multiple test scenarios** (Enterprise, Startup, Healthcare)
- ✅ **Custom evaluators** for proposal quality assessment

#### Testing Patterns
```python
# Use TestModel for fast, deterministic tests
with selector._judge_agent.override(model=TestModel()):
    with selector._selection_agent.override(model=TestModel()):
        result = await selector.generate_best_of_n(
            target_agent=mock_agent,
            prompt="Generate RFQ proposal",
            context=context,
            n=5
        )

# Use FunctionModel for controlled evaluation responses
def mock_judge_evaluation(messages, info):
    return {
        "overall_score": 0.85,
        "reasoning": "High-quality proposal with comprehensive features"
    }

with selector._judge_agent.override(model=FunctionModel(mock_judge_evaluation)):
    result = await selector.generate_best_of_n(...)
```

### Test with TestModel
```python
from pydantic_ai.models.test import TestModel
from rfq_system.agents.core.rfq_parser import RFQParser

# Test without API calls
agent = RFQParser()
with agent.override(model=TestModel()):
    result = await agent.process(test_input, context)
```

## 📚 API Documentation

### REST Endpoints

#### Process RFQ
```bash
POST /api/v1/process
Content-Type: application/json

{
  "customer_request": "We need a CRM system",
  "priority": "high",
  "execution_mode": "parallel"
}
```

#### Health Checks
```bash
GET /health                    # Basic health check
GET /health/detailed           # Detailed system health
GET /health/agents            # Agent-specific health
GET /health/ready             # Readiness probe
GET /health/live              # Liveness probe
```

#### Metrics
```bash
GET /metrics                  # Prometheus-style metrics
```

### Agent Management
```bash
GET /api/v1/agents           # List all agents
GET /api/v1/agents/{id}/health  # Agent health status
POST /api/v1/agents/{id}/restart # Restart agent
```

## 🔄 Migration from Old Structure

If you have an existing installation with the old flat structure:

### Automated Migration
```bash
# Preview migration (dry run)
python scripts/migrate.py migrate --dry-run

# Perform migration with backup
python scripts/migrate.py migrate

# Check migration status
python scripts/migrate.py status

# Clean up old files after verification
python scripts/migrate.py cleanup
```

### Manual Migration Steps
1. **Backup**: Create backup of current installation
2. **Structure**: Update to new directory structure
3. **Imports**: Update import statements in custom code
4. **Config**: Update configuration files
5. **Tests**: Run tests to verify functionality

## 🏥 Monitoring & Observability

### Health Monitoring
```python
from rfq_system.monitoring.health.system_health import SystemHealthMonitor

monitor = SystemHealthMonitor()
health_report = await monitor.get_comprehensive_health()
```

### Performance Metrics
```python
from rfq_system.monitoring.metrics.performance import PerformanceTracker

tracker = PerformanceTracker()
metrics = tracker.get_system_metrics()
```

### Distributed Tracing
```python
from rfq_system.monitoring.tracing.tracer import DistributedTracer

tracer = DistributedTracer()
with tracer.trace("rfq_processing") as span:
    result = await process_rfq(request)
```

## 🔌 MCP Integration

### Server Implementation
```python
from mcp_server.server import RFQMCPServer

server = RFQMCPServer()
await server.start()
```

### Tool Definitions
```python
from mcp_server.tools.rfq_tools import process_rfq_tool

@process_rfq_tool
async def process_customer_request(request: str) -> dict:
    """Process a customer RFQ request."""
    return await rfq_system.process(request)
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.12-slim

COPY . /app
WORKDIR /app

RUN pip install -e .

CMD ["python", "-m", "api.main"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rfq-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rfq-system
  template:
    metadata:
      labels:
        app: rfq-system
    spec:
      containers:
      - name: rfq-api
        image: rfq-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
```

### Environment Variables
```bash
# Core configuration
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
LOG_LEVEL=INFO
ENVIRONMENT=production

# Performance tuning
MAX_CONCURRENT_TASKS=10
DEFAULT_TIMEOUT=30.0
ENABLE_HEALTH_MONITORING=true

# Model configuration
RFQ_PARSER_MODEL=gpt-4o
CUSTOMER_INTENT_MODEL=gpt-4o-mini
```

## 📈 Performance Optimization

### Parallel Execution
- 3-5x speed improvement with parallel agent coordination
- Configurable concurrency limits
- Intelligent task prioritization

### Model Selection
- Optimized model assignments per agent type
- Cost vs. performance trade-offs
- Environment-based overrides

### Caching & Memory
- Intelligent caching of agent results
- Conversation memory for context retention
- Performance metrics tracking

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
uv install --dev

# Install pre-commit hooks
pre-commit install

# Run code quality checks
ruff check .
black .
mypy .
```

### Adding New Agents
1. Inherit from `BaseAgent` or `SpecializedAgent`
2. Implement required abstract methods
3. Add to agent registry
4. Write comprehensive tests
5. Update documentation

### Testing Guidelines
- Write unit tests for all agent functionality
- Include integration tests for multi-agent workflows
- Add performance tests for critical paths
- Use TestModel for API-free testing

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review examples in `/examples`
### **Competitive Intelligence**
- Market positioning analysis
- Competitor threat assessment
- Win probability calculation
- Differentiation strategy recommendations

### **Risk Assessment**
- 10-point risk scoring system
- 5 risk categories: Financial, Operational, Customer, Project, Market
- Mitigation strategy development
- Go/no-go recommendations

### **Contract & Legal**
- Payment and delivery terms optimization
- Compliance requirement identification
- Liability limitation strategies
- Industry-specific legal considerations

### **Professional Proposals**
- Executive summary generation
- Technical approach documentation
- Pricing justification and ROI analysis
- Implementation timeline development

## System Architecture

The system implements all four levels of [PydanticAI multi-agent complexity](https://ai.pydantic.dev/multi-agent-applications/):

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATED RFQ SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│  Enhanced Orchestrator (Agent Delegation + Parallel Execution) │
├─────────────────────────────────────────────────────────────────┤
│  Core Agents        │  Specialized Agents  │  Memory & Control  │
│  ─────────────      │  ─────────────────   │  ───────────────   │
│  • RFQ Parser       │  • Competitive Intel │  • Memory Agent    │
│  • Intent Analysis  │  • Risk Assessment   │  • Graph Controller│
│  • Decision Making  │  • Contract Terms    │  • Health Monitor  │
│  • Question Gen     │  • Proposal Writer   │  • Performance Opt │
│  • Pricing Strategy │                      │                    │
├─────────────────────────────────────────────────────────────────┤
│            Customer Request → Comprehensive Analysis            │
│                         ↓                                       │
│    Parallel Processing → Risk + Competitive + Contract         │
│                         ↓                                       │
│              Professional Proposal + Quote                     │
└─────────────────────────────────────────────────────────────────┘
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rfc-pydanticai-openai
   ```

2. **Install dependencies with uv**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

4. **Optional: Customize agent models** (see [Model Configuration](#model-configuration)):
   ```bash
   export RFQ_COMPETITIVE_INTELLIGENCE_MODEL='openai:gpt-4o'
   export RFQ_RISK_ASSESSMENT_MODEL='openai:gpt-4o-mini'
   ```

## Usage Examples

### **Comprehensive System Demo**
Experience the full power of 13+ agents working together:
```bash
python demo_integrated_system.py
```

### **Basic RFQ Processing**
Quick demonstration of core functionality:
```bash
python main.py
```

### **Complete Flow Simulation**
End-to-end customer interaction simulation:
```bash
python main.py --complete
```

### **Advanced Integration Examples**

**Parallel Execution Mode** (fastest processing):
```python
from agents.integration_framework import IntegratedRFQSystem

system = IntegratedRFQSystem()
result = await system.process_rfq_comprehensive(
    customer_message="Enterprise software for 500 users...",
    deps=deps,
    execution_mode="parallel",
    include_competitive_analysis=True,
    include_risk_assessment=True,
    include_contract_terms=True,
    include_proposal=True
)
```

**Sequential Execution Mode** (dependency management):
```python
result = await system.process_rfq_comprehensive(
    customer_message="Government contract with strict compliance...",
    deps=deps,
    execution_mode="sequential",  # For complex dependencies
    include_all=True
)
```

## Testing

The system includes comprehensive test coverage with organized test structure:

### **Quick Start**
```bash
# Run all tests (recommended)
python test_runner.py

# Run from tests directory
cd tests && python run_all_tests.py
```

### **Test Categories**
```bash
# Unit tests - individual agent functionality
pytest tests/unit/ -v

# Integration tests - multi-agent workflows  
pytest tests/integration/ -v

# Performance tests - scalability and load testing
pytest tests/performance/ -v

# Specific test files
pytest tests/unit/test_agents/test_enhanced_agents.py -v
```

### **Test Structure**
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agents/        # Agent-specific unit tests
│   ├── test_models/        # Data model tests
│   └── test_utils/         # Utility function tests
├── integration/            # Integration tests for multi-component workflows
├── performance/            # Performance and scalability tests
├── fixtures/               # Test data and fixtures
├── conftest.py            # Pytest configuration and shared fixtures
└── run_all_tests.py       # Comprehensive test runner
```

### **Test Features**
- **API-Free Testing**: Uses PydanticAI TestModel for fast, deterministic tests
- **Comprehensive Coverage**: 15+ test scenarios covering all agent types
- **Performance Monitoring**: Load testing and parallel execution validation
- **Quality Evaluation**: Structure validation and multi-agent workflow testing
- **CI/CD Ready**: Fast execution with clear success/failure reporting

## Performance & Monitoring

### **System Health Monitoring**
```python
health_report = await system.get_system_health_report()
print(f"System Status: {health_report.overall_status}")
print(f"Healthy Agents: {health_report.healthy_agents}/{health_report.total_agents}")
```

### **Performance Optimization**
```python
optimization = await system.optimize_system_performance()
print(f"Success Rate: {optimization['key_metrics']['success_rate']:.1%}")
print(f"Avg Response Time: {optimization['key_metrics']['avg_response_time']:.2f}s")
```

### **Confidence Scoring**
The system provides multi-factor confidence scores:
- Base confidence from requirements completeness
- Competitive analysis win probability
- Risk-adjusted confidence factors
- Overall system recommendation

## ⚙️ Model Configuration

### **Intelligent Defaults**
Each agent uses optimized models based on task complexity:
- **Complex Reasoning** (gpt-4o): Competitive Intelligence, Risk Assessment, Proposal Writing
- **Efficient Processing** (gpt-4o-mini): State Tracking, Performance Evaluation

### **Environment Variable Overrides**
Customize any agent's model:
```bash
# Cost optimization
export RFQ_RISK_ASSESSMENT_MODEL='openai:gpt-4o-mini'

# Quality optimization
export RFQ_COMPETITIVE_INTELLIGENCE_MODEL='openai:gpt-4o'

# View current configuration
python show_model_config.py
```

### **Agent Model Mapping**
| Agent Type | Environment Variable | Default Model |
|------------|---------------------|---------------|
| `competitive_intelligence` | `RFQ_COMPETITIVE_INTELLIGENCE_MODEL` | `openai:gpt-4o` |
| `risk_assessment` | `RFQ_RISK_ASSESSMENT_MODEL` | `openai:gpt-4o-mini` |
| `contract_terms` | `RFQ_CONTRACT_TERMS_MODEL` | `openai:gpt-4o` |
| `proposal_writer` | `RFQ_PROPOSAL_WRITER_MODEL` | `openai:gpt-4o` |
| `rfq_orchestrator` | `RFQ_RFQ_ORCHESTRATOR_MODEL` | `openai:gpt-4o` |

## Scenario Recording & Analysis

### **Automatic Scenario Recording**
Every interaction is automatically recorded with comprehensive metadata:
```bash
# View all recorded scenarios
python view_scenarios.py

# Analyze specific scenario
python view_scenarios.py --details reports/20250621_225323_scenario_3.json

# Performance analysis across scenarios
python view_scenarios.py --analyze
```

### **Recorded Data Includes**
- Complete conversation flow
- All agent processing results
- Model usage and performance metrics
- Confidence scores and recommendations
- Error information and recovery actions

## Demo Scenarios

The system includes comprehensive demo scenarios:

### **Enterprise Software RFQ**
- High-value deal ($2M annually)
- Complex requirements (500 users, global offices)
- Full analysis pipeline demonstration

### **Startup MVP Development**
- Competitive market scenario
- Budget constraints and strategic positioning
- Negotiation and value proposition focus

### **Government Contract**
- High-risk, regulated environment
- Compliance and security requirements
- Formal procurement process simulation

## Advanced Features

### **Execution Modes**
- **Parallel**: Maximum speed with concurrent agent execution
- **Sequential**: Dependency management for complex scenarios
- **Selective**: Choose specific agents based on requirements

### **Error Recovery**
- Graceful degradation when agents fail
- Health monitoring with automatic recovery
- Comprehensive error logging and analysis

### **Extensibility**
- Easy addition of new specialized agents
- Modular architecture for custom workflows
- Plugin system for industry-specific requirements

## Documentation & Learning

### **PydanticAI Integration**
This system showcases advanced [PydanticAI patterns](https://ai.pydantic.dev/multi-agent-applications/):
- Agent delegation via tools
- Programmatic agent hand-off
- Graph-based control flow
- Multi-agent orchestration

### **Evaluation Framework**
Built-in evaluation using [PydanticEvals](https://ai.pydantic.dev/evals/):
- LLM judge evaluation
- Performance benchmarking
- Quality assessment across scenarios

### **Testing & Validation**
Comprehensive testing framework:
```bash
python test_model_assignment.py      # Model configuration tests
python test_scenario_recording.py    # Recording functionality tests
python demonstrate_model_logic.py    # Model assignment demonstration
```

## Production Deployment

### **Ready for Production**
- Health monitoring and alerting
- Performance optimization
- Error recovery and logging
- Scalable architecture

### **Cost Optimization**
- Intelligent model selection per agent
- Parallel execution reduces total time
- Configurable agent inclusion based on scenario

### **Quality Assurance**
- Multi-factor confidence scoring
- Comprehensive validation
- Continuous performance monitoring

## Next Steps

The system provides a foundation for:
1. **Industry-Specific Customization**: Add domain experts for healthcare, finance, etc.
2. **Integration Expansion**: Connect with CRM, ERP, and procurement systems
3. **Advanced Analytics**: ML-based optimization and predictive analytics
4. **Multi-Language Support**: Extend to global markets and languages

## License

This project demonstrates advanced multi-agent LLM capabilities using PydanticAI. See the documentation for implementation details and best practices.

---

**Built with ❤️ using [PydanticAI](https://ai.pydantic.dev/) - The Production-Ready AI Framework**
