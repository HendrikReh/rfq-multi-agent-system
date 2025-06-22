# Examples Directory

This directory contains demonstration scripts and usage examples for the Enhanced Multi-Agent RFQ System.

## Demo Scripts

### **Comprehensive System Demo**
**File**: `demo_integrated_system.py`
**Purpose**: Experience the full power of 13+ agents working together

```bash
python examples/demo_integrated_system.py
```

**Features**:
- Complete multi-agent workflow demonstration
- Parallel execution of specialized agents
- Health monitoring and performance metrics
- Comprehensive RFQ processing with all agent types

### **Complete Flow Simulation**
**File**: `demo_complete_flow.py`
**Purpose**: End-to-end customer interaction simulation

```bash
python examples/demo_complete_flow.py
```

**Features**:
- Full conversational workflows
- Multiple customer personas (startup, enterprise, SMB)
- Question refinement and adaptive responses
- Scenario recording and analytics

### **Model Logic Demonstration**
**File**: `demonstrate_model_logic.py`
**Purpose**: Shows model assignment and decision-making logic

```bash
python examples/demonstrate_model_logic.py
```

**Features**:
- Model configuration display
- Agent-to-model mapping demonstration
- Cost optimization strategies
- Performance vs. cost trade-offs

## Utility Scripts

### **Model Configuration Viewer**
**File**: `show_model_config.py`
**Purpose**: Display current model assignments for all agents

```bash
python examples/show_model_config.py
```

**Output**:
- Current model assignments per agent
- Environment variable overrides
- Cost implications
- Performance characteristics

### **Scenario Analytics**
**File**: `view_scenarios.py`
**Purpose**: Analyze recorded RFQ scenarios

```bash
# List all scenarios
python examples/view_scenarios.py

# View detailed scenario information
python examples/view_scenarios.py --details reports/scenario_file.json

# Analyze performance across scenarios
python examples/view_scenarios.py --analyze
```

**Features**:
- Scenario listing and summary
- Detailed conversation flow analysis
- Performance analytics and insights
- Error scenario diagnostics

## Usage Patterns

### **Getting Started**
1. Start with the basic demo:
   ```bash
   python examples/demo_integrated_system.py
   ```

2. Explore complete workflows:
   ```bash
   python examples/demo_complete_flow.py
   ```

3. Analyze results:
   ```bash
   python examples/view_scenarios.py --analyze
   ```

### **Development & Testing**
- Use `demonstrate_model_logic.py` to understand model assignments
- Use `show_model_config.py` to verify environment configuration
- Use `demo_integrated_system.py` for comprehensive system testing

### **Business Demonstration**
- Run `demo_complete_flow.py` for stakeholder presentations
- Use `view_scenarios.py` to show analytics and insights
- Demonstrate different customer personas and scenarios

## Environment Setup

All examples require:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

Optional model customization:
```bash
export RFQ_COMPETITIVE_INTELLIGENCE_MODEL='openai:gpt-4o'
export RFQ_RISK_ASSESSMENT_MODEL='openai:gpt-4o-mini'
```

## Output and Reports

Examples generate various outputs:
- **Console output**: Real-time processing information
- **JSON reports**: Scenario recordings in `reports/` directory
- **Analytics data**: Performance metrics and insights
- **System health**: Agent status and performance monitoring

## Customization

You can modify the examples to:
- Add new customer personas
- Test different RFQ scenarios
- Experiment with model configurations
- Customize output formats
- Add new analytics metrics 