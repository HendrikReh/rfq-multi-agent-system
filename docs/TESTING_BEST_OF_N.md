# Testing the Best-of-N Selector and LLM Judge

This document provides comprehensive guidance on testing the `BestOfNSelector` and LLM judge functionality in the RFQ Multi-Agent System.

## Overview

The Best-of-N selection system includes several testable components:

- **BestOfNSelector**: Core selection logic with parallel candidate generation
- **LLM Judge**: Structured evaluation with configurable criteria
- **Evaluation Criteria**: Customizable weights for different quality aspects
- **Confidence Scoring**: Selection confidence based on score distribution

## Testing Approaches

### 1. Unit Tests with pytest

The primary testing approach uses pytest with `TestModel` and `FunctionModel` to avoid API calls:

```bash
# Run all Best-of-N tests
OPENAI_API_KEY=test-key uv run pytest tests/unit/test_best_of_n_evaluation.py -v

# Run specific test categories
uv run pytest tests/unit/test_best_of_n_evaluation.py::TestBestOfNSelector -v
uv run pytest tests/unit/test_best_of_n_evaluation.py::TestEvaluationCriteria -v
```

### 2. Simple Evaluation Script

For quick testing without pytest overhead:

```bash
# Run standalone evaluation
OPENAI_API_KEY=test-key uv run python tests/evaluation/test_best_of_n_simple.py
```

### 3. Interactive Demo

For hands-on testing with real models:

```bash
# Run interactive demo (requires real API key)
OPENAI_API_KEY=your-real-key uv run python examples/demo_best_of_n_selection.py
```

## Test Categories

### Core Functionality Tests

#### Basic Best-of-N Generation
```python
async def test_basic_best_of_n_generation():
    """Test that the selector generates multiple candidates and selects the best."""
    mock_agent = MockRFQAgent(mixed_quality_proposals)
    
    with selector._judge_agent.override(model=TestModel()):
        with selector._selection_agent.override(model=TestModel()):
            result = await selector.generate_best_of_n(
                target_agent=mock_agent,
                prompt="Generate RFQ proposal",
                context=agent_context,
                n=3
            )
    
    # Validates: candidate count, unique IDs, confidence scoring
    assert result.n_candidates == 3
    assert len(result.all_evaluations) == 3
    assert 0.0 <= result.selection_confidence <= 1.0
```

#### LLM Judge with Controlled Responses
```python
async def test_llm_judge_with_function_model():
    """Test LLM judge using FunctionModel for predictable evaluation."""
    
    def mock_judge_evaluation(messages, info: AgentInfo):
        prompt = str(messages[-1].parts[0].content)
        
        if "comprehensive" in prompt.lower():
            return {
                "overall_score": 0.85,
                "reasoning": "High-quality proposal with comprehensive features",
                # ... other scores
            }
        else:
            return {
                "overall_score": 0.45,
                "reasoning": "Basic proposal lacking detail",
                # ... other scores
            }
    
    with selector._judge_agent.override(model=FunctionModel(mock_judge_evaluation)):
        result = await selector.generate_best_of_n(...)
    
    # Validates: structured evaluation, reasoning, score ranges
```

### Advanced Testing Scenarios

#### Custom Evaluation Criteria
```python
async def test_custom_evaluation_criteria():
    """Test that custom criteria weights are applied correctly."""
    
    # Emphasize completeness and relevance over accuracy
    custom_criteria = EvaluationCriteria(
        accuracy_weight=0.15,
        completeness_weight=0.40,
        relevance_weight=0.35,
        clarity_weight=0.10
    )
    
    result = await selector.generate_best_of_n(
        criteria=custom_criteria,
        # ... other params
    )
    
    # The system should use these weights in evaluation
```

#### Confidence Scoring Validation
```python
async def test_confidence_scoring():
    """Test that confidence reflects score distribution quality."""
    
    def mock_judge_with_varied_scores(messages, info: AgentInfo):
        # Return different scores to test confidence calculation
        if "candidate_0" in prompt:
            return {"overall_score": 0.9}  # High score
        elif "candidate_1" in prompt:
            return {"overall_score": 0.5}  # Medium score
        else:
            return {"overall_score": 0.2}  # Low score
    
    # With varied scores, confidence should be reasonable
    # With all similar scores, confidence should be lower
```

#### Performance and Timing
```python
async def test_parallel_generation_performance():
    """Test that parallel generation performs efficiently."""
    
    start_time = time.time()
    result = await selector.generate_best_of_n(n=5)
    execution_time = time.time() - start_time
    
    # Should be fast with TestModel
    assert execution_time < 1.0
    candidates_per_second = result.n_candidates / execution_time
    assert candidates_per_second > 10
```

### Error Handling Tests

#### Timeout Behavior
```python
async def test_error_handling_and_timeouts():
    """Test graceful handling of timeouts and errors."""
    
    deps = BestOfNDependencies(
        generation_timeout=0.1,  # Very short timeout
        evaluation_timeout=0.1
    )
    
    # Should still work with TestModel (fast execution)
    result = await selector.generate_best_of_n(deps=deps)
    assert isinstance(result, BestOfNResult)
```

## Testing Strategies

### 1. Mock Agent Patterns

Create agents that return predictable responses for testing:

```python
class MockRFQAgent(BaseAgent):
    def __init__(self, responses: List[RFQProposal]):
        self.responses = responses
        self.call_count = 0
    
    async def process(self, input_data, context):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
```

### 2. Quality-Controlled Testing

Use different quality levels to test selection logic:

```python
# High-quality responses
high_quality_proposals = [
    RFQProposal(
        title="Enterprise CRM Solution with Advanced Analytics",
        description="Comprehensive system with advanced features...",
        key_features=["Advanced analytics", "Workflow automation", ...]
    )
]

# Low-quality responses  
low_quality_proposals = [
    RFQProposal(
        title="Software",
        description="We can build it.",
        key_features=["Software"]
    )
]
```

### 3. Model Override Patterns

Use `TestModel` for deterministic testing:

```python
# For fast, deterministic tests
with selector._judge_agent.override(model=TestModel()):
    with selector._selection_agent.override(model=TestModel()):
        result = await selector.generate_best_of_n(...)
```

Use `FunctionModel` for controlled responses:

```python
# For predictable evaluation responses
def mock_evaluation(messages, info):
    return {"overall_score": 0.8, "reasoning": "Good proposal"}

with selector._judge_agent.override(model=FunctionModel(mock_evaluation)):
    result = await selector.generate_best_of_n(...)
```

## Testing Best Practices

### 1. Environment Setup

Always set up the environment properly for testing:

```python
import os
from pydantic_ai import models

# Prevent accidental API calls
models.ALLOW_MODEL_REQUESTS = False
os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-evaluation')
```

### 2. Assertion Patterns

Use comprehensive assertions to validate results:

```python
# Validate structure
assert isinstance(result, BestOfNResult)
assert result.n_candidates == expected_count
assert len(result.all_evaluations) == expected_count

# Validate content quality
assert result.best_candidate is not None
assert result.best_evaluation is not None
assert all(eval.reasoning for eval in result.all_evaluations)

# Validate score ranges
for evaluation in result.all_evaluations:
    assert 0.0 <= evaluation.overall_score <= 1.0
    assert 0.0 <= evaluation.accuracy_score <= 1.0
    # ... other score validations

# Validate selection logic
best_score = result.best_evaluation.overall_score
other_scores = [e.overall_score for e in result.all_evaluations 
               if e.candidate_id != result.best_candidate.candidate_id]
if other_scores:
    assert best_score >= max(other_scores)
```

### 3. Fixtures and Reusability

Use pytest fixtures for common test data:

```python
@pytest.fixture
def mixed_quality_proposals():
    """Mixed quality proposals for testing selection."""
    return [high_quality_proposal, medium_quality_proposal, low_quality_proposal]

@pytest.fixture
def best_of_n_selector():
    """Configured selector for testing."""
    return BestOfNSelector(
        evaluation_model="openai:gpt-4o-mini",
        max_parallel_generations=5,
        enable_detailed_evaluation=True
    )
```

## Integration with Existing Tests

### Running with Test Suite

The Best-of-N tests integrate with your existing test infrastructure:

```bash
# Run all tests including Best-of-N
uv run python test_runner.py

# Run specific test categories
uv run pytest tests/unit/ -k "best_of_n"
uv run pytest tests/unit/test_best_of_n_evaluation.py
```

### CI/CD Integration

The tests are designed to work in CI environments:

- No external API dependencies when using `TestModel`
- Fast execution (< 1 second for full test suite)
- Deterministic results for reliable CI builds
- Proper environment variable handling

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in the Python path
2. **API Key Errors**: Set `OPENAI_API_KEY` environment variable for fixture creation
3. **Model Request Errors**: Use `models.ALLOW_MODEL_REQUESTS = False` for testing
4. **Async Issues**: Use `pytest.mark.asyncio` for async test functions

### Debug Tips

1. **Verbose Output**: Use `-v` flag with pytest for detailed test output
2. **Specific Tests**: Run individual test methods for focused debugging
3. **Print Debugging**: Add print statements in mock functions to trace execution
4. **Model Responses**: Inspect `FunctionModel` calls to understand evaluation flow

## Real-World Testing

For testing with actual LLM models (requires API keys):

```python
# Set real API key
os.environ['OPENAI_API_KEY'] = 'your-real-api-key'

# Allow model requests
models.ALLOW_MODEL_REQUESTS = True

# Run without model overrides
result = await selector.generate_best_of_n(
    target_agent=real_agent,
    prompt="Generate enterprise CRM proposal",
    context=context,
    n=5
)

# Analyze real evaluation quality
print(f"Selection confidence: {result.selection_confidence}")
for eval in result.all_evaluations:
    print(f"Candidate {eval.candidate_id}: {eval.overall_score:.2f}")
    print(f"Reasoning: {eval.reasoning}")
```

## Performance Benchmarking

For performance testing:

```python
import time
import statistics

# Run multiple iterations
times = []
for _ in range(10):
    start = time.time()
    result = await selector.generate_best_of_n(n=5)
    times.append(time.time() - start)

print(f"Average time: {statistics.mean(times):.3f}s")
print(f"Std deviation: {statistics.stdev(times):.3f}s")
print(f"Candidates/second: {5 / statistics.mean(times):.1f}")
```

This comprehensive testing approach ensures the Best-of-N selector works reliably across different scenarios and provides high-quality candidate selection for your RFQ system. 