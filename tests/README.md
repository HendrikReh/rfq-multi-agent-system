# Test Suite Documentation

This directory contains the organized test suite for the Enhanced Multi-Agent RFQ System.

## Directory Structure

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
├── run_all_tests.py       # Comprehensive test runner
└── README.md              # This file
```

## Test Categories

### Unit Tests (`unit/`)
- **Agent Tests**: Individual agent functionality
  - `test_enhanced_agents.py` - Core enhanced agents
  - `test_model_assignment.py` - Model assignment logic
- **Model Tests**: Data validation and serialization
- **Utility Tests**: Helper functions and utilities

### Integration Tests (`integration/`)
- **Multi-Agent Workflows**: End-to-end agent coordination
  - `test_evaluations.py` - Agent output quality evaluation
  - `test_verification.py` - System verification tests
  - `test_scenario_recording.py` - Scenario recording and playback
  - `test_simple.py` - Basic integration tests

### Performance Tests (`performance/`)
- **Load Testing**: System performance under load
  - `test_performance.py` - Parallel execution and scalability
- **Resource Usage**: Memory and CPU utilization
- **Response Time**: Latency measurements

## Running Tests

### Quick Start
```bash
# Run all tests from project root
python test_runner.py

# Run from tests directory
cd tests && python run_all_tests.py
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests only
pytest tests/performance/ -v

# Specific test file
pytest tests/unit/test_agents/test_enhanced_agents.py -v
```

### Test Configuration

The test suite is configured to:
- **Disable real model requests**: `ALLOW_MODEL_REQUESTS = False`
- **Use test models**: PydanticAI `TestModel` and `FunctionModel`
- **Mock external dependencies**: API calls, file system, etc.
- **Provide comprehensive reporting**: Success rates, timing, recommendations

## Test Environment Setup

Tests automatically configure the environment:
```python
# Set in conftest.py and test files
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
models.ALLOW_MODEL_REQUESTS = False
```

## Test Data

Test fixtures are organized in the `fixtures/` directory:
- Sample RFQ requirements
- Mock customer data
- Expected output examples
- Performance benchmarks

## Adding New Tests

### Unit Tests
1. Create test file in appropriate `unit/` subdirectory
2. Follow naming convention: `test_*.py`
3. Use `TestModel` for agent testing
4. Include docstrings and assertions

### Integration Tests
1. Add to `integration/` directory
2. Test multi-component workflows
3. Use realistic test scenarios
4. Verify end-to-end functionality

### Performance Tests
1. Add to `performance/` directory
2. Include timing measurements
3. Test with various load levels
4. Document performance expectations

## Test Best Practices

### PydanticAI Testing
- Use `TestModel` for fast, deterministic testing
- Use `FunctionModel` for custom response logic
- Override agents with test models using `agent.override(model=TestModel())`
- Validate output structure and types

### Async Testing
- Mark async tests with `@pytest.mark.anyio`
- Use `await` for agent calls
- Handle async context properly

### Mocking
- Mock external API calls
- Use `unittest.mock` for dependencies
- Keep mocks simple and focused

## Continuous Integration

The test suite is designed for CI/CD integration:
- Fast execution with test models
- Comprehensive reporting
- Clear success/failure indicators
- Detailed error messages

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure Python path includes project root
2. **Model Requests**: Verify `ALLOW_MODEL_REQUESTS = False`
3. **Async Errors**: Use proper async/await syntax
4. **Path Issues**: Run tests from correct directory

### Getting Help
- Check test output for specific error messages
- Review test logs in detail
- Verify test environment setup
- Ensure all dependencies are installed 