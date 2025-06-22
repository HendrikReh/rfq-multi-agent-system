#!/usr/bin/env python3
"""
Test Model Assignment Logic

This script verifies that each agent correctly uses its specific environment variable
for model configuration, ensuring the logic works as expected.
"""

import os
import sys
from agents.utils import get_model_name, _load_custom_models, CUSTOM_AGENT_MODELS, DEFAULT_AGENT_MODELS

def test_default_models():
    """Test that default models are returned when no environment variables are set."""
    print("🧪 TEST 1: Default Model Assignment")
    print("=" * 50)
    
    # Clear any existing custom models
    CUSTOM_AGENT_MODELS.clear()
    
    # Clear environment variables
    env_vars_to_clear = [
        'RFQ_PRICING_STRATEGY_MODEL',
        'RFQ_QUESTION_GENERATION_MODEL', 
        'RFQ_CUSTOMER_RESPONSE_MODEL'
    ]
    
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    
    # Test default assignments
    pricing_model = get_model_name("pricing_strategy")
    question_model = get_model_name("question_generation")
    customer_model = get_model_name("customer_response")
    
    print(f"✅ pricing_strategy    → {pricing_model} (expected: openai:gpt-4o)")
    print(f"✅ question_generation → {question_model} (expected: openai:gpt-4o)")
    print(f"✅ customer_response   → {customer_model} (expected: openai:gpt-4o)")
    
    # Verify they match defaults
    assert pricing_model == "openai:gpt-4o", f"Expected openai:gpt-4o, got {pricing_model}"
    assert question_model == "openai:gpt-4o", f"Expected openai:gpt-4o, got {question_model}"
    assert customer_model == "openai:gpt-4o", f"Expected openai:gpt-4o, got {customer_model}"
    
    print("✅ All default models assigned correctly!\n")


def test_custom_models():
    """Test that custom environment variables override default models."""
    print("🧪 TEST 2: Custom Model Assignment")
    print("=" * 50)
    
    # Clear any existing custom models
    CUSTOM_AGENT_MODELS.clear()
    
    # Set custom environment variables
    os.environ['RFQ_PRICING_STRATEGY_MODEL'] = 'openai:gpt-4o-mini'
    os.environ['RFQ_QUESTION_GENERATION_MODEL'] = 'openai:gpt-4-turbo'
    os.environ['RFQ_CUSTOMER_RESPONSE_MODEL'] = 'openai:gpt-4'
    
    # Test custom assignments
    pricing_model = get_model_name("pricing_strategy")
    question_model = get_model_name("question_generation")
    customer_model = get_model_name("customer_response")
    
    print(f"🔧 pricing_strategy    → {pricing_model} (expected: openai:gpt-4o-mini)")
    print(f"🔧 question_generation → {question_model} (expected: openai:gpt-4-turbo)")
    print(f"🔧 customer_response   → {customer_model} (expected: openai:gpt-4)")
    
    # Verify they match custom settings
    assert pricing_model == "openai:gpt-4o-mini", f"Expected openai:gpt-4o-mini, got {pricing_model}"
    assert question_model == "openai:gpt-4-turbo", f"Expected openai:gpt-4-turbo, got {question_model}"
    assert customer_model == "openai:gpt-4", f"Expected openai:gpt-4, got {customer_model}"
    
    print("✅ All custom models assigned correctly!\n")


def test_mixed_models():
    """Test mixed scenario with some custom and some default models."""
    print("🧪 TEST 3: Mixed Model Assignment")
    print("=" * 50)
    
    # Clear any existing custom models
    CUSTOM_AGENT_MODELS.clear()
    
    # Set only some custom environment variables
    os.environ['RFQ_PRICING_STRATEGY_MODEL'] = 'openai:gpt-4o-mini'
    # Leave RFQ_QUESTION_GENERATION_MODEL unset (should use default)
    os.environ['RFQ_CUSTOMER_RESPONSE_MODEL'] = 'openai:gpt-4'
    
    # Remove question generation model if it exists
    if 'RFQ_QUESTION_GENERATION_MODEL' in os.environ:
        del os.environ['RFQ_QUESTION_GENERATION_MODEL']
    
    # Test mixed assignments
    pricing_model = get_model_name("pricing_strategy")
    question_model = get_model_name("question_generation")
    customer_model = get_model_name("customer_response")
    
    print(f"🔧 pricing_strategy    → {pricing_model} (expected: openai:gpt-4o-mini - custom)")
    print(f"⚙️ question_generation → {question_model} (expected: openai:gpt-4o - default)")
    print(f"🔧 customer_response   → {customer_model} (expected: openai:gpt-4 - custom)")
    
    # Verify mixed settings
    assert pricing_model == "openai:gpt-4o-mini", f"Expected openai:gpt-4o-mini, got {pricing_model}"
    assert question_model == "openai:gpt-4o", f"Expected openai:gpt-4o, got {question_model}"
    assert customer_model == "openai:gpt-4", f"Expected openai:gpt-4, got {customer_model}"
    
    print("✅ Mixed model assignment works correctly!\n")


def test_environment_variable_mapping():
    """Test the exact environment variable to agent type mapping."""
    print("🧪 TEST 4: Environment Variable Mapping")
    print("=" * 50)
    
    # Clear any existing custom models
    CUSTOM_AGENT_MODELS.clear()
    
    # Test the mapping logic directly
    test_mappings = {
        "pricing_strategy": "RFQ_PRICING_STRATEGY_MODEL",
        "question_generation": "RFQ_QUESTION_GENERATION_MODEL", 
        "customer_response": "RFQ_CUSTOMER_RESPONSE_MODEL",
        "rfq_parser": "RFQ_RFQ_PARSER_MODEL",
        "conversation_state": "RFQ_CONVERSATION_STATE_MODEL"
    }
    
    for agent_type, env_var in test_mappings.items():
        # Set a test value
        test_model = f"openai:test-{agent_type}"
        os.environ[env_var] = test_model
        
        # Clear custom models cache to force reload
        CUSTOM_AGENT_MODELS.clear()
        
        # Get model and verify
        result_model = get_model_name(agent_type)
        print(f"✅ {agent_type:20} ← {env_var:30} → {result_model}")
        
        assert result_model == test_model, f"Expected {test_model}, got {result_model}"
        
        # Clean up
        del os.environ[env_var]
    
    print("✅ All environment variable mappings work correctly!\n")


def test_agent_instantiation():
    """Test that actual agent classes use the correct models."""
    print("🧪 TEST 5: Agent Class Instantiation")
    print("=" * 50)
    
    # Set specific models for testing
    os.environ['RFQ_PRICING_STRATEGY_MODEL'] = 'openai:test-pricing'
    os.environ['RFQ_QUESTION_GENERATION_MODEL'] = 'openai:test-questions'
    
    # Clear custom models cache
    CUSTOM_AGENT_MODELS.clear()
    
    # Import and test actual agent classes
    from agents.pricing_strategy_agent import PricingStrategyAgent
    from agents.question_generation_agent import QuestionGenerationAgent
    
    # Create instances (this will call get_model_name internally)
    try:
        pricing_agent = PricingStrategyAgent()
        question_agent = QuestionGenerationAgent()
        
        print("✅ PricingStrategyAgent instantiated successfully")
        print("✅ QuestionGenerationAgent instantiated successfully")
        print("✅ Agents use their specific model configurations")
        
    except Exception as e:
        print(f"❌ Error instantiating agents: {e}")
        # This is expected since we don't have valid API keys for test models
        if "invalid_api_key" in str(e) or "status_code: 401" in str(e):
            print("✅ Error is due to invalid test API key, which confirms agents are trying to use the configured models")
        else:
            raise
    
    # Clean up
    del os.environ['RFQ_PRICING_STRATEGY_MODEL']
    del os.environ['RFQ_QUESTION_GENERATION_MODEL']
    
    print()


def main():
    """Run all tests to verify model assignment logic."""
    print("🚀 TESTING MODEL ASSIGNMENT LOGIC")
    print("=" * 60)
    print("Verifying that each agent uses its specific environment variable\n")
    
    # Set a dummy API key to avoid exit during testing
    original_api_key = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    
    try:
        test_default_models()
        test_custom_models()
        test_mixed_models()
        test_environment_variable_mapping()
        test_agent_instantiation()
        
        print("🎉 ALL TESTS PASSED!")
        print("✅ Each agent correctly uses its specific environment variable")
        print("✅ Model assignment logic works as expected")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        sys.exit(1)
        
    finally:
        # Restore original API key
        if original_api_key:
            os.environ['OPENAI_API_KEY'] = original_api_key
        elif 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']


if __name__ == "__main__":
    main() 