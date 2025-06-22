"""
Shared utilities for RFQ processing agents.

This module contains common utility functions used across different agents,
including flexible model configuration for optimal agent performance.
"""

import os
import sys
from typing import Dict, Literal

# Type definition for agent types
AgentType = Literal[
    "rfq_parser",
    "conversation_state", 
    "customer_intent",
    "interaction_decision",
    "question_generation",
    "pricing_strategy",
    "evaluation_intelligence",
    "customer_response",
    "rfq_orchestrator"
]

# Default model configuration optimized for each agent's specific tasks
DEFAULT_AGENT_MODELS: Dict[AgentType, str] = {
    # High-precision parsing requires strong reasoning
    "rfq_parser": "openai:gpt-4o",
    
    # Simple state tracking can use efficient model
    "conversation_state": "openai:gpt-4o-mini",
    
    # Complex intent analysis benefits from advanced reasoning
    "customer_intent": "openai:gpt-4o",
    
    # Strategic decision making requires sophisticated reasoning
    "interaction_decision": "openai:gpt-4o",
    
    # Creative question generation benefits from advanced model
    "question_generation": "openai:gpt-4o",
    
    # Complex pricing strategy requires strong analytical capabilities
    "pricing_strategy": "openai:gpt-4o",
    
    # Performance evaluation can use efficient model
    "evaluation_intelligence": "openai:gpt-4o-mini",
    
    # Customer simulation requires creative and contextual responses
    "customer_response": "openai:gpt-4o",
    
    # Main orchestrator coordinates everything, needs strong reasoning
    "rfq_orchestrator": "openai:gpt-4o"
}

# Custom model overrides from environment variables
CUSTOM_AGENT_MODELS: Dict[AgentType, str] = {}


def _load_custom_models() -> None:
    """Load custom model configurations from environment variables."""
    global CUSTOM_AGENT_MODELS
    
    for agent_type in DEFAULT_AGENT_MODELS.keys():
        env_var = f"RFQ_{agent_type.upper()}_MODEL"
        custom_model = os.getenv(env_var)
        if custom_model:
            CUSTOM_AGENT_MODELS[agent_type] = custom_model


def get_model_name(agent_type: AgentType = "rfq_orchestrator") -> str:
    """
    Get the appropriate OpenAI model for a specific agent type.
    
    Args:
        agent_type: The type of agent requesting a model
        
    Returns:
        str: OpenAI model name optimized for the agent's tasks
        
    Raises:
        SystemExit: If OPENAI_API_KEY is not found
        
    Examples:
        # Use default optimized models
        parser_model = get_model_name("rfq_parser")  # Returns "openai:gpt-4o"
        state_model = get_model_name("conversation_state")  # Returns "openai:gpt-4o-mini"
        
        # Override with environment variables
        # export RFQ_PRICING_STRATEGY_MODEL="openai:gpt-4o-mini"
        pricing_model = get_model_name("pricing_strategy")  # Returns custom model
    """
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Load custom models on first call
    if not CUSTOM_AGENT_MODELS:
        _load_custom_models()
    
    # Return custom model if specified, otherwise default
    return CUSTOM_AGENT_MODELS.get(agent_type, DEFAULT_AGENT_MODELS[agent_type])


def get_all_agent_models() -> Dict[AgentType, str]:
    """
    Get the complete mapping of agent types to their configured models.
    
    Returns:
        Dict mapping agent types to their OpenAI model names
    """
    if not CUSTOM_AGENT_MODELS:
        _load_custom_models()
    
    result = DEFAULT_AGENT_MODELS.copy()
    result.update(CUSTOM_AGENT_MODELS)
    return result


def print_model_configuration() -> None:
    """Print the current model configuration for all agents."""
    print("🤖 Current Agent Model Configuration:")
    print("=" * 50)
    
    models = get_all_agent_models()
    for agent_type, model in models.items():
        is_custom = agent_type in CUSTOM_AGENT_MODELS
        marker = "🔧" if is_custom else "⚙️"
        print(f"{marker} {agent_type:20} → {model}")
    
    print("\n💡 To customize models, set environment variables:")
    print("   export RFQ_<AGENT_TYPE>_MODEL='openai:model-name'")
    print("   Example: export RFQ_PRICING_STRATEGY_MODEL='openai:gpt-4o-mini'") 