#!/usr/bin/env python3
"""
Demonstrate Model Assignment Logic

This script demonstrates exactly how each agent uses its specific environment variable
for model configuration. It shows the complete flow from .env to agent instantiation.
"""

import os
from agents.utils import get_model_name, get_all_agent_models

def demonstrate_flow():
    """Demonstrate the complete flow from environment variables to agent models."""
    
    print("🔍 DEMONSTRATING MODEL ASSIGNMENT LOGIC")
    print("=" * 60)
    
    # Show the logic step by step
    print("\n📋 STEP 1: Environment Variable to Agent Type Mapping")
    print("-" * 50)
    
    mappings = {
        "pricing_strategy": "RFQ_PRICING_STRATEGY_MODEL",
        "question_generation": "RFQ_QUESTION_GENERATION_MODEL", 
        "customer_response": "RFQ_CUSTOMER_RESPONSE_MODEL",
        "rfq_parser": "RFQ_RFQ_PARSER_MODEL",
        "conversation_state": "RFQ_CONVERSATION_STATE_MODEL",
        "customer_intent": "RFQ_CUSTOMER_INTENT_MODEL",
        "interaction_decision": "RFQ_INTERACTION_DECISION_MODEL",
        "evaluation_intelligence": "RFQ_EVALUATION_INTELLIGENCE_MODEL",
        "rfq_orchestrator": "RFQ_RFQ_ORCHESTRATOR_MODEL"
    }
    
    for agent_type, env_var in mappings.items():
        print(f"  {agent_type:20} ← {env_var}")
    
    print("\n🔧 STEP 2: Setting Custom Environment Variables")
    print("-" * 50)
    
    # Set some custom models
    os.environ['RFQ_PRICING_STRATEGY_MODEL'] = 'openai:gpt-4o-mini'
    os.environ['RFQ_QUESTION_GENERATION_MODEL'] = 'openai:gpt-4-turbo'
    print("  export RFQ_PRICING_STRATEGY_MODEL='openai:gpt-4o-mini'")
    print("  export RFQ_QUESTION_GENERATION_MODEL='openai:gpt-4-turbo'")
    print("  (Other agents will use defaults)")
    
    print("\n⚙️ STEP 3: How get_model_name() Works")
    print("-" * 50)
    
    print("  1. Agent calls: get_model_name('pricing_strategy')")
    print("  2. Function creates env var: 'RFQ_' + 'PRICING_STRATEGY' + '_MODEL'")
    print("  3. Function checks: os.getenv('RFQ_PRICING_STRATEGY_MODEL')")
    print("  4. If found: returns custom model")
    print("  5. If not found: returns default model")
    
    print("\n🤖 STEP 4: Actual Model Resolution")
    print("-" * 50)
    
    # Clear cache to reload environment variables
    from agents.utils import CUSTOM_AGENT_MODELS
    CUSTOM_AGENT_MODELS.clear()
    
    for agent_type in mappings.keys():
        model = get_model_name(agent_type)
        env_var = mappings[agent_type]
        custom_value = os.environ.get(env_var)
        
        if custom_value:
            print(f"🔧 {agent_type:20} → {model} (custom from {env_var})")
        else:
            print(f"⚙️ {agent_type:20} → {model} (default)")
    
    print("\n📝 STEP 5: Agent Class Implementation")
    print("-" * 50)
    
    print("  In agents/pricing_strategy_agent.py:")
    print("    Line 17: model_name = get_model_name('pricing_strategy')")
    print("    Line 18: self.agent = Agent(model_name, ...)")
    print("")
    print("  In agents/question_generation_agent.py:")
    print("    Line 19: model_name = get_model_name('question_generation')")
    print("    Line 20: self.agent = Agent(model_name, ...)")
    print("")
    print("  Each agent follows the same pattern:")
    print("    1. Call get_model_name() with their specific agent_type")
    print("    2. Use the returned model name to create PydanticAI Agent")
    print("    3. This ensures each agent uses its specific environment variable")
    
    print("\n✅ STEP 6: Verification")
    print("-" * 50)
    
    # Show that the logic works
    pricing_model = get_model_name("pricing_strategy")
    question_model = get_model_name("question_generation") 
    parser_model = get_model_name("rfq_parser")
    
    print(f"  PricingStrategyAgent will use: {pricing_model}")
    print(f"  QuestionGenerationAgent will use: {question_model}")
    print(f"  RFQParser will use: {parser_model}")
    
    # Clean up
    del os.environ['RFQ_PRICING_STRATEGY_MODEL']
    del os.environ['RFQ_QUESTION_GENERATION_MODEL']
    
    print("\n🎯 SUMMARY")
    print("-" * 50)
    print("✅ Each agent type maps to a specific environment variable")
    print("✅ Each agent class calls get_model_name() with its agent_type")
    print("✅ get_model_name() checks the corresponding environment variable")
    print("✅ If set: uses custom model, if not: uses optimized default")
    print("✅ This ensures complete flexibility and proper model assignment")


def show_agent_source_code():
    """Show the exact source code lines where agents use their models."""
    
    print("\n📄 EXACT SOURCE CODE IMPLEMENTATION")
    print("=" * 60)
    
    agents_info = [
        ("PricingStrategyAgent", "agents/pricing_strategy_agent.py", 17, "pricing_strategy"),
        ("QuestionGenerationAgent", "agents/question_generation_agent.py", 19, "question_generation"),
        ("CustomerResponseAgent", "agents/customer_response_agent.py", 20, "customer_response"),
        ("RFQParser", "agents/rfq_parser.py", 16, "rfq_parser"),
        ("ConversationStateAgent", "agents/conversation_state_agent.py", 16, "conversation_state")
    ]
    
    for agent_name, file_path, line_num, agent_type in agents_info:
        env_var = f"RFQ_{agent_type.upper()}_MODEL"
        print(f"\n{agent_name}:")
        print(f"  File: {file_path}")
        print(f"  Line {line_num}: model_name = get_model_name('{agent_type}')")
        print(f"  Environment Variable: {env_var}")
        print(f"  Result: Uses model from {env_var} if set, otherwise default")


if __name__ == "__main__":
    # Set dummy API key for demonstration
    os.environ['OPENAI_API_KEY'] = 'demo-key'
    
    demonstrate_flow()
    show_agent_source_code()
    
    print("\n🚀 To test this yourself:")
    print("  1. Set: export RFQ_PRICING_STRATEGY_MODEL='openai:gpt-4o-mini'")
    print("  2. Run: python show_model_config.py")
    print("  3. See that PricingStrategyAgent uses your custom model") 