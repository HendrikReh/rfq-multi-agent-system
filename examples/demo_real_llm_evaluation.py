#!/usr/bin/env python3
"""
Real LLM Evaluation Demo with Model Selection

This script demonstrates how to run real LLM evaluation tests for the Best-of-N selector
with interactive model selection for each agent. Choose from OpenAI's model series
with intelligent defaults based on task complexity.

WARNING: This requires a real OpenAI API key and will incur costs.

NOTE: PydanticEvals v0.3.2 has a duration reporting bug (always shows 1.0s).
Duration display is disabled; accurate timing is provided in the detailed analysis.

Features:
- Interactive model selection for each agent type
- OpenAI model series support (GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo)
- Intelligent defaults based on task complexity
- Real LLM API calls for Best-of-N candidate generation
- LLM judge evaluation with structured scoring
- Comprehensive JSON report generation
- Performance analysis with accurate timing
- Cost management and safety features

Usage:
    OPENAI_API_KEY=your-real-key python examples/demo_real_llm_evaluation.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pydantic_ai import models


# OpenAI Model Options with descriptions
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Fast and economical, good for simpler tasks",
        "cost": "Very Low",
        "speed": "Very Fast",
        "use_case": "Simple tasks, maximum speed/cost efficiency"
    },
    "gpt-4": {
        "name": "GPT-4",
        "description": "Original GPT-4, high capability",
        "cost": "High",
        "speed": "Slow",
        "use_case": "Complex reasoning, high-quality outputs"
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "Previous generation, still very capable",
        "cost": "High",
        "speed": "Medium",
        "use_case": "Complex reasoning, legacy compatibility"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Most capable model, best for complex reasoning",
        "cost": "High",
        "speed": "Medium",
        "use_case": "Complex analysis, high-quality outputs"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini", 
        "description": "Efficient model, good balance of capability and cost",
        "cost": "Low",
        "speed": "Fast",
        "use_case": "Most tasks, cost-effective choice"
    },
    "o1-mini": {
        "name": "o1 Mini",
        "description": "Reasoning-focused model, mini version",
        "cost": "Medium",
        "speed": "Medium",
        "use_case": "Complex reasoning tasks, cost-effective"
    },
    "o1-preview": {
        "name": "o1 Preview",
        "description": "Advanced reasoning model (preview version)",
        "cost": "Very High",
        "speed": "Slow",
        "use_case": "Most complex reasoning, research tasks"
    },
    "gpt-4.1": {
        "name": "GPT-4.1",
        "description": "Future GPT-4 version (may not be available)",
        "cost": "High",
        "speed": "Medium",
        "use_case": "Next-generation capabilities"
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "description": "Future mini version (may not be available)",
        "cost": "Medium",
        "speed": "Fast",
        "use_case": "Future efficient processing"
    },
    "gpt-4.1-nano": {
        "name": "GPT-4.1 Nano",
        "description": "Future nano version (may not be available)",
        "cost": "Low",
        "speed": "Very Fast",
        "use_case": "Future ultra-efficient tasks"
    },
    "o3": {
        "name": "o3",
        "description": "Next-generation reasoning model (may not be available)",
        "cost": "Very High",
        "speed": "Slow",
        "use_case": "Advanced reasoning and research"
    },
    "o3-mini": {
        "name": "o3 Mini",
        "description": "Efficient reasoning model (may not be available)",
        "cost": "Medium",
        "speed": "Medium",
        "use_case": "Reasoning tasks, cost-effective"
    },
    "o4-mini": {
        "name": "o4 Mini",
        "description": "Future reasoning model (may not be available)",
        "cost": "Medium",
        "speed": "Medium",
        "use_case": "Future reasoning capabilities"
    }
}

# Agent Types with intelligent defaults and descriptions
AGENT_CONFIGS = {
    "target_agent": {
        "name": "Target Agent (RFQ Proposal Generator)",
        "description": "Generates RFQ proposals for evaluation",
        "default": "gpt-4o-mini",
        "recommendation": "gpt-4o-mini for balanced quality/cost, gpt-4o for highest quality"
    },
    "evaluation_judge": {
        "name": "Evaluation Judge (LLM Judge)",
        "description": "Evaluates and scores generated proposals",
        "default": "gpt-4o-mini", 
        "recommendation": "gpt-4o-mini is sufficient for evaluation tasks"
    },
    "selection_agent": {
        "name": "Selection Agent (Best Candidate Picker)",
        "description": "Selects the best candidate from evaluations",
        "default": "gpt-4o-mini",
        "recommendation": "gpt-4o-mini works well for selection logic"
    }
}


def display_model_options():
    """Display available OpenAI models with details."""
    print("\n📋 Available OpenAI Models:")
    print("=" * 60)
    
    for model_id, info in OPENAI_MODELS.items():
        print(f"\n🤖 {info['name']} ({model_id})")
        print(f"   📝 {info['description']}")
        print(f"   💰 Cost: {info['cost']} | ⚡ Speed: {info['speed']}")
        print(f"   🎯 Best for: {info['use_case']}")


def get_model_choice(agent_type: str, config: Dict) -> str:
    """Get user's model choice for a specific agent."""
    print(f"\n🔧 Configure {config['name']}")
    print("=" * 50)
    print(f"📖 Purpose: {config['description']}")
    print(f"💡 Recommendation: {config['recommendation']}")
    print(f"⚙️  Default: {OPENAI_MODELS[config['default']]['name']} ({config['default']})")
    
    print(f"\n📋 Model Options:")
    for i, (model_id, info) in enumerate(OPENAI_MODELS.items(), 1):
        default_marker = " (DEFAULT)" if model_id == config['default'] else ""
        print(f"   {i}. {info['name']} ({model_id}){default_marker}")
        print(f"      💰 {info['cost']} cost | ⚡ {info['speed']} speed")
    
    while True:
        choice = input(f"\nSelect model (1-{len(OPENAI_MODELS)}) or press Enter for default: ").strip()
        
        if not choice:  # Use default
            return config['default']
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(OPENAI_MODELS):
                selected_model = list(OPENAI_MODELS.keys())[choice_num - 1]
                print(f"✅ Selected: {OPENAI_MODELS[selected_model]['name']} ({selected_model})")
                return selected_model
            else:
                print(f"❌ Please enter a number between 1 and {len(OPENAI_MODELS)}")
        except ValueError:
            print("❌ Please enter a valid number or press Enter for default")


def configure_models() -> Dict[str, str]:
    """Interactive model configuration for all agents."""
    print("\n🎛️  Model Configuration")
    print("=" * 60)
    print("Choose models for each agent type. Defaults are optimized for")
    print("balanced performance and cost. You can customize based on your needs.")
    
    # Show model options first
    display_model_options()
    
    # Get choices for each agent
    model_config = {}
    for agent_type, config in AGENT_CONFIGS.items():
        model_config[agent_type] = get_model_choice(agent_type, config)
    
    return model_config


def display_configuration_summary(model_config: Dict[str, str]):
    """Display the final model configuration."""
    print("\n📊 Final Model Configuration")
    print("=" * 60)
    
    total_cost_level = 0
    cost_levels = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4}
    
    for agent_type, model_id in model_config.items():
        config = AGENT_CONFIGS[agent_type]
        model_info = OPENAI_MODELS[model_id]
        
        print(f"\n🤖 {config['name']}")
        print(f"   Model: {model_info['name']} ({model_id})")
        print(f"   Cost: {model_info['cost']} | Speed: {model_info['speed']}")
        
        total_cost_level += cost_levels.get(model_info['cost'], 2)
    
    # Cost estimation
    avg_cost_level = total_cost_level / len(model_config)
    if avg_cost_level <= 1.5:
        cost_estimate = "💚 Very Low"
    elif avg_cost_level <= 2.5:
        cost_estimate = "💛 Low-Medium"
    elif avg_cost_level <= 3.5:
        cost_estimate = "🟠 Medium-High"
    else:
        cost_estimate = "🔴 High"
    
    print(f"\n💰 Overall Cost Estimate: {cost_estimate}")
    print(f"📈 This configuration will run 3 test cases with 3 candidates each")
    print(f"🔢 Total API calls: ~15-20 calls (proposal generation + evaluation)")


async def run_evaluation_with_models(model_config: Dict[str, str]):
    """Run the evaluation with the selected model configuration."""
    print(f"\n🚀 Starting Real LLM Best-of-N Evaluation")
    print("⚠️  This will make actual API calls and incur costs")
    print("=" * 60)
    
    try:
        # Import the real LLM test module
        sys.path.append(str(Path(__file__).parent.parent / "tests" / "evaluation"))
        from test_best_of_n_real_llm import run_real_llm_evaluation_with_models
        
        # Enable real model requests
        models.ALLOW_MODEL_REQUESTS = True
        
        # Run the evaluation with custom models
        report_filepath = await run_real_llm_evaluation_with_models(model_config)
        
        return report_filepath
        
    except ImportError:
        # Fallback to the original function if the new one doesn't exist
        print("📝 Using default model configuration...")
        from test_best_of_n_real_llm import run_real_llm_evaluation
        return await run_real_llm_evaluation()


async def main():
    """Run the real LLM evaluation demo with model selection."""
    
    print("🚀 Real LLM Evaluation Demo for Best-of-N Selector")
    print("🎛️  Interactive Model Selection Edition")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'test-key':
        print("❌ Real OpenAI API key required!")
        print("   Set your API key: export OPENAI_API_KEY=your-real-key")
        print("   Then run: python examples/demo_real_llm_evaluation.py")
        return
    
    print(f"✅ API key found: {api_key[:8]}...")
    print("⚠️  WARNING: This will make real API calls and incur costs!")
    print("⚠️  NOTE: PydanticEvals v0.3.2 duration reporting disabled due to bug")
    print("    Accurate timing measurements are included in the detailed output.")
    print("📊 This demo will generate a comprehensive JSON report in ./reports")
    
    # Model configuration
    print("\n🎯 Would you like to:")
    print("   1. Use intelligent defaults (recommended)")
    print("   2. Configure models interactively")
    
    config_choice = input("\nChoice (1-2): ").strip()
    
    if config_choice == "2":
        model_config = configure_models()
    else:
        print("\n⚙️  Using intelligent defaults...")
        model_config = {agent_type: config['default'] for agent_type, config in AGENT_CONFIGS.items()}
    
    # Display final configuration
    display_configuration_summary(model_config)
    
    # Ask for confirmation
    response = input("\nProceed with this configuration? (y/N): ")
    if response.lower() != 'y':
        print("❌ Evaluation cancelled.")
        return
    
    print("\n🔄 Starting real LLM evaluation...")
    
    # Run the evaluation
    try:
        report_filepath = await run_evaluation_with_models(model_config)
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📋 Comprehensive report generated:")
        print(f"   📁 {report_filepath}")
        print(f"\n💡 You can now:")
        print(f"   • View the JSON report: cat {report_filepath}")
        print(f"   • Compare with existing reports in ./reports")
        print(f"   • Analyze performance metrics and evaluation scores")
        print(f"   • Use the report data for further analysis")
        
        # Display model configuration in report
        print(f"\n🎛️  Model Configuration Used:")
        for agent_type, model_id in model_config.items():
            agent_name = AGENT_CONFIGS[agent_type]['name']
            model_name = OPENAI_MODELS[model_id]['name']
            print(f"   • {agent_name}: {model_name} ({model_id})")
        
    except ImportError as e:
        print(f"❌ Failed to import evaluation module: {e}")
        print("   Make sure you're running from the project root directory.")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("   Check the error report in ./reports for details.")
        raise
    finally:
        # Reset model requests
        models.ALLOW_MODEL_REQUESTS = False
    
    print("\n✅ Real LLM evaluation demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 