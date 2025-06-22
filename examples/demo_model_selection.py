#!/usr/bin/env python3
"""
Interactive Model Selection Demo

This script demonstrates the interactive model selection interface for the
Best-of-N evaluation system without requiring real API calls.

Features:
- Interactive model selection for each agent type
- Cost estimation and configuration summary
- User-friendly interface with clear recommendations
- No API calls required - pure demonstration

Usage:
    python examples/demo_model_selection.py
"""

import sys
from pathlib import Path

# Add the examples directory to path for imports
sys.path.append(str(Path(__file__).parent))

from demo_real_llm_evaluation import (
    display_model_options,
    configure_models,
    display_configuration_summary,
    OPENAI_MODELS,
    AGENT_CONFIGS
)


def demo_model_selection_interface():
    """Demonstrate the interactive model selection interface."""
    
    print("🎛️  Interactive Model Selection Demo")
    print("=" * 60)
    print("This demo shows the model selection interface for Best-of-N evaluation")
    print("without requiring real API calls or keys.")
    print()
    
    # Show available models
    display_model_options()
    
    print("\n🎯 Demo Mode: Interactive Model Configuration")
    print("=" * 60)
    print("In the real demo, you would choose between:")
    print("   1. Use intelligent defaults (recommended)")
    print("   2. Configure models interactively")
    print()
    print("For this demo, we'll show the interactive configuration...")
    
    # Show agent configurations
    print("\n🔧 Agent Configuration Options")
    print("=" * 50)
    
    for agent_type, config in AGENT_CONFIGS.items():
        print(f"\n🤖 {config['name']}")
        print(f"   📖 Purpose: {config['description']}")
        print(f"   💡 Recommendation: {config['recommendation']}")
        print(f"   ⚙️  Default: {OPENAI_MODELS[config['default']]['name']} ({config['default']})")
        
        # Show model options for this agent
        print(f"   📋 Available Models:")
        for i, (model_id, info) in enumerate(OPENAI_MODELS.items(), 1):
            default_marker = " (DEFAULT)" if model_id == config['default'] else ""
            print(f"      {i}. {info['name']} ({model_id}){default_marker}")
            print(f"         💰 {info['cost']} cost | ⚡ {info['speed']} speed")
            print(f"         🎯 {info['use_case']}")
    
    # Simulate different configurations
    print("\n📊 Example Configurations")
    print("=" * 50)
    
    # Configuration 1: Cost-optimized
    print("\n💚 Configuration 1: Cost-Optimized")
    cost_config = {
        "target_agent": "gpt-3.5-turbo",
        "evaluation_judge": "gpt-4o-mini", 
        "selection_agent": "gpt-4o-mini"
    }
    display_configuration_summary(cost_config)
    
    # Configuration 2: Quality-optimized
    print("\n🔴 Configuration 2: Quality-Optimized")
    quality_config = {
        "target_agent": "gpt-4o",
        "evaluation_judge": "gpt-4o",
        "selection_agent": "gpt-4o"
    }
    display_configuration_summary(quality_config)
    
    # Configuration 3: Balanced (defaults)
    print("\n💛 Configuration 3: Balanced (Defaults)")
    balanced_config = {
        "target_agent": "gpt-4o-mini",
        "evaluation_judge": "gpt-4o-mini",
        "selection_agent": "gpt-4o-mini"
    }
    display_configuration_summary(balanced_config)
    
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("To use the real interactive model selection:")
    print("   OPENAI_API_KEY=your-real-key python examples/demo_real_llm_evaluation.py")
    print()
    print("Features shown:")
    print("   ✅ Model descriptions with cost/speed information")
    print("   ✅ Agent-specific recommendations")
    print("   ✅ Configuration summaries with cost estimates")
    print("   ✅ User-friendly selection interface")
    print("   ✅ Intelligent defaults for different use cases")


def show_model_comparison():
    """Show a detailed comparison of OpenAI models."""
    
    print("\n📋 Detailed Model Comparison")
    print("=" * 60)
    
    # Table header
    print(f"{'Model':<15} {'Cost':<10} {'Speed':<12} {'Best Use Case':<40}")
    print("=" * 80)
    
    # Model rows
    for model_id, info in OPENAI_MODELS.items():
        print(f"{model_id:<15} {info['cost']:<10} {info['speed']:<12} {info['use_case']:<40}")
    
    print("\n💡 Recommendations by Use Case:")
    print("   🚀 Maximum Speed: GPT-3.5 Turbo")
    print("   💰 Lowest Cost: GPT-3.5 Turbo")
    print("   ⚖️  Best Balance: GPT-4o Mini")
    print("   🎯 Highest Quality: GPT-4o")
    print("   🔄 Legacy Support: GPT-4 Turbo")


def show_cost_analysis():
    """Show cost analysis for different configurations."""
    
    print("\n💰 Cost Analysis")
    print("=" * 60)
    
    configurations = {
        "Maximum Economy": {
            "target_agent": "gpt-3.5-turbo",
            "evaluation_judge": "gpt-3.5-turbo",
            "selection_agent": "gpt-3.5-turbo"
        },
        "Balanced Performance": {
            "target_agent": "gpt-4o-mini",
            "evaluation_judge": "gpt-4o-mini", 
            "selection_agent": "gpt-4o-mini"
        },
        "Premium Quality": {
            "target_agent": "gpt-4o",
            "evaluation_judge": "gpt-4o",
            "selection_agent": "gpt-4o"
        },
        "Hybrid Approach": {
            "target_agent": "gpt-4o",
            "evaluation_judge": "gpt-4o-mini",
            "selection_agent": "gpt-4o-mini"
        }
    }
    
    for config_name, config in configurations.items():
        print(f"\n📊 {config_name}")
        print("-" * 40)
        for agent_type, model_id in config.items():
            agent_name = AGENT_CONFIGS[agent_type]['name']
            model_info = OPENAI_MODELS[model_id]
            print(f"   {agent_name}: {model_info['name']}")
            print(f"      Cost: {model_info['cost']} | Speed: {model_info['speed']}")
        
        # Calculate rough cost level
        cost_levels = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4}
        total_cost = sum(cost_levels.get(OPENAI_MODELS[model]['cost'], 2) for model in config.values())
        avg_cost = total_cost / len(config)
        
        if avg_cost <= 1.5:
            cost_estimate = "💚 Very Low"
        elif avg_cost <= 2.5:
            cost_estimate = "💛 Low-Medium"
        elif avg_cost <= 3.5:
            cost_estimate = "🟠 Medium-High"
        else:
            cost_estimate = "🔴 High"
        
        print(f"   Overall Cost: {cost_estimate}")


if __name__ == "__main__":
    print("🚀 Best-of-N Evaluation: Interactive Model Selection Demo")
    print("=" * 70)
    
    try:
        demo_model_selection_interface()
        show_model_comparison()
        show_cost_analysis()
        
        print("\n✨ Interactive model selection provides:")
        print("   • Clear model descriptions and recommendations")
        print("   • Real-time cost estimation")
        print("   • Agent-specific configuration")
        print("   • User-friendly selection interface")
        print("   • Comprehensive configuration summaries")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    print("\n🎯 Ready to try the real interactive evaluation!") 