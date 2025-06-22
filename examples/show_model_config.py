#!/usr/bin/env python3
"""
Model Configuration Display Utility

This script shows the current model configuration for all RFQ agents
and demonstrates how to customize models using environment variables.
"""

import os
from agents.utils import print_model_configuration, get_all_agent_models


def main():
    """Display current model configuration and usage examples."""
    print("🚀 RFQ Multi-Agent System - Model Configuration")
    print("=" * 60)
    print()
    
    # Show current configuration
    print_model_configuration()
    
    print("\n" + "=" * 60)
    print("📋 Model Optimization Guide:")
    print()
    
    print("💰 Cost Optimization:")
    print("   • Use gpt-4o-mini for simple tasks (state tracking, evaluation)")
    print("   • Use gpt-4o for complex reasoning (intent analysis, pricing)")
    print("   • Use gpt-4o for creative tasks (question generation, customer simulation)")
    print()
    
    print("⚡ Performance Optimization:")
    print("   • gpt-4o-mini: Faster, cheaper, good for structured tasks")
    print("   • gpt-4o: Slower, more expensive, better reasoning and creativity")
    print("   • gpt-4: Previous generation, balanced cost/performance")
    print()
    
    print("🎯 Recommended Model Assignments:")
    print("   • RFQ Parser: gpt-4o (needs precise requirement extraction)")
    print("   • Customer Intent: gpt-4o (complex sentiment and intent analysis)")
    print("   • Pricing Strategy: gpt-4o (sophisticated pricing calculations)")
    print("   • Question Generation: gpt-4o (creative, contextual questions)")
    print("   • Customer Response: gpt-4o (realistic customer simulation)")
    print("   • Conversation State: gpt-4o-mini (simple state tracking)")
    print("   • Evaluation: gpt-4o-mini (straightforward performance metrics)")
    print()
    
    print("🔧 Customization Examples:")
    print("   # Use cheaper model for pricing (cost optimization)")
    print("   export RFQ_PRICING_STRATEGY_MODEL='openai:gpt-4o-mini'")
    print()
    print("   # Use latest model for customer simulation (quality optimization)")
    print("   export RFQ_CUSTOMER_RESPONSE_MODEL='openai:gpt-4o'")
    print()
    print("   # Use specific model for question generation")
    print("   export RFQ_QUESTION_GENERATION_MODEL='openai:gpt-4'")
    print()
    
    print("💡 Pro Tips:")
    print("   • Test different models with your specific use cases")
    print("   • Monitor costs and performance for optimal configuration")
    print("   • Use gpt-4o for production, gpt-4o-mini for development")
    print("   • Consider using different models for different customer tiers")


if __name__ == "__main__":
    main() 