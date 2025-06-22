#!/usr/bin/env python3
"""
Real LLM Evaluation Demo

This script demonstrates how to run real LLM evaluation tests for the Best-of-N selector.
It shows the complete evaluation process using actual API calls and generates comprehensive
JSON reports similar to the existing scenario reports in ./reports.

WARNING: This requires a real OpenAI API key and will incur costs.

NOTE: PydanticEvals v0.3.2 has a duration reporting bug (always shows 1.0s).
Duration display is disabled; accurate timing is provided in the detailed analysis.

Features:
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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pydantic_ai import models


async def main():
    """Run the real LLM evaluation demo."""
    
    print("🚀 Real LLM Evaluation Demo for Best-of-N Selector")
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
    
    # Ask for confirmation
    response = input("\nContinue with real LLM evaluation? (y/N): ")
    if response.lower() != 'y':
        print("❌ Evaluation cancelled.")
        return
    
    print("\n🔄 Starting real LLM evaluation...")
    
    # Import and run the evaluation
    try:
        # Import the real LLM test module
        sys.path.append(str(Path(__file__).parent.parent / "tests" / "evaluation"))
        from test_best_of_n_real_llm import run_real_llm_evaluation
        
        # Enable real model requests
        models.ALLOW_MODEL_REQUESTS = True
        
        # Run the evaluation
        report_filepath = await run_real_llm_evaluation()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📋 Comprehensive report generated:")
        print(f"   📁 {report_filepath}")
        print(f"\n💡 You can now:")
        print(f"   • View the JSON report: cat {report_filepath}")
        print(f"   • Compare with existing reports in ./reports")
        print(f"   • Analyze performance metrics and evaluation scores")
        print(f"   • Use the report data for further analysis")
        
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