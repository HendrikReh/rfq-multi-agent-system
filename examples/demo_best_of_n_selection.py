#!/usr/bin/env python3
"""
Best-of-N Selection Demo

Demonstrates the Best-of-N selection implementation following PydanticAI best practices.
Shows how to generate multiple candidate outputs and use LLM judge evaluation to
select the best one for RFQ processing scenarios.
"""

import asyncio
import os
import time
from typing import List

# PydanticAI imports
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

# Import our system components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rfq_system.agents.evaluation.best_of_n_selector import (
    BestOfNSelector,
    EvaluationCriteria,
    BestOfNDependencies,
    BestOfNAgent,
    BestOfNToolDeps
)
from rfq_system.core.interfaces.agent import BaseAgent, AgentContext, AgentStatus

# Import existing RFQ agents for demonstration
sys.path.append(str(Path(__file__).parent.parent))
from agents.proposal_writer_agent import ProposalWriterAgent
from agents.pricing_strategy_agent import PricingStrategyAgent
from agents.models import RFQRequirements, CustomerIntent, RFQDependencies

# Set up environment
models.ALLOW_MODEL_REQUESTS = False  # Use TestModel for demo
os.environ.setdefault('OPENAI_API_KEY', 'demo-key-for-testing')


class DemoRFQAgent(BaseAgent):
    """Demo RFQ agent that generates different quality proposals."""
    
    def __init__(self, agent_id: str = "demo_rfq_agent", quality_level: str = "mixed"):
        self.agent_id = agent_id
        self.quality_level = quality_level
        self.model = "demo-model"
        self.call_count = 0
        
        # Different quality responses based on configuration
        if quality_level == "high":
            self.responses = [
                "Comprehensive Enterprise CRM Solution: We propose a fully integrated CRM system with advanced analytics, custom workflows, mobile access, and 24/7 support. Our solution includes user training, data migration, and ongoing maintenance. Timeline: 4-6 months. Investment: $125,000 including implementation and first-year support.",
                "Premium CRM Platform: Our enterprise-grade solution offers advanced reporting, AI-powered insights, custom integrations, and scalable architecture. Includes comprehensive training program and dedicated support team. Timeline: 5 months. Investment: $135,000 with full implementation services.",
                "Professional CRM Implementation: Complete business solution with workflow automation, advanced security, mobile apps, and integration capabilities. Includes data migration, user training, and ongoing support. Timeline: 4 months. Investment: $115,000 total."
            ]
        elif quality_level == "medium":
            self.responses = [
                "CRM System Development: We can build a custom CRM system with user management, contact tracking, and basic reporting. Timeline: 3-4 months. Cost: $75,000.",
                "Business CRM Solution: Custom development including contact management, sales tracking, and reporting features. Timeline: 4 months. Cost: $80,000.",
                "CRM Application: We offer CRM development with standard features and basic customization. Timeline: 3 months. Cost: $70,000."
            ]
        else:  # mixed quality
            self.responses = [
                "Comprehensive Enterprise CRM Solution: We propose a fully integrated CRM system with advanced analytics, custom workflows, mobile access, and 24/7 support. Our solution includes user training, data migration, and ongoing maintenance. Timeline: 4-6 months. Investment: $125,000 including implementation and first-year support.",
                "Basic CRM: We can make a CRM system. Cost: $50k. Timeline: 2-3 months.",
                "CRM Development Services: Professional CRM solution with contact management, sales pipeline, reporting dashboard, and user access controls. Includes testing, deployment, and documentation. Timeline: 4 months. Investment: $85,000.",
                "Software development available for CRM needs.",
                "Premium CRM Platform: Our enterprise-grade solution offers advanced reporting, AI-powered insights, custom integrations, and scalable architecture. Includes comprehensive training program and dedicated support team. Timeline: 5 months. Investment: $135,000 with full implementation services."
            ]
    
    async def process(self, input_data, context):
        """Generate different quality responses."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate processing time
        return response
    
    async def health_check(self):
        return type('HealthStatus', (), {'status': AgentStatus.HEALTHY})()
    
    def get_capabilities(self):
        """Return agent capabilities."""
        return ["rfq_proposal_generation", "demo_responses"]
    
    async def initialize(self):
        """Initialize the agent."""
        pass
    
    async def shutdown(self):
        """Shutdown the agent."""
        pass


async def demo_basic_best_of_n():
    """Demonstrate basic Best-of-N selection with different quality outputs."""
    print("🎯 DEMO: Basic Best-of-N Selection")
    print("=" * 60)
    
    # Create selector and demo agent
    selector = BestOfNSelector(
        evaluation_model="openai:gpt-4o-mini",
        max_parallel_generations=5,
        enable_detailed_evaluation=True
    )
    
    demo_agent = DemoRFQAgent(quality_level="mixed")
    context = AgentContext(
        request_id="demo-basic",
        user_id="demo-user",
        session_id="demo-session"
    )
    
    # Define evaluation criteria for RFQ proposals
    rfq_criteria = EvaluationCriteria(
        accuracy_weight=0.2,      # Technical accuracy
        completeness_weight=0.4,  # Comprehensive coverage  
        relevance_weight=0.3,     # Business relevance
        clarity_weight=0.1        # Clear communication
    )
    
    prompt = """Generate a professional RFQ proposal for a CRM system with the following requirements:
- 100 user licenses
- Contact management and sales pipeline
- Reporting and analytics
- Mobile access
- Integration capabilities
- Budget range: $75,000 - $150,000
- Timeline: 4-6 months preferred"""
    
    print(f"📝 Prompt: {prompt[:100]}...")
    print(f"🔢 Generating {5} candidates with mixed quality levels...")
    
    # Override internal agents with TestModel for demo
    with selector._judge_agent.override(model=TestModel()):
        with selector._selection_agent.override(model=TestModel()):
            start_time = time.time()
            
            result = await selector.generate_best_of_n(
                target_agent=demo_agent,
                prompt=prompt,
                context=context,
                n=5,
                criteria=rfq_criteria
            )
            
            total_time = (time.time() - start_time) * 1000
    
    # Display results
    print(f"\n📊 RESULTS ({total_time:.0f}ms total)")
    print("-" * 60)
    print(f"✅ Generated {result.n_candidates} candidates")
    print(f"🎯 Selection confidence: {result.selection_confidence:.2f}")
    print(f"⚡ Generation time: {result.total_generation_time_ms:.0f}ms")
    print(f"🧠 Evaluation time: {result.total_evaluation_time_ms:.0f}ms")
    
    print(f"\n🏆 BEST CANDIDATE:")
    print(f"ID: {result.best_candidate.candidate_id}")
    print(f"Score: {result.best_evaluation.overall_score:.3f}")
    print(f"Output: {result.best_candidate.output[:200]}...")
    
    print(f"\n📈 ALL CANDIDATES:")
    for i, (candidate, evaluation) in enumerate(zip(result.all_candidates, result.all_evaluations)):
        print(f"{i+1}. {candidate.candidate_id} (Score: {evaluation.overall_score:.3f})")
        print(f"   {candidate.output[:100]}...")
        print()


async def demo_custom_criteria():
    """Demonstrate Best-of-N with custom evaluation criteria."""
    print("\n🎨 DEMO: Custom Evaluation Criteria")
    print("=" * 60)
    
    selector = BestOfNSelector(evaluation_model="openai:gpt-4o-mini")
    demo_agent = DemoRFQAgent(quality_level="mixed")
    context = AgentContext(
        request_id="demo-custom",
        user_id="demo-user", 
        session_id="demo-session"
    )
    
    # Custom criteria emphasizing different aspects
    scenarios = [
        {
            "name": "Cost-Focused Evaluation",
            "criteria": EvaluationCriteria(
                accuracy_weight=0.4,      # High accuracy for cost estimates
                completeness_weight=0.2,  # Less emphasis on completeness
                relevance_weight=0.3,     # Business relevance important
                clarity_weight=0.1        # Basic clarity needed
            )
        },
        {
            "name": "Quality-Focused Evaluation", 
            "criteria": EvaluationCriteria(
                accuracy_weight=0.2,      # Basic accuracy
                completeness_weight=0.5,  # High emphasis on completeness
                relevance_weight=0.2,     # Some business relevance
                clarity_weight=0.1        # Basic clarity
            )
        },
        {
            "name": "Communication-Focused Evaluation",
            "criteria": EvaluationCriteria(
                accuracy_weight=0.2,      # Basic accuracy
                completeness_weight=0.2,  # Some completeness
                relevance_weight=0.2,     # Some relevance
                clarity_weight=0.4        # High emphasis on clarity
            )
        }
    ]
    
    prompt = "Generate a proposal for a small business CRM system (25 users, $30k budget)."
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        criteria = scenario['criteria']
        print(f"   Weights: Accuracy={criteria.accuracy_weight}, Completeness={criteria.completeness_weight}, "
              f"Relevance={criteria.relevance_weight}, Clarity={criteria.clarity_weight}")
        
        with selector._judge_agent.override(model=TestModel()):
            with selector._selection_agent.override(model=TestModel()):
                result = await selector.generate_best_of_n(
                    target_agent=demo_agent,
                    prompt=prompt,
                    context=context,
                    n=3,
                    criteria=criteria
                )
        
        print(f"   🏆 Best: {result.best_candidate.candidate_id} (Score: {result.best_evaluation.overall_score:.3f})")
        print(f"   🎯 Confidence: {result.selection_confidence:.3f}")


async def demo_agent_delegation():
    """Demonstrate Best-of-N using agent delegation pattern."""
    print("\n🤝 DEMO: Agent Delegation Pattern")
    print("=" * 60)
    
    # Create Best-of-N agent with tool delegation
    selector = BestOfNSelector(evaluation_model="openai:gpt-4o-mini")
    target_agent = DemoRFQAgent(quality_level="high")
    context = AgentContext(
        request_id="demo-delegation",
        user_id="demo-user",
        session_id="demo-session"
    )
    
    # Set up agent delegation dependencies
    deps = BestOfNToolDeps(
        selector=selector,
        target_agent=target_agent,
        context=context
    )
    
    best_of_n_agent = BestOfNAgent(model="openai:gpt-4o-mini")
    
    print("🤖 Using BestOfNAgent with tool delegation...")
    print("📝 Prompt: Generate the best possible enterprise CRM proposal")
    
    # Override all agents with TestModel
    with selector._judge_agent.override(model=TestModel()):
        with selector._selection_agent.override(model=TestModel()):
            with best_of_n_agent.override(model=TestModel()):
                result = await best_of_n_agent.run(
                    "Generate the best possible enterprise CRM proposal for 200 users with advanced analytics",
                    deps
                )
    
    print(f"✅ Agent delegation completed")
    print(f"📄 Result type: {type(result.data).__name__}")
    print(f"📊 Result: {str(result.data)[:200]}...")


async def demo_performance_comparison():
    """Compare performance of Best-of-N vs single generation."""
    print("\n⚡ DEMO: Performance Comparison")
    print("=" * 60)
    
    selector = BestOfNSelector(evaluation_model="openai:gpt-4o-mini")
    demo_agent = DemoRFQAgent(quality_level="mixed")
    context = AgentContext(
        request_id="demo-perf",
        user_id="demo-user",
        session_id="demo-session"
    )
    
    prompt = "Generate a CRM proposal for 50 users, $40k budget, 3-month timeline."
    
    # Test single generation
    print("🔄 Testing single generation...")
    start_time = time.time()
    single_result = await demo_agent.process(prompt, context)
    single_time = (time.time() - start_time) * 1000
    
    # Test Best-of-N generation
    print("🔄 Testing Best-of-N generation (n=3)...")
    with selector._judge_agent.override(model=TestModel()):
        with selector._selection_agent.override(model=TestModel()):
            start_time = time.time()
            best_of_n_result = await selector.generate_best_of_n(
                target_agent=demo_agent,
                prompt=prompt,
                context=context,
                n=3
            )
            best_of_n_time = (time.time() - start_time) * 1000
    
    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Single Generation:")
    print(f"  ⏱️  Time: {single_time:.0f}ms")
    print(f"  📄 Output: {single_result[:100]}...")
    
    print(f"\nBest-of-N Generation:")
    print(f"  ⏱️  Time: {best_of_n_time:.0f}ms ({best_of_n_time/single_time:.1f}x slower)")
    print(f"  🎯 Confidence: {best_of_n_result.selection_confidence:.3f}")
    print(f"  🏆 Best Score: {best_of_n_result.best_evaluation.overall_score:.3f}")
    print(f"  📄 Best Output: {best_of_n_result.best_candidate.output[:100]}...")
    
    print(f"\n💡 Trade-off: {best_of_n_time/single_time:.1f}x time for {best_of_n_result.selection_confidence:.1%} confidence")


async def main():
    """Run all Best-of-N demos."""
    print("🚀 BEST-OF-N SELECTION DEMONSTRATION")
    print("=" * 80)
    print("Showcasing PydanticAI agent delegation and LLM judge evaluation")
    print("=" * 80)
    
    try:
        # Run all demos
        await demo_basic_best_of_n()
        await demo_custom_criteria()
        await demo_agent_delegation()
        await demo_performance_comparison()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ Multiple candidate generation with parallel execution")
        print("  ✅ LLM judge evaluation with custom criteria")
        print("  ✅ Agent delegation patterns following PydanticAI best practices")
        print("  ✅ Confidence scoring and selection logic")
        print("  ✅ Performance monitoring and timeout handling")
        print("  ✅ Comprehensive error handling and graceful degradation")
        
        print("\n📚 Best Practices Followed:")
        print("  🔧 Agent delegation via tools (Level 2 multi-agent complexity)")
        print("  🧪 TestModel for fast, deterministic testing")
        print("  ⚡ Asyncio for parallel execution and proper error handling")
        print("  📊 LLM judge evaluation for objective quality assessment")
        print("  🎯 Configurable evaluation criteria for different use cases")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """Run the Best-of-N selection demo."""
    asyncio.run(main()) 