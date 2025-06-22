#!/usr/bin/env python3
"""
Simple Test Suite for Core Enhanced Agent Functionality

Basic tests to verify the enhanced agents work correctly.
"""

import asyncio
import os
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

# Import agents
from agents.competitive_intelligence_agent import CompetitiveIntelligenceAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.contract_terms_agent import ContractTermsAgent
from agents.proposal_writer_agent import ProposalWriterAgent

# Import models
from agents.models import RFQRequirements, CustomerIntent, Quote

# Disable real model requests
models.ALLOW_MODEL_REQUESTS = False
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'


async def test_agents():
    """Test core agent functionality."""
    print("🧪 Testing Enhanced Multi-Agent RFQ System")
    print("=" * 60)
    
    # Create test data
    requirements = RFQRequirements(
        product_type="Enterprise Software",
        quantity=100,
        budget_range="$50,000 - $100,000",
        timeline="3 months"
    )
    
    intent = CustomerIntent(
        primary_intent="purchase_evaluation",
        sentiment="positive",
        urgency_level=4,
        price_sensitivity=3,
        readiness_to_buy=4
    )
    
    quote = Quote(
        quote_id="TEST-001",
        items=[{"product": "Enterprise Software", "quantity": 100, "price": 75000}],
        total_price=75000.0,
        delivery_terms="Standard delivery",
        validity_period="30 days"
    )
    
    try:
        # Test CompetitiveIntelligenceAgent
        print("✅ Testing CompetitiveIntelligenceAgent...")
        comp_agent = CompetitiveIntelligenceAgent()
        with comp_agent.agent.override(model=TestModel()):
            comp_result = await comp_agent.analyze_competitive_landscape(requirements, intent)
        print(f"   Market Position: {comp_result.market_position}")
        print(f"   Win Probability: {comp_result.win_probability}")
        
        # Test RiskAssessmentAgent
        print("✅ Testing RiskAssessmentAgent...")
        risk_agent = RiskAssessmentAgent()
        with risk_agent.agent.override(model=TestModel()):
            risk_result = await risk_agent.assess_risks(requirements, intent)
        print(f"   Risk Level: {risk_result.overall_risk_level}")
        print(f"   Risk Score: {risk_result.risk_score}")
        
        # Test ContractTermsAgent
        print("✅ Testing ContractTermsAgent...")
        contract_agent = ContractTermsAgent()
        with contract_agent.agent.override(model=TestModel()):
            contract_result = await contract_agent.develop_contract_terms(requirements, intent)
        print(f"   Payment Terms: {contract_result.payment_terms}")
        print(f"   Delivery Terms: {contract_result.delivery_terms}")
        
        # Test ProposalWriterAgent
        print("✅ Testing ProposalWriterAgent...")
        proposal_agent = ProposalWriterAgent()
        with proposal_agent.agent.override(model=TestModel()):
            proposal_result = await proposal_agent.generate_proposal(requirements, intent, quote)
        print(f"   Executive Summary: {proposal_result.executive_summary[:50]}...")
        print(f"   Technical Approach: {proposal_result.technical_approach[:50]}...")
        
        print("\n🎉 All tests passed! Enhanced multi-agent system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agents())
    exit(0 if success else 1)
