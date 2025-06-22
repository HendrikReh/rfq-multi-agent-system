#!/usr/bin/env python3
"""
Simplified Evaluation Test Suite for Enhanced Multi-Agent RFQ System

Tests basic functionality and structure validation without requiring real LLM calls.
This version is compatible with ALLOW_MODEL_REQUESTS = False.
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

# Import agents
from agents.competitive_intelligence_agent import CompetitiveIntelligenceAgent, CompetitiveAnalysis
from agents.risk_assessment_agent import RiskAssessmentAgent, RiskAssessment
from agents.contract_terms_agent import ContractTermsAgent, ContractTerms
from agents.proposal_writer_agent import ProposalWriterAgent, ProposalDocument
from agents.models import RFQRequirements, CustomerIntent

# Disable real model requests during testing
models.ALLOW_MODEL_REQUESTS = False
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

# Test scenarios
ENTERPRISE_SCENARIO = RFQRequirements(
    product_type="Enterprise CRM Software",
    quantity=500,
    budget_range="$200,000 - $500,000",
    delivery_date=datetime(2024, 6, 30),
    special_requirements=["SSO integration", "Custom reporting", "API access"]
)

GOVERNMENT_SCENARIO = RFQRequirements(
    product_type="Government Database System",
    quantity=1000,
    budget_range="$1,000,000+",
    delivery_date=datetime(2024, 12, 31),
    compliance_requirements=["FedRAMP", "SOC2", "FISMA"],
    special_requirements=["Security clearance required", "On-premise deployment"]
)

STARTUP_SCENARIO = RFQRequirements(
    product_type="Startup Analytics Platform",
    quantity=50,
    budget_range="$25,000 - $50,000",
    delivery_date=datetime(2024, 3, 31),
    special_requirements=["Cloud-native", "Scalable architecture"]
)

HIGH_URGENCY_INTENT = CustomerIntent(
    primary_intent="purchase_evaluation",
    sentiment="positive",
    urgency_level=5,
    price_sensitivity=2,
    readiness_to_buy=4
)

BUDGET_CONSCIOUS_INTENT = CustomerIntent(
    primary_intent="price_comparison",
    sentiment="neutral",
    urgency_level=3,
    price_sensitivity=5,
    readiness_to_buy=3
)


class TestCompetitiveIntelligenceEvaluation:
    """Simplified evaluation tests for CompetitiveIntelligenceAgent."""
    
    async def test_competitive_analysis_structure(self):
        """Test that competitive analysis produces valid structure."""
        agent = CompetitiveIntelligenceAgent()
        
        with agent.agent.override(model=TestModel()):
            result = await agent.analyze_competitive_landscape(ENTERPRISE_SCENARIO, HIGH_URGENCY_INTENT)
        
        # Verify structure
        assert isinstance(result, CompetitiveAnalysis)
        assert hasattr(result, 'market_position')
        assert hasattr(result, 'competitor_threats')
        assert hasattr(result, 'competitive_advantages')
        assert hasattr(result, 'win_probability')
        assert isinstance(result.win_probability, (int, float))
        assert 0.0 <= result.win_probability <= 1.0
        
        print(f"✅ CompetitiveIntelligenceAgent - Structure valid, Win Probability: {result.win_probability}")


class TestRiskAssessmentEvaluation:
    """Simplified evaluation tests for RiskAssessmentAgent."""
    
    async def test_risk_assessment_structure(self):
        """Test that risk assessment produces valid structure."""
        agent = RiskAssessmentAgent()
        
        with agent.agent.override(model=TestModel()):
            result = await agent.assess_risks(GOVERNMENT_SCENARIO, HIGH_URGENCY_INTENT)
        
        # Verify structure
        assert isinstance(result, RiskAssessment)
        assert hasattr(result, 'overall_risk_level')
        assert hasattr(result, 'financial_risks')
        assert hasattr(result, 'operational_risks')
        assert hasattr(result, 'risk_score')
        assert isinstance(result.risk_score, (int, float))
        assert 0.0 <= result.risk_score <= 10.0
        
        print(f"✅ RiskAssessmentAgent - Structure valid, Risk Score: {result.risk_score}/10")


class TestContractTermsEvaluation:
    """Simplified evaluation tests for ContractTermsAgent."""
    
    async def test_contract_terms_structure(self):
        """Test that contract terms produces valid structure."""
        agent = ContractTermsAgent()
        
        with agent.agent.override(model=TestModel()):
            result = await agent.develop_contract_terms(ENTERPRISE_SCENARIO, BUDGET_CONSCIOUS_INTENT)
        
        # Verify structure
        assert isinstance(result, ContractTerms)
        assert hasattr(result, 'payment_terms')
        assert hasattr(result, 'delivery_terms')
        assert hasattr(result, 'liability_limitations')
        assert hasattr(result, 'compliance_requirements')
        assert isinstance(result.liability_limitations, list)
        assert isinstance(result.compliance_requirements, list)
        
        print(f"✅ ContractTermsAgent - Structure valid, Payment Terms: {result.payment_terms}")


class TestProposalWriterEvaluation:
    """Simplified evaluation tests for ProposalWriterAgent."""
    
    async def test_proposal_structure(self):
        """Test that proposal generates valid structure."""
        agent = ProposalWriterAgent()
        
        # Mock quote for proposal generation
        from agents.models import Quote
        mock_quote = Quote(
            quote_id="TEST-001",
            items=[{"product": "Enterprise Software", "quantity": 100, "price": 50000}],
            total_price=50000.0,
            delivery_terms="Standard delivery",
            validity_period="30 days"
        )
        
        with agent.agent.override(model=TestModel()):
            result = await agent.generate_proposal(STARTUP_SCENARIO, HIGH_URGENCY_INTENT, mock_quote)
        
        # Verify structure
        assert isinstance(result, ProposalDocument)
        assert hasattr(result, 'executive_summary')
        assert hasattr(result, 'technical_approach')
        assert hasattr(result, 'pricing_section')
        assert hasattr(result, 'terms_and_conditions')
        assert hasattr(result, 'next_steps')
        assert isinstance(result.appendices, list)
        
        print(f"✅ ProposalWriterAgent - Structure valid, Executive Summary: {result.executive_summary[:50]}...")


class TestIntegratedEvaluation:
    """Test multiple agents working together."""
    
    async def test_multi_agent_workflow(self):
        """Test that multiple agents can work together in sequence."""
        
        # Initialize agents
        competitive_agent = CompetitiveIntelligenceAgent()
        risk_agent = RiskAssessmentAgent()
        contract_agent = ContractTermsAgent()
        
        scenario = ENTERPRISE_SCENARIO
        intent = HIGH_URGENCY_INTENT
        
        # Run agents in sequence with TestModel
        with competitive_agent.agent.override(model=TestModel()), \
             risk_agent.agent.override(model=TestModel()), \
             contract_agent.agent.override(model=TestModel()):
            
            competitive_result = await competitive_agent.analyze_competitive_landscape(scenario, intent)
            risk_result = await risk_agent.assess_risks(scenario, intent)
            contract_result = await contract_agent.develop_contract_terms(scenario, intent)
        
        # Verify all results are valid
        assert isinstance(competitive_result, CompetitiveAnalysis)
        assert isinstance(risk_result, RiskAssessment)
        assert isinstance(contract_result, ContractTerms)
        
        print(f"✅ Multi-Agent Workflow - All agents produced valid outputs")
        print(f"   Competitive Win Probability: {competitive_result.win_probability}")
        print(f"   Risk Score: {risk_result.risk_score}/10")
        print(f"   Contract Payment Terms: {contract_result.payment_terms}")


if __name__ == "__main__":
    """Run simplified evaluation tests."""
    print("🔍 SIMPLIFIED EVALUATION TESTS FOR ENHANCED MULTI-AGENT RFQ SYSTEM")
    print("=" * 80)
    print("Running structure validation and basic functionality tests...\n")
    
    # Set test environment
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    models.ALLOW_MODEL_REQUESTS = False
    
    async def run_evaluation_tests():
        """Run simplified evaluation test suite."""
        try:
            # Test competitive intelligence
            print("🎯 Testing CompetitiveIntelligenceAgent Structure...")
            comp_eval = TestCompetitiveIntelligenceEvaluation()
            await comp_eval.test_competitive_analysis_structure()
            
            # Test risk assessment
            print("\n🛡️ Testing RiskAssessmentAgent Structure...")
            risk_eval = TestRiskAssessmentEvaluation()
            await risk_eval.test_risk_assessment_structure()
            
            # Test contract terms
            print("\n📋 Testing ContractTermsAgent Structure...")
            contract_eval = TestContractTermsEvaluation()
            await contract_eval.test_contract_terms_structure()
            
            # Test proposal writer
            print("\n📄 Testing ProposalWriterAgent Structure...")
            proposal_eval = TestProposalWriterEvaluation()
            await proposal_eval.test_proposal_structure()
            
            # Test integrated workflow
            print("\n🔄 Testing Multi-Agent Workflow...")
            integrated_eval = TestIntegratedEvaluation()
            await integrated_eval.test_multi_agent_workflow()
            
            print("\n🎉 All simplified evaluation tests completed successfully!")
            print("\nNote: This simplified version tests structure and basic functionality.")
            print("For full LLM-based quality evaluation, run with ALLOW_MODEL_REQUESTS=True")
            print("and a valid OpenAI API key.")
            
        except Exception as e:
            print(f"❌ Evaluation test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run evaluation tests
    asyncio.run(run_evaluation_tests())
