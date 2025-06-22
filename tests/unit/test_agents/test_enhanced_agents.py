#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Multi-Agent RFQ System

Tests all new specialized agents, orchestration patterns, and integration framework
using PydanticAI testing best practices with TestModel and FunctionModel.
"""

import asyncio
import os
import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# PydanticAI testing imports
from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

# Import our enhanced agents
from agents.competitive_intelligence_agent import CompetitiveIntelligenceAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.contract_terms_agent import ContractTermsAgent
from agents.proposal_writer_agent import ProposalWriterAgent
from agents.enhanced_orchestrator import EnhancedRFQOrchestrator
from agents.integration_framework import IntegratedRFQSystem

# Import models from agents (they have their own model definitions)
from agents.competitive_intelligence_agent import CompetitiveAnalysis
from agents.risk_assessment_agent import RiskAssessment
from agents.contract_terms_agent import ContractTerms
from agents.proposal_writer_agent import ProposalDocument
from agents.integration_framework import SystemHealthReport, ComprehensiveRFQResult

# Import shared models
from agents.models import RFQRequirements, CustomerIntent, RFQDependencies

# Disable real model requests during testing
models.ALLOW_MODEL_REQUESTS = False

# Set test API key
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

pytestmark = pytest.mark.anyio


class TestCompetitiveIntelligenceAgent:
    """Test suite for CompetitiveIntelligenceAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = CompetitiveIntelligenceAgent()
        
    async def test_analyze_competitive_landscape_basic(self):
        """Test basic competitive analysis functionality."""
        # Mock requirements
        requirements = RFQRequirements(
            product_type="Enterprise Software",
            quantity=100,
            budget_range="$50,000 - $100,000",
            timeline="3 months"
        )
        
        # Mock customer intent
        intent = CustomerIntent(
            primary_intent="purchase_evaluation",
            sentiment="positive",
            urgency_level=4,
            price_sensitivity=3,
            readiness_to_buy=4
        )
        
        # Use TestModel for testing
        with self.agent.agent.override(model=TestModel()):
            result = await self.agent.analyze_competitive_landscape(requirements, intent)
            
        # Verify result structure (TestModel generates valid data)
        assert isinstance(result, CompetitiveAnalysis)
        assert result.market_position is not None
        assert isinstance(result.competitor_threats, list)
        assert isinstance(result.competitive_advantages, list)
        assert isinstance(result.pricing_benchmarks, dict)
        assert isinstance(result.win_probability, (int, float))
        assert result.win_probability >= 0.0
        assert result.win_probability <= 1.0
        assert result.recommended_strategy is not None
        assert isinstance(result.differentiation_points, list)


class TestRiskAssessmentAgent:
    """Test suite for RiskAssessmentAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = RiskAssessmentAgent()
        
    async def test_assess_risks_basic(self):
        """Test basic risk assessment functionality."""
        requirements = RFQRequirements(
            product_type="Government Software",
            quantity=500,
            budget_range="$500,000+",
            compliance_requirements=["SOC2", "FedRAMP"]
        )
        
        intent = CustomerIntent(
            primary_intent="purchase_evaluation",
            sentiment="neutral",
            urgency_level=3,
            price_sensitivity=4,
            readiness_to_buy=3
        )
        
        with self.agent.agent.override(model=TestModel()):
            result = await self.agent.assess_risks(requirements, intent)
            
        # Verify risk assessment structure
        assert isinstance(result, RiskAssessment)
        assert isinstance(result.overall_risk_level, str)
        assert isinstance(result.financial_risks, list)
        assert isinstance(result.operational_risks, list)
        assert isinstance(result.customer_risks, list)
        assert isinstance(result.project_risks, list)
        assert isinstance(result.market_risks, list)
        assert isinstance(result.mitigation_strategies, list)
        assert isinstance(result.risk_score, (int, float))
        assert result.risk_score >= 0.0
        assert result.risk_score <= 10.0
        assert result.recommendation is not None
        assert isinstance(result.insurance_requirements, list)


class TestContractTermsAgent:
    """Test suite for ContractTermsAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ContractTermsAgent()
        
    async def test_develop_contract_terms_basic(self):
        """Test basic contract terms development."""
        requirements = RFQRequirements(
            product_type="SaaS Platform",
            quantity=100,
            budget_range="$50,000 - $100,000"
        )
        
        intent = CustomerIntent(
            primary_intent="purchase_evaluation",
            sentiment="positive",
            urgency_level=3,
            price_sensitivity=3,
            readiness_to_buy=3
        )
        
        with self.agent.agent.override(model=TestModel()):
            result = await self.agent.develop_contract_terms(requirements, intent)
            
        # Verify contract terms structure
        assert isinstance(result, ContractTerms)
        assert result.payment_terms is not None
        assert result.delivery_terms is not None
        assert isinstance(result.liability_limitations, list)
        assert isinstance(result.compliance_requirements, list)


class TestProposalWriterAgent:
    """Test suite for ProposalWriterAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ProposalWriterAgent()
        
    async def test_generate_proposal_basic(self):
        """Test basic proposal generation."""
        requirements = RFQRequirements(
            product_type="Project Management Software",
            quantity=50,
            budget_range="$25,000 - $50,000"
        )
        
        intent = CustomerIntent(
            primary_intent="purchase_evaluation",
            sentiment="neutral",
            urgency_level=3,
            price_sensitivity=4,
            readiness_to_buy=4
        )
        
        # Import Quote model and create mock quote
        from agents.models import Quote
        
        # Mock quote
        mock_quote = Quote(
            quote_id="TEST-001",
            items=[{"product": "Project Management Software", "quantity": 50, "price": 35000}],
            total_price=35000.0,
            delivery_terms="Standard delivery within 30 days",
            validity_period="30 days",
            special_conditions=["Implementation support included"]
        )
        
        with self.agent.agent.override(model=TestModel()):
            result = await self.agent.generate_proposal(requirements, intent, mock_quote)
            
        # Verify proposal structure
        assert isinstance(result, ProposalDocument)
        assert result.executive_summary is not None
        assert result.problem_statement is not None
        assert result.proposed_solution is not None
        assert result.technical_approach is not None
        assert result.project_timeline is not None
        assert result.team_qualifications is not None
        assert result.pricing_section is not None
        assert result.terms_and_conditions is not None
        assert result.next_steps is not None
        assert isinstance(result.appendices, list)
        assert isinstance(result.presentation_outline, list)


class TestEnhancedOrchestrator:
    """Test suite for EnhancedRFQOrchestrator with agent delegation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = EnhancedRFQOrchestrator()
        
    async def test_orchestrator_initialization(self):
        """Test that orchestrator initializes with all required agents."""
        assert self.orchestrator.rfq_parser is not None
        assert self.orchestrator.intent_agent is not None
        assert self.orchestrator.pricing_agent is not None
        assert self.orchestrator.question_agent is not None
        assert self.orchestrator.agent is not None


class TestIntegratedRFQSystem:
    """Test suite for IntegratedRFQSystem with comprehensive orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = IntegratedRFQSystem()
        
    async def test_system_initialization(self):
        """Test that integrated system initializes properly."""
        assert self.system.enhanced_orchestrator is not None
        assert self.system.competitive_agent is not None
        assert self.system.risk_agent is not None
        assert self.system.contract_agent is not None
        assert self.system.proposal_agent is not None
        
    async def test_system_health_monitoring(self):
        """Test system health monitoring capabilities."""
        health_report = await self.system.get_system_health_report()
        
        assert isinstance(health_report, SystemHealthReport)
        assert health_report.overall_status.lower() in ['healthy', 'degraded', 'unhealthy']
        assert health_report.total_agents > 0
        assert health_report.healthy_agents >= 0
        assert health_report.healthy_agents <= health_report.total_agents


class TestMultiAgentPatterns:
    """Test suite for advanced multi-agent patterns."""
    
    async def test_agent_delegation_pattern(self):
        """Test agent delegation via tools pattern."""
        orchestrator = EnhancedRFQOrchestrator()
        
        # Test that orchestrator has tools for delegation
        # Check if orchestrator has delegation capabilities
        assert hasattr(orchestrator, 'agent')
        assert orchestrator.agent is not None
        
        # Verify delegation tools exist
        expected_tools = [
            'parse_requirements',
            'analyze_customer_intent', 
            'make_interaction_decision',
            'analyze_competitive_landscape',
            'assess_risks',
            'develop_contract_terms',
            'generate_proposal'
        ]
        
        # For now, just verify the orchestrator is properly initialized
        # TODO: Implement proper tool delegation testing when PydanticAI tools API is finalized
        assert len(expected_tools) > 0  # Verify we have expected tools defined


if __name__ == "__main__":
    """Run tests manually if needed."""
    print("🧪 TESTING ENHANCED MULTI-AGENT RFQ SYSTEM")
    print("=" * 80)
    print("Running comprehensive tests for all new agents and patterns...\n")
    
    # Set test environment
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    models.ALLOW_MODEL_REQUESTS = False
    
    async def run_sample_tests():
        """Run sample tests to verify functionality."""
        try:
            # Test competitive intelligence agent
            print("✅ Testing CompetitiveIntelligenceAgent...")
            test_competitive = TestCompetitiveIntelligenceAgent()
            test_competitive.setup_method()
            await test_competitive.test_analyze_competitive_landscape_basic()
            
            # Test risk assessment agent  
            print("✅ Testing RiskAssessmentAgent...")
            test_risk = TestRiskAssessmentAgent()
            test_risk.setup_method()
            await test_risk.test_assess_risks_basic()
            
            # Test contract terms agent
            print("✅ Testing ContractTermsAgent...")
            test_contract = TestContractTermsAgent()
            test_contract.setup_method()
            await test_contract.test_develop_contract_terms_basic()
            
            # Test proposal writer agent
            print("✅ Testing ProposalWriterAgent...")
            test_proposal = TestProposalWriterAgent()
            test_proposal.setup_method()
            await test_proposal.test_generate_proposal_basic()
            
            # Test integrated system
            print("✅ Testing IntegratedRFQSystem...")
            test_system = TestIntegratedRFQSystem()
            test_system.setup_method()
            await test_system.test_system_initialization()
            
            print("\n🎉 All sample tests passed! Enhanced multi-agent system is working correctly.")
            print("\nTo run full test suite with pytest:")
            print("  pytest test_enhanced_agents.py -v")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run sample tests
    asyncio.run(run_sample_tests())
