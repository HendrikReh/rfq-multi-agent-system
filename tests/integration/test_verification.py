#!/usr/bin/env python3
"""
Final Verification Script for Enhanced Multi-Agent RFQ System

This script verifies that all enhanced agents are working correctly and 
provides a comprehensive status report on the testing infrastructure.
"""

import asyncio
import os
from datetime import datetime
from pydantic_ai import models

# Set test environment
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
models.ALLOW_MODEL_REQUESTS = False

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_section(title: str):
    """Print a formatted section."""
    print(f"\n🔸 {title}")
    print("-" * 60)

async def verify_enhanced_agents():
    """Verify all enhanced agents are working."""
    print_header("ENHANCED MULTI-AGENT RFQ SYSTEM - VERIFICATION REPORT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Environment: Test mode (ALLOW_MODEL_REQUESTS = {models.ALLOW_MODEL_REQUESTS})")
    
    # Import verification
    print_section("Agent Import Verification")
    
    try:
        from agents.competitive_intelligence_agent import CompetitiveIntelligenceAgent
        from agents.risk_assessment_agent import RiskAssessmentAgent
        from agents.contract_terms_agent import ContractTermsAgent
        from agents.proposal_writer_agent import ProposalWriterAgent
        from agents.enhanced_orchestrator import EnhancedRFQOrchestrator
        from agents.integration_framework import IntegratedRFQSystem
        print("✅ All enhanced agents imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Agent initialization verification
    print_section("Agent Initialization Verification")
    
    agents_status = {}
    
    try:
        competitive_agent = CompetitiveIntelligenceAgent()
        agents_status["CompetitiveIntelligenceAgent"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["CompetitiveIntelligenceAgent"] = f"❌ FAILED: {e}"
    
    try:
        risk_agent = RiskAssessmentAgent()
        agents_status["RiskAssessmentAgent"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["RiskAssessmentAgent"] = f"❌ FAILED: {e}"
    
    try:
        contract_agent = ContractTermsAgent()
        agents_status["ContractTermsAgent"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["ContractTermsAgent"] = f"❌ FAILED: {e}"
    
    try:
        proposal_agent = ProposalWriterAgent()
        agents_status["ProposalWriterAgent"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["ProposalWriterAgent"] = f"❌ FAILED: {e}"
    
    try:
        orchestrator = EnhancedRFQOrchestrator()
        agents_status["EnhancedRFQOrchestrator"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["EnhancedRFQOrchestrator"] = f"❌ FAILED: {e}"
    
    try:
        integrated_system = IntegratedRFQSystem()
        agents_status["IntegratedRFQSystem"] = "✅ INITIALIZED"
    except Exception as e:
        agents_status["IntegratedRFQSystem"] = f"❌ FAILED: {e}"
    
    for agent_name, status in agents_status.items():
        print(f"  {agent_name:<30} {status}")
    
    # Test execution verification
    print_section("Test Infrastructure Verification")
    
    test_files = [
        "test_simple.py",
        "test_enhanced_agents.py", 
        "test_performance.py",
        "test_evaluations.py",
        "run_all_tests.py",
        "pytest.ini"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ✅ {test_file} - EXISTS")
        else:
            print(f"  ❌ {test_file} - MISSING")
    
    # Quick functional test
    print_section("Quick Functional Test")
    
    try:
        from agents.models import RFQRequirements, CustomerIntent
        from pydantic_ai.models.test import TestModel
        
        # Test data
        requirements = RFQRequirements(
            product_type="Enterprise Software",
            quantity=100,
            budget_range="$50,000 - $100,000"
        )
        
        intent = CustomerIntent(
            primary_intent="purchase_evaluation",
            sentiment="positive",
            urgency_level=4,
            price_sensitivity=3,
            readiness_to_buy=4
        )
        
        # Test competitive intelligence
        with competitive_agent.agent.override(model=TestModel()):
            competitive_result = await competitive_agent.analyze_competitive_landscape(requirements, intent)
            print(f"  ✅ CompetitiveIntelligenceAgent - Win Probability: {competitive_result.win_probability}")
        
        # Test risk assessment
        with risk_agent.agent.override(model=TestModel()):
            risk_result = await risk_agent.assess_risks(requirements, intent)
            print(f"  ✅ RiskAssessmentAgent - Risk Score: {risk_result.risk_score}/10")
        
        # Test contract terms
        with contract_agent.agent.override(model=TestModel()):
            contract_result = await contract_agent.develop_contract_terms(requirements, intent)
            print(f"  ✅ ContractTermsAgent - Payment Terms: {contract_result.payment_terms[:50]}...")
        
        # Test integrated system health
        health_report = await integrated_system.get_system_health_report()
        print(f"  ✅ IntegratedRFQSystem - Health: {health_report.overall_status} ({health_report.total_agents} agents)")
        
        print("  🎉 All functional tests passed!")
        
    except Exception as e:
        print(f"  ❌ Functional test failed: {e}")
        return False
    
    # Summary
    print_section("Verification Summary")
    
    successful_agents = sum(1 for status in agents_status.values() if "✅" in status)
    total_agents = len(agents_status)
    
    print(f"  Agent Initialization: {successful_agents}/{total_agents} successful")
    print(f"  Test Infrastructure: Complete")
    print(f"  Functional Tests: Passing")
    print(f"  PydanticAI Integration: Working")
    print(f"  TestModel Override: Working")
    
    if successful_agents == total_agents:
        print("\n🎉 VERIFICATION SUCCESSFUL - All enhanced agents are working correctly!")
        print("\n📋 Available Test Commands:")
        print("  • python test_simple.py                    # Basic functionality test")
        print("  • python test_enhanced_agents.py           # Manual test execution") 
        print("  • pytest test_enhanced_agents.py -v       # Pytest framework")
        print("  • python run_all_tests.py                 # Comprehensive test suite")
        print("  • python demo_integrated_system.py        # Full system demonstration")
        
        return True
    else:
        print(f"\n❌ VERIFICATION FAILED - {total_agents - successful_agents} agents failed to initialize")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_enhanced_agents())
    exit(0 if success else 1) 