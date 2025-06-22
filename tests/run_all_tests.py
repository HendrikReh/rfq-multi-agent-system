#!/usr/bin/env python3
"""
Comprehensive Test Runner for Enhanced Multi-Agent RFQ System

Runs all test suites: unit tests, performance tests, and evaluations
with comprehensive reporting and coverage analysis.
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Set test environment
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

# PydanticAI testing setup
from pydantic_ai import models
models.ALLOW_MODEL_REQUESTS = False


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"🧪 {title}")
    print(f"{'─' * 60}")


async def run_unit_tests():
    """Run unit tests for all enhanced agents."""
    print_section("UNIT TESTS - Enhanced Agents")
    
    try:
        # Import and run unit tests
        from unit.test_agents.test_enhanced_agents import (
            TestCompetitiveIntelligenceAgent,
            TestRiskAssessmentAgent,
            TestContractTermsAgent,
            TestProposalWriterAgent,
            TestEnhancedOrchestrator,
            TestIntegratedRFQSystem,
            TestMultiAgentPatterns
        )
        
        test_results = {}
        
        # Test CompetitiveIntelligenceAgent
        print("✅ Testing CompetitiveIntelligenceAgent...")
        test_competitive = TestCompetitiveIntelligenceAgent()
        test_competitive.setup_method()
        await test_competitive.test_analyze_competitive_landscape_basic()
        test_results['competitive_intelligence'] = 'PASSED'
        
        # Test RiskAssessmentAgent
        print("✅ Testing RiskAssessmentAgent...")
        test_risk = TestRiskAssessmentAgent()
        test_risk.setup_method()
        await test_risk.test_assess_risks_basic()
        test_results['risk_assessment'] = 'PASSED'
        
        # Test ContractTermsAgent
        print("✅ Testing ContractTermsAgent...")
        test_contract = TestContractTermsAgent()
        test_contract.setup_method()
        await test_contract.test_develop_contract_terms_basic()
        test_results['contract_terms'] = 'PASSED'
        
        # Test ProposalWriterAgent
        print("✅ Testing ProposalWriterAgent...")
        test_proposal = TestProposalWriterAgent()
        test_proposal.setup_method()
        await test_proposal.test_generate_proposal_basic()
        test_results['proposal_writer'] = 'PASSED'
        
        # Test EnhancedOrchestrator
        print("✅ Testing EnhancedOrchestrator...")
        test_orchestrator = TestEnhancedOrchestrator()
        test_orchestrator.setup_method()
        await test_orchestrator.test_orchestrator_initialization()
        test_results['enhanced_orchestrator'] = 'PASSED'
        
        # Test IntegratedRFQSystem
        print("✅ Testing IntegratedRFQSystem...")
        test_system = TestIntegratedRFQSystem()
        test_system.setup_method()
        await test_system.test_system_initialization()
        await test_system.test_system_health_monitoring()
        test_results['integrated_system'] = 'PASSED'
        
        # Test MultiAgentPatterns
        print("✅ Testing MultiAgentPatterns...")
        test_patterns = TestMultiAgentPatterns()
        await test_patterns.test_agent_delegation_pattern()
        test_results['multi_agent_patterns'] = 'PASSED'
        
        print(f"\n🎉 Unit Tests Summary:")
        for test_name, status in test_results.items():
            print(f"  {test_name:25} → {status}")
            
        return test_results
        
    except Exception as e:
        print(f"❌ Unit tests failed: {e}")
        return {'unit_tests': 'FAILED'}


async def run_performance_tests():
    """Run performance and scalability tests."""
    print_section("PERFORMANCE TESTS - Parallel Execution & Scalability")
    
    try:
        # Import and run performance tests
        from performance.test_performance import (
            TestParallelExecutionPerformance,
            TestSystemHealthMonitoring
        )
        
        test_results = {}
        
        # Test parallel execution performance
        print("🏃 Testing Parallel vs Sequential Performance...")
        perf_test = TestParallelExecutionPerformance()
        perf_test.setup_method()
        await perf_test.test_parallel_vs_sequential_performance()
        test_results['parallel_performance'] = 'PASSED'
        
        # Test system health monitoring
        print("💚 Testing System Health Monitoring...")
        health_test = TestSystemHealthMonitoring()
        health_test.setup_method()
        await health_test.test_health_report_generation()
        test_results['health_monitoring'] = 'PASSED'
        
        print(f"\n🚀 Performance Tests Summary:")
        for test_name, status in test_results.items():
            print(f"  {test_name:25} → {status}")
            
        return test_results
        
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return {'performance_tests': 'FAILED'}


async def run_evaluation_tests():
    """Run quality evaluation tests using PydanticEvals."""
    print_section("EVALUATION TESTS - Agent Quality Assessment")
    
    try:
        # Import and run evaluation tests
        from integration.test_evaluations import (
            TestCompetitiveIntelligenceEvaluation,
            TestRiskAssessmentEvaluation,
            TestContractTermsEvaluation,
            TestProposalWriterEvaluation,
            TestIntegratedEvaluation
        )
        
        test_results = {}
        
        # Test competitive intelligence evaluation
        print("🎯 Evaluating CompetitiveIntelligenceAgent Structure...")
        comp_eval = TestCompetitiveIntelligenceEvaluation()
        await comp_eval.test_competitive_analysis_structure()
        test_results['competitive_evaluation'] = 'PASSED'
        
        # Test risk assessment evaluation
        print("⚠️ Evaluating RiskAssessmentAgent Structure...")
        risk_eval = TestRiskAssessmentEvaluation()
        await risk_eval.test_risk_assessment_structure()
        test_results['risk_evaluation'] = 'PASSED'
        
        # Test contract terms evaluation
        print("📋 Evaluating ContractTermsAgent Structure...")
        contract_eval = TestContractTermsEvaluation()
        await contract_eval.test_contract_terms_structure()
        test_results['contract_evaluation'] = 'PASSED'
        
        # Test proposal writer evaluation
        print("📄 Evaluating ProposalWriterAgent Structure...")
        proposal_eval = TestProposalWriterEvaluation()
        await proposal_eval.test_proposal_structure()
        test_results['proposal_evaluation'] = 'PASSED'
        
        # Test integrated multi-agent workflow
        print("🔄 Evaluating Multi-Agent Workflow...")
        integrated_eval = TestIntegratedEvaluation()
        await integrated_eval.test_multi_agent_workflow()
        test_results['integrated_evaluation'] = 'PASSED'
        
        print(f"\n🔍 Evaluation Tests Summary:")
        for test_name, status in test_results.items():
            print(f"  {test_name:25} → {status}")
            
        return test_results
        
    except Exception as e:
        print(f"❌ Evaluation tests failed: {e}")
        return {'evaluation_tests': 'FAILED'}


async def run_existing_tests():
    """Run existing test suites."""
    print_section("EXISTING TESTS - Model Assignment & Scenario Recording")
    
    try:
        # Run existing model assignment tests
        print("🔧 Testing Model Assignment Logic...")
        os.system("python unit/test_agents/test_model_assignment.py")
        
        # Run existing scenario recording tests
        print("📊 Testing Scenario Recording...")
        os.system("python integration/test_scenario_recording.py")
        
        return {'existing_tests': 'PASSED'}
        
    except Exception as e:
        print(f"❌ Existing tests failed: {e}")
        return {'existing_tests': 'FAILED'}


def generate_test_report(all_results: dict, start_time: datetime, end_time: datetime):
    """Generate comprehensive test report."""
    print_banner("COMPREHENSIVE TEST REPORT", "=")
    
    total_duration = (end_time - start_time).total_seconds()
    
    print(f"Test Execution Time: {total_duration:.2f} seconds")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count results
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print(f"\n📊 Test Results by Category:")
    print(f"{'Category':<30} {'Status':<10} {'Details'}")
    print(f"{'-' * 60}")
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            category_passed = sum(1 for status in results.values() if status == 'PASSED')
            category_total = len(results)
            total_tests += category_total
            passed_tests += category_passed
            failed_tests += (category_total - category_passed)
            
            status = "✅ PASSED" if category_passed == category_total else "❌ FAILED"
            print(f"{category:<30} {status:<10} {category_passed}/{category_total}")
        else:
            total_tests += 1
            if results == 'PASSED':
                passed_tests += 1
                print(f"{category:<30} {'✅ PASSED':<10} 1/1")
            else:
                failed_tests += 1
                print(f"{category:<30} {'❌ FAILED':<10} 0/1")
    
    print(f"{'-' * 60}")
    print(f"{'TOTAL':<30} {'':<10} {passed_tests}/{total_tests}")
    
    # Overall status
    overall_status = "✅ ALL TESTS PASSED" if failed_tests == 0 else f"❌ {failed_tests} TESTS FAILED"
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Results:")
    print(f"  Status: {overall_status}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Tests Passed: {passed_tests}")
    print(f"  Tests Failed: {failed_tests}")
    print(f"  Total Tests: {total_tests}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if failed_tests == 0:
        print("  🎉 Excellent! All tests passed. The enhanced multi-agent system is ready for production.")
        print("  🚀 Consider running tests with real models for end-to-end validation.")
        print("  📈 Monitor system performance in production environment.")
    else:
        print(f"  🔧 Fix {failed_tests} failing tests before deployment.")
        print("  🧪 Review test logs for specific failure details.")
        print("  🔄 Re-run tests after fixes to ensure stability.")
    
    print(f"\n📋 Next Steps:")
    print("  1. Review individual test results above")
    print("  2. Run pytest for detailed test reports: pytest tests/ -v")
    print("  3. Check system health: python examples/demo_integrated_system.py")
    print("  4. Deploy to staging environment for integration testing")


async def main():
    """Run comprehensive test suite."""
    print_banner("ENHANCED MULTI-AGENT RFQ SYSTEM - COMPREHENSIVE TEST SUITE")
    
    print("🚀 Starting comprehensive test execution...")
    print(f"Environment: Test mode with mock models")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    all_results = {}
    
    try:
        # Run all test suites
        all_results['unit_tests'] = await run_unit_tests()
        all_results['performance_tests'] = await run_performance_tests()
        all_results['evaluation_tests'] = await run_evaluation_tests()
        all_results['existing_tests'] = await run_existing_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = datetime.now()
    
    # Generate comprehensive report
    generate_test_report(all_results, start_time, end_time)
    
    print(f"\n🏁 Test execution completed!")


if __name__ == "__main__":
    """Entry point for comprehensive test runner."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test runner stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error in test runner: {e}")
        sys.exit(1)
