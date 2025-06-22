#!/usr/bin/env python3
"""
Generate Sample Best-of-N Evaluation Report

This script generates a comprehensive sample JSON report that demonstrates
the structure and content of Best-of-N LLM evaluation reports.

This shows what the real evaluation would generate without requiring API calls.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "tests" / "evaluation"))

from test_best_of_n_real_llm import BestOfNEvaluationReport, RFQInput, RFQProposal


def create_sample_evaluation_report():
    """Create a comprehensive sample evaluation report."""
    
    print("🏗️  Generating Sample Best-of-N Evaluation Report")
    print("=" * 60)
    
    # Create report
    report_id = "sample_20250622_demo"
    evaluation_report = BestOfNEvaluationReport(report_id)
    
    # Sample cases data
    sample_cases = [
        {
            "name": "enterprise_crm_system",
            "input": RFQInput(
                requirements="Enterprise-grade CRM system for 500+ users with advanced analytics, workflow automation, and mobile access",
                budget_range="$100,000 - $300,000",
                timeline_preference="6-8 months",
                industry="technology"
            ),
            "proposal": RFQProposal(
                title="Enterprise CRM Solution with Advanced Analytics",
                description="A comprehensive customer relationship management system designed for enterprise-scale operations. Features include advanced analytics dashboard, automated workflow engine, mobile-first design, and seamless integration capabilities. Built with scalability and security in mind.",
                timeline_months=7,
                cost_estimate=185000,
                key_features=[
                    "Advanced Analytics Dashboard",
                    "Automated Workflow Engine", 
                    "Mobile-First Design",
                    "API Integration Hub",
                    "Role-Based Access Control",
                    "Real-time Reporting"
                ],
                confidence_level="high"
            ),
            "candidates_generated": 5,
            "selection_confidence": 0.89,
            "best_score": 0.92,
            "evaluation_reasoning": "Excellent proposal with comprehensive feature set, realistic timeline, and strong technical approach. Demonstrates deep understanding of enterprise requirements.",
            "duration": 12.3,
            "scores": {
                "BestOfNQualityEvaluator": 0.88,
                "ProposalQualityEvaluator": 0.91
            }
        },
        {
            "name": "startup_mvp_development", 
            "input": RFQInput(
                requirements="MVP development for a fintech startup - payment processing, user authentication, basic dashboard",
                budget_range="$25,000 - $75,000",
                timeline_preference="3-4 months",
                industry="fintech"
            ),
            "proposal": RFQProposal(
                title="Fintech MVP with Core Payment Features",
                description="Lean but robust MVP focusing on essential fintech functionality. Includes secure payment processing, user authentication system, and intuitive dashboard. Built with modern technologies for rapid scaling.",
                timeline_months=4,
                cost_estimate=58000,
                key_features=[
                    "Secure Payment Processing",
                    "Multi-Factor Authentication",
                    "User Dashboard",
                    "Transaction History",
                    "Basic Analytics"
                ],
                confidence_level="medium"
            ),
            "candidates_generated": 4,
            "selection_confidence": 0.76,
            "best_score": 0.84,
            "evaluation_reasoning": "Well-structured MVP proposal with appropriate scope for startup budget. Good balance of features and timeline.",
            "duration": 9.7,
            "scores": {
                "BestOfNQualityEvaluator": 0.82,
                "ProposalQualityEvaluator": 0.79
            }
        },
        {
            "name": "healthcare_compliance_system",
            "input": RFQInput(
                requirements="HIPAA-compliant patient management system with secure data handling and audit trails",
                budget_range="$150,000 - $400,000",
                timeline_preference="8-12 months",
                industry="healthcare"
            ),
            "proposal": RFQProposal(
                title="HIPAA-Compliant Patient Management Platform",
                description="Comprehensive patient management system built to exceed HIPAA compliance requirements. Features end-to-end encryption, detailed audit trails, role-based access controls, and automated compliance reporting. Designed for healthcare organizations requiring the highest security standards.",
                timeline_months=10,
                cost_estimate=275000,
                key_features=[
                    "HIPAA Compliance Framework",
                    "End-to-End Encryption",
                    "Comprehensive Audit Trails",
                    "Role-Based Access Control",
                    "Automated Compliance Reporting",
                    "Secure Data Backup",
                    "Patient Portal Integration"
                ],
                confidence_level="high"
            ),
            "candidates_generated": 5,
            "selection_confidence": 0.93,
            "best_score": 0.96,
            "evaluation_reasoning": "Outstanding proposal demonstrating deep understanding of healthcare compliance requirements. Comprehensive security approach with detailed implementation plan.",
            "duration": 15.2,
            "scores": {
                "BestOfNQualityEvaluator": 0.94,
                "ProposalQualityEvaluator": 0.93
            }
        }
    ]
    
    # Add cases to report
    for case_data in sample_cases:
        # Create mock result objects
        class MockCandidate:
            def __init__(self, output, generation_time=None):
                self.output = output
                self.generation_time = generation_time
        
        class MockEvaluation:
            def __init__(self, overall_score, reasoning):
                self.overall_score = overall_score
                self.reasoning = reasoning
        
        class MockResult:
            def __init__(self, candidates, best_candidate, best_evaluation, n_candidates, selection_confidence):
                self.candidates = candidates
                self.best_candidate = best_candidate
                self.best_evaluation = best_evaluation
                self.n_candidates = n_candidates
                self.selection_confidence = selection_confidence
        
        # Create mock candidates (simulate multiple candidates)
        candidates = []
        for i in range(case_data["candidates_generated"]):
            candidate = MockCandidate(
                output=case_data["proposal"] if i == 0 else None,  # Only first candidate has full proposal
                generation_time=case_data["duration"] / case_data["candidates_generated"]
            )
            candidates.append(candidate)
        
        # Mock result
        mock_result = MockResult(
            candidates=candidates,
            best_candidate=candidates[0],
            best_evaluation=MockEvaluation(
                overall_score=case_data["best_score"],
                reasoning=case_data["evaluation_reasoning"]
            ),
            n_candidates=case_data["candidates_generated"],
            selection_confidence=case_data["selection_confidence"]
        )
        
        # Add to report
        evaluation_report.add_case_result(
            case_name=case_data["name"],
            case_input=case_data["input"],
            best_of_n_result=mock_result,
            evaluation_scores=case_data["scores"],
            actual_duration=case_data["duration"]
        )
    
    # Set metadata
    evaluation_report.set_metadata(
        api_key_type="sample",
        model_config={
            "evaluation_model": "openai:gpt-4o-mini",
            "target_agent_model": "openai:gpt-4o-mini",
            "max_parallel_generations": 3,
            "quality_bias_variants": ["high_quality", "medium_quality", "basic_quality", "balanced"]
        },
        evaluation_config={
            "n_candidates": 5,
            "evaluation_criteria": {
                "accuracy_weight": 0.25,
                "completeness_weight": 0.35,
                "relevance_weight": 0.25,
                "clarity_weight": 0.15
            },
            "dataset_cases": len(sample_cases),
            "evaluation_framework": "PydanticEvals v0.3.2",
            "custom_evaluators": ["BestOfNQualityEvaluator", "ProposalQualityEvaluator"]
        }
    )
    
    # Set performance metrics
    case_timings = {case["name"]: case["duration"] for case in sample_cases}
    total_duration = sum(case_timings.values())
    evaluation_report.set_performance_metrics(total_duration, case_timings)
    
    # Save report
    filepath = evaluation_report.save_to_file()
    
    print(f"✅ Sample evaluation report generated!")
    print(f"📁 File: {filepath}")
    print(f"📊 Report contains:")
    print(f"   • {len(sample_cases)} evaluation cases")
    print(f"   • {sum(case['candidates_generated'] for case in sample_cases)} total candidates generated")
    print(f"   • {total_duration:.1f}s total evaluation time")
    print(f"   • Performance analysis and quality metrics")
    
    # Display summary
    print(f"\n📈 Evaluation Summary:")
    avg_confidence = sum(case["selection_confidence"] for case in sample_cases) / len(sample_cases)
    avg_score = sum(case["best_score"] for case in sample_cases) / len(sample_cases)
    
    print(f"   Average Selection Confidence: {avg_confidence:.3f}")
    print(f"   Average Best Score: {avg_score:.3f}")
    print(f"   Cases Evaluated: {len(sample_cases)}")
    print(f"   All Cases Passed: ✅ (scores > 0.75)")
    
    # Show file preview
    print(f"\n📋 Report Structure Preview:")
    with open(filepath, 'r') as f:
        report_data = json.load(f)
    
    print(f"   Metadata Keys: {list(report_data['metadata'].keys())}")
    print(f"   Performance Metrics: {list(report_data['performance_metrics'].keys())}")
    print(f"   Analytics Keys: {list(report_data['analytics'].keys())}")
    
    return filepath


def main():
    """Main function."""
    print("🎯 Best-of-N Evaluation Report Generator")
    print("   Creates comprehensive sample reports similar to existing scenario reports")
    print()
    
    try:
        filepath = create_sample_evaluation_report()
        
        print(f"\n💡 Next Steps:")
        print(f"   • View the report: cat {filepath}")
        print(f"   • Compare with existing reports in ./reports")
        print(f"   • Use as template for real evaluations")
        print(f"   • Analyze performance patterns")
        
        print(f"\n🔗 Related Files:")
        print(f"   • Real evaluation: tests/evaluation/test_best_of_n_real_llm.py")
        print(f"   • Demo script: examples/demo_real_llm_evaluation.py")
        print(f"   • Existing reports: ./reports/")
        
    except Exception as e:
        print(f"❌ Failed to generate sample report: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 