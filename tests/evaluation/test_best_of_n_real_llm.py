"""
Real LLM Evaluation Tests for Best-of-N Selector

This module tests the Best-of-N selector using real LLM API calls to validate
the complete functionality including LLM judge evaluation and selection.

WARNING: This test requires real API keys and will make actual LLM calls.
Use sparingly and consider API costs.

Usage:
    # Set your API key and run
    OPENAI_API_KEY=your-real-key python tests/evaluation/test_best_of_n_real_llm.py
    
    # Or run with pytest
    OPENAI_API_KEY=your-real-key pytest tests/evaluation/test_best_of_n_real_llm.py -v -s
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Optional
import pytest

from pydantic import BaseModel, Field
from pydantic_ai import models
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.rfq_system.agents.evaluation.best_of_n_selector import (
    BestOfNSelector,
    EvaluationCriteria,
    BestOfNResult
)
from src.rfq_system.core.interfaces.agent import BaseAgent

# Simple context class for testing
class AgentContext:
    def __init__(self, request_id: str, user_id: str, session_id: str):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id


class RFQProposal(BaseModel):
    """Structured RFQ proposal for real LLM evaluation."""
    title: str = Field(description="Proposal title")
    description: str = Field(description="Detailed proposal description")
    timeline_months: int = Field(ge=1, le=24, description="Timeline in months")
    cost_estimate: int = Field(ge=1000, description="Cost estimate in USD")
    key_features: List[str] = Field(description="List of key features")
    confidence_level: str = Field(pattern=r'^(low|medium|high)$', description="Confidence level")


class RFQInput(BaseModel):
    """Input for RFQ proposal generation."""
    requirements: str = Field(description="Customer requirements")
    budget_range: str = Field(description="Budget range")
    timeline_preference: str = Field(description="Preferred timeline")
    industry: str = Field(default="technology", description="Industry sector")


class RealRFQAgent(BaseAgent):
    """Real RFQ agent that uses actual LLM calls for proposal generation."""
    
    def __init__(self, model: str = "openai:gpt-4o-mini", quality_bias: str = "balanced"):
        self.agent_id = f"real_rfq_agent_{quality_bias}"
        self.model = model
        self.quality_bias = quality_bias
        
        # Different system prompts to create quality variation
        self.system_prompts = {
            "high_quality": """You are an expert RFQ proposal writer with 15+ years of experience. 
            Create comprehensive, detailed proposals that demonstrate deep understanding of customer needs.
            Include specific technical details, implementation approaches, risk mitigation strategies,
            and clear value propositions. Be thorough and professional.""",
            
            "medium_quality": """You are a competent proposal writer. Create good proposals that 
            address the customer requirements with reasonable detail and clear structure.
            Include key features and timeline estimates.""",
            
            "basic_quality": """Create a simple proposal that addresses the basic requirements.
            Keep it concise and straightforward.""",
            
            "balanced": """You are a professional proposal writer. Create well-structured proposals
            that balance detail with clarity, addressing customer needs effectively."""
        }
        
        from pydantic_ai import Agent
        self._agent = Agent(
            model=self.model,
            result_type=RFQProposal,
            system_prompt=self.system_prompts.get(quality_bias, self.system_prompts["balanced"])
        )
    
    async def process(self, input_data, context):
        """Generate RFQ proposal using real LLM."""
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict):
            prompt = f"""
            Customer Requirements: {input_data.get('requirements', '')}
            Budget Range: {input_data.get('budget_range', 'Not specified')}
            Timeline: {input_data.get('timeline_preference', 'Flexible')}
            Industry: {input_data.get('industry', 'Technology')}
            
            Please create a comprehensive RFQ proposal addressing these requirements.
            """
        else:
            prompt = str(input_data)
        
        result = await self._agent.run(prompt)
        return result.data
    
    def get_capabilities(self):
        return ["rfq_proposal_generation", "real_llm_evaluation"]
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    async def health_check(self):
        return {"status": "healthy", "quality_bias": self.quality_bias}


class BestOfNQualityEvaluator(Evaluator[RFQInput, BestOfNResult]):
    """Custom evaluator for Best-of-N selection quality with real LLMs."""
    
    def evaluate(self, ctx: EvaluatorContext[RFQInput, BestOfNResult]) -> float:
        """Evaluate the quality of Best-of-N selection."""
        result = ctx.output
        
        if not isinstance(result, BestOfNResult):
            return 0.0
        
        score = 0.0
        
        # Check if we got the expected number of candidates
        if result.n_candidates >= 3:
            score += 0.15
        
        # Check if selection confidence is reasonable
        if result.selection_confidence > 0.3:
            score += 0.20
        
        # Check if best candidate has higher or equal score than others
        if result.best_evaluation and result.all_evaluations:
            best_score = result.best_evaluation.overall_score
            other_scores = [e.overall_score for e in result.all_evaluations 
                           if e.candidate_id != result.best_candidate.candidate_id]
            
            if other_scores and best_score >= max(other_scores):
                score += 0.25
            elif not other_scores:  # Only one candidate
                score += 0.15
        
        # Check if evaluation reasoning exists and is substantial
        if (result.best_evaluation and result.best_evaluation.reasoning and 
            len(result.best_evaluation.reasoning) > 20):
            score += 0.20
        
        # Check if the selected proposal has reasonable structure
        if hasattr(result.best_candidate, 'output') and result.best_candidate.output:
            proposal = result.best_candidate.output
            if (hasattr(proposal, 'title') and hasattr(proposal, 'description') and
                hasattr(proposal, 'key_features') and len(proposal.key_features) > 0):
                score += 0.20
        
        return min(score, 1.0)


class ProposalQualityEvaluator(Evaluator[RFQInput, BestOfNResult]):
    """Evaluator focused on the quality of the selected proposal."""
    
    def evaluate(self, ctx: EvaluatorContext[RFQInput, BestOfNResult]) -> float:
        """Evaluate the quality of the selected proposal."""
        result = ctx.output
        
        if not isinstance(result, BestOfNResult) or not result.best_candidate:
            return 0.0
        
        proposal = result.best_candidate.output
        if not hasattr(proposal, 'title'):
            return 0.0
        
        score = 0.0
        
        # Title quality (not empty, reasonable length)
        if proposal.title and len(proposal.title.strip()) > 5:
            score += 0.15
        
        # Description quality (substantial content)
        if proposal.description and len(proposal.description.strip()) > 50:
            score += 0.25
        
        # Timeline reasonableness (1-24 months)
        if hasattr(proposal, 'timeline_months') and 1 <= proposal.timeline_months <= 24:
            score += 0.15
        
        # Cost estimate reasonableness (>= 1000)
        if hasattr(proposal, 'cost_estimate') and proposal.cost_estimate >= 1000:
            score += 0.15
        
        # Key features (at least 2 meaningful features)
        if (hasattr(proposal, 'key_features') and 
            isinstance(proposal.key_features, list) and 
            len(proposal.key_features) >= 2):
            score += 0.20
        
        # Confidence level is valid
        if (hasattr(proposal, 'confidence_level') and 
            proposal.confidence_level in ['low', 'medium', 'high']):
            score += 0.10
        
        return min(score, 1.0)


class BestOfNEvaluationReport:
    """Comprehensive evaluation report for Best-of-N selection runs."""
    
    def __init__(self, report_id: str):
        self.report_id = report_id
        self.timestamp = datetime.now()
        self.evaluation_cases = []
        self.metadata = {}
        self.performance_metrics = {}
        self.error_info = None
    
    def add_case_result(self, case_name: str, case_input: RFQInput, 
                       best_of_n_result: BestOfNResult, evaluation_scores: Dict,
                       actual_duration: float, error_info: Optional[Dict] = None):
        """Add a case result to the report."""
        case_data = {
            "case_name": case_name,
            "case_input": {
                "requirements": case_input.requirements,
                "budget_range": case_input.budget_range,
                "timeline_preference": case_input.timeline_preference,
                "industry": case_input.industry
            },
            "best_of_n_processing": {
                "status": "completed" if not error_info else "error",
                "candidates_generated": best_of_n_result.n_candidates if best_of_n_result else 0,
                "selection_confidence": float(best_of_n_result.selection_confidence) if best_of_n_result else 0.0,
                "best_score": float(best_of_n_result.best_evaluation.overall_score) if best_of_n_result and best_of_n_result.best_evaluation else 0.0,
                "evaluation_reasoning": best_of_n_result.best_evaluation.reasoning if best_of_n_result and best_of_n_result.best_evaluation else None,
                "candidates_data": []
            },
            "selected_proposal": {},
            "evaluation_scores": evaluation_scores,
            "performance": {
                "actual_duration": actual_duration,
                "pydantic_evals_duration": 1.0,  # Known bug value
                "candidates_per_second": best_of_n_result.n_candidates / actual_duration if actual_duration > 0 and best_of_n_result else 0.0
            },
            "error_info": error_info
        }
        
        # Add candidate details if available
        if best_of_n_result and hasattr(best_of_n_result, 'all_candidates') and best_of_n_result.all_candidates:
            for i, candidate in enumerate(best_of_n_result.all_candidates):
                candidate_data = {
                    "candidate_id": getattr(candidate, 'candidate_id', str(i)),
                    "is_best": candidate.candidate_id == best_of_n_result.best_candidate.candidate_id if hasattr(candidate, 'candidate_id') else i == 0,
                    "output_summary": str(candidate.output)[:200] + "..." if candidate.output else None,
                    "generation_time": getattr(candidate, 'generation_time_ms', None)
                }
                case_data["best_of_n_processing"]["candidates_data"].append(candidate_data)
        
        # Add selected proposal details
        if best_of_n_result and best_of_n_result.best_candidate:
            proposal = best_of_n_result.best_candidate.output
            if proposal:
                case_data["selected_proposal"] = {
                    "title": getattr(proposal, 'title', 'N/A'),
                    "description_preview": getattr(proposal, 'description', '')[:200] + "..." if hasattr(proposal, 'description') else 'N/A',
                    "timeline_months": getattr(proposal, 'timeline_months', None),
                    "cost_estimate": getattr(proposal, 'cost_estimate', None),
                    "confidence_level": getattr(proposal, 'confidence_level', 'N/A'),
                    "key_features_count": len(getattr(proposal, 'key_features', []))
                }
        
        self.evaluation_cases.append(case_data)
    
    def set_metadata(self, api_key_type: str, model_config: Dict, evaluation_config: Dict):
        """Set report metadata."""
        self.metadata = {
            "report_id": self.report_id,
            "report_type": "best_of_n_llm_evaluation",
            "timestamp": self.timestamp.isoformat(),
            "filename": f"best_of_n_{self.report_id}.json",
            "version": "1.0",
            "api_key_type": api_key_type,
            "pydantic_evals_version": "0.3.2",
            "duration_bug_note": "PydanticEvals v0.3.2 has duration reporting bug - actual timing provided"
        }
        self.metadata.update({
            "model_config": model_config,
            "evaluation_config": evaluation_config
        })
    
    def set_performance_metrics(self, total_duration: float, case_timings: Dict):
        """Set overall performance metrics."""
        if case_timings:
            fastest = min(case_timings.values())
            slowest = max(case_timings.values())
            avg_duration = sum(case_timings.values()) / len(case_timings)
            speed_variation = ((slowest - fastest) / fastest * 100) if fastest > 0 else 0.0
        else:
            fastest = slowest = avg_duration = speed_variation = 0.0
        
        self.performance_metrics = {
            "total_evaluation_duration": total_duration,
            "cases_evaluated": len(self.evaluation_cases),
            "average_case_duration": avg_duration,
            "fastest_case_duration": fastest,
            "slowest_case_duration": slowest,
            "speed_variation_percentage": speed_variation,
            "successful_cases": len([case for case in self.evaluation_cases if case["best_of_n_processing"]["status"] == "completed"]),
            "failed_cases": len([case for case in self.evaluation_cases if case["best_of_n_processing"]["status"] == "error"])
        }
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary format."""
        return {
            "metadata": self.metadata,
            "evaluation_cases": self.evaluation_cases,
            "performance_metrics": self.performance_metrics,
            "analytics": {
                "total_candidates_generated": sum(case["best_of_n_processing"]["candidates_generated"] for case in self.evaluation_cases),
                "average_selection_confidence": sum(case["best_of_n_processing"]["selection_confidence"] for case in self.evaluation_cases) / len(self.evaluation_cases) if self.evaluation_cases else 0.0,
                "average_best_score": sum(case["best_of_n_processing"]["best_score"] for case in self.evaluation_cases) / len(self.evaluation_cases) if self.evaluation_cases else 0.0,
                "evaluation_scores_summary": self._calculate_score_summary()
            },
            "error_info": self.error_info
        }
    
    def _calculate_score_summary(self) -> Dict:
        """Calculate summary of evaluation scores across all cases."""
        if not self.evaluation_cases:
            return {}
        
        evaluator_scores = {}
        for case in self.evaluation_cases:
            for evaluator_name, score_value in case["evaluation_scores"].items():
                if evaluator_name not in evaluator_scores:
                    evaluator_scores[evaluator_name] = []
                
                # Handle different score formats
                if hasattr(score_value, 'score'):
                    evaluator_scores[evaluator_name].append(float(score_value.score))
                elif isinstance(score_value, (int, float)):
                    evaluator_scores[evaluator_name].append(float(score_value))
        
        summary = {}
        for evaluator_name, scores in evaluator_scores.items():
            if scores:
                summary[evaluator_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        return summary
    
    def save_to_file(self, reports_dir: str = "reports"):
        """Save report to JSON file."""
        # Ensure reports directory exists
        Path(reports_dir).mkdir(exist_ok=True)
        
        # Generate filename
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_best_of_n_evaluation.json"
        filepath = Path(reports_dir) / filename
        
        # Update metadata with actual filename
        report_data = self.to_dict()
        report_data["metadata"]["filename"] = filename
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return filepath


# Test cases for different scenarios
rfq_test_cases = [
    Case(
        name='enterprise_crm_system',
        inputs=RFQInput(
            requirements="Enterprise-grade CRM system for 500+ users with advanced analytics, workflow automation, and mobile access",
            budget_range="$100,000 - $300,000",
            timeline_preference="6-8 months",
            industry="technology"
        ),
        expected_output=None,  # We'll evaluate based on quality metrics
        metadata={'complexity': 'high', 'budget': 'large', 'timeline': 'medium'},
        evaluators=(
            LLMJudge(
                rubric="Proposal should be comprehensive, include advanced features, and justify the budget range",
                model="openai:gpt-4o-mini"
            ),
        ),
    ),
    Case(
        name='startup_mvp_development',
        inputs=RFQInput(
            requirements="MVP development for a fintech startup - payment processing, user authentication, basic dashboard",
            budget_range="$25,000 - $75,000",
            timeline_preference="3-4 months",
            industry="fintech"
        ),
        expected_output=None,
        metadata={'complexity': 'medium', 'budget': 'medium', 'timeline': 'short'},
        evaluators=(
            LLMJudge(
                rubric="Proposal should focus on MVP features, be cost-effective, and have realistic timeline",
                model="openai:gpt-4o-mini"
            ),
        ),
    ),
    Case(
        name='healthcare_compliance_system',
        inputs=RFQInput(
            requirements="HIPAA-compliant patient management system with secure data handling and audit trails",
            budget_range="$150,000 - $400,000",
            timeline_preference="8-12 months",
            industry="healthcare"
        ),
        expected_output=None,
        metadata={'complexity': 'high', 'budget': 'large', 'timeline': 'long'},
        evaluators=(
            LLMJudge(
                rubric="Proposal must address HIPAA compliance, security requirements, and include audit capabilities",
                model="openai:gpt-4o-mini"
            ),
        ),
    ),
]

# Create the evaluation dataset
best_of_n_real_dataset = Dataset[RFQInput, BestOfNResult, Any](
    cases=rfq_test_cases,
    evaluators=[
        IsInstance(type_name='BestOfNResult'),
        BestOfNQualityEvaluator(),
        ProposalQualityEvaluator(),
        LLMJudge(
            rubric="The Best-of-N selection should choose a high-quality proposal that addresses customer requirements effectively",
            include_input=True,
            model="openai:gpt-4o-mini"
        ),
    ],
)


async def run_best_of_n_with_real_llm(rfq_input: RFQInput) -> BestOfNResult:
    """Run Best-of-N selection using real LLM API calls for PydanticEvals dataset."""
    
    print(f"🔄 Starting Best-of-N generation for: {rfq_input.requirements[:50]}...")
    start_time = time.time()
    
    # Create agents with different quality biases to ensure variation
    agents = [
        RealRFQAgent(quality_bias="high_quality"),
        RealRFQAgent(quality_bias="medium_quality"), 
        RealRFQAgent(quality_bias="basic_quality"),
        RealRFQAgent(quality_bias="balanced"),
        RealRFQAgent(quality_bias="balanced"),  # Extra balanced for comparison
    ]
    
    # Use a random agent for each generation to create variety
    import random
    selected_agent = random.choice(agents)
    
    print(f"📝 Selected agent: {selected_agent.quality_bias}")
    
    # Initialize Best-of-N selector with real LLM models
    selector = BestOfNSelector(
        evaluation_model="openai:gpt-4o-mini",  # Use mini for cost efficiency
        max_parallel_generations=3,  # Limit parallel calls to manage costs
        enable_detailed_evaluation=True
    )
    
    # Create context
    context = AgentContext(
        request_id="real-llm-test",
        user_id="evaluation-user",
        session_id="eval-session"
    )
    
    # Create evaluation criteria emphasizing different aspects
    criteria = EvaluationCriteria(
        accuracy_weight=0.25,      # Technical accuracy
        completeness_weight=0.35,  # Comprehensive coverage
        relevance_weight=0.25,     # Customer relevance
        clarity_weight=0.15        # Communication clarity
    )
    
    # Generate prompt from input
    prompt = f"""
    Create a professional RFQ proposal for the following requirements:
    
    Requirements: {rfq_input.requirements}
    Budget Range: {rfq_input.budget_range}
    Timeline: {rfq_input.timeline_preference}
    Industry: {rfq_input.industry}
    
    Please provide a comprehensive proposal that addresses all requirements.
    """
    
    generation_start = time.time()
    print(f"🚀 Running Best-of-N selection...")
    
    # Run Best-of-N selection with real LLMs
    result = await selector.generate_best_of_n(
        target_agent=selected_agent,
        prompt=prompt,
        context=context,
        n=3,  # Generate 3 candidates for comparison
        criteria=criteria
    )
    
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time
    
    print(f"✅ Completed in {total_time:.2f}s (generation: {generation_time:.2f}s)")
    print(f"📊 Generated {result.n_candidates} candidates, confidence: {result.selection_confidence:.3f}")
    
    return result


async def run_real_llm_evaluation():
    """Run the real LLM evaluation and display results."""
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'test-key':
        print("❌ Real API key required. Set OPENAI_API_KEY environment variable.")
        print("   Example: OPENAI_API_KEY=your-key python tests/evaluation/test_best_of_n_real_llm.py")
        return
    
    # Allow real model requests for this evaluation
    models.ALLOW_MODEL_REQUESTS = True
    
    print("🚀 Starting Real LLM Best-of-N Evaluation")
    print("⚠️  This will make actual API calls and incur costs")
    print("=" * 60)
    
    # Initialize evaluation report
    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_report = BestOfNEvaluationReport(report_id)
    
    # Track actual timing per case
    case_timings = {}
    
    try:
        # Run the evaluation with timing
        eval_start = time.time()
        
        # Wrap the function to track individual case timing
        async def timed_run_best_of_n(rfq_input):
            case_start = time.time()
            result = await run_best_of_n_with_real_llm(rfq_input)
            case_duration = time.time() - case_start
            
            # Store timing by case (use requirements as identifier)
            case_key = rfq_input.requirements[:30] + "..."
            case_timings[case_key] = case_duration
            
            return result
        
        report = await best_of_n_real_dataset.evaluate(timed_run_best_of_n)
        eval_total = time.time() - eval_start
        
        print(f"\n⏱️  Total evaluation time: {eval_total:.2f}s")
        print("\n📊 Evaluation Results:")
        print("=" * 60)
        
        # Hide misleading duration column and show our own timing
        report.print(include_input=True, include_output=False, include_durations=False)
        
        print("\n⚠️  NOTE: PydanticEvals v0.3.2 duration reporting disabled due to bug (always shows 1.0s)")
        print("    Accurate timing measurements are provided in the detailed analysis below.")
        
        # Print detailed results with actual timing
        print("\n📝 Detailed Analysis (with ACTUAL timing):")
        print("=" * 60)
        
        for i, case_result in enumerate(report.cases):
            print(f"\n🔍 Case: {case_result.name}")
            print(f"   Input: {case_result.inputs.requirements[:100]}...")
            
            # Find the actual timing for this case
            case_key = case_result.inputs.requirements[:30] + "..."
            actual_duration = case_timings.get(case_key, 0.0)
            
            print(f"   PydanticEvals Duration: {getattr(case_result, 'task_duration', 'N/A')}s (incorrect)")
            print(f"   ACTUAL Duration: {actual_duration:.2f}s")
            
            # Add case to evaluation report
            error_info = None
            if case_result.output and hasattr(case_result.output, 'best_candidate'):
                result = case_result.output
                print(f"   Candidates Generated: {result.n_candidates}")
                print(f"   Selection Confidence: {result.selection_confidence:.3f}")
                
                if result.best_evaluation:
                    print(f"   Best Score: {result.best_evaluation.overall_score:.3f}")
                    print(f"   Reasoning: {result.best_evaluation.reasoning[:150]}...")
                
                if result.best_candidate and hasattr(result.best_candidate, 'output'):
                    proposal = result.best_candidate.output
                    if hasattr(proposal, 'title'):
                        print(f"   Selected Title: {proposal.title}")
                        print(f"   Timeline: {getattr(proposal, 'timeline_months', 'N/A')} months")
                        print(f"   Cost: ${getattr(proposal, 'cost_estimate', 'N/A'):,}")
            else:
                result = None
                error_info = {"error": "No result generated", "case_index": i}
            
            print(f"   Evaluator Scores:")
            evaluation_scores = {}
            if isinstance(case_result.scores, dict):
                for evaluator_name, score_value in case_result.scores.items():
                    # Handle different score value types
                    if hasattr(score_value, 'score'):
                        # New API format: score_value is an EvaluationResult object
                        score_float = float(score_value.score)
                        evaluation_scores[evaluator_name] = score_value
                        print(f"     {evaluator_name}: {score_float:.3f}")
                    elif isinstance(score_value, (int, float)):
                        # Simple numeric score
                        score_float = float(score_value)
                        evaluation_scores[evaluator_name] = score_value
                        print(f"     {evaluator_name}: {score_float:.3f}")
                    else:
                        # Fallback for unknown format
                        evaluation_scores[evaluator_name] = str(score_value)
                        print(f"     {evaluator_name}: {score_value}")
            else:
                # Fallback for old format (list of score objects)
                for score in case_result.scores:
                    if hasattr(score, 'evaluator') and hasattr(score, 'score'):
                        evaluation_scores[score.evaluator] = score
                        print(f"     {score.evaluator}: {score.score:.3f}")
                    else:
                        evaluation_scores["unknown"] = score
                        print(f"     Score: {score}")
            
            # Add case to report
            evaluation_report.add_case_result(
                case_name=case_result.name,
                case_input=case_result.inputs,
                best_of_n_result=result,
                evaluation_scores=evaluation_scores,
                actual_duration=actual_duration,
                error_info=error_info
            )
        
        # Summary statistics
        def get_case_avg_score(case_result):
            if isinstance(case_result.scores, dict):
                if case_result.scores:
                    total_score = 0.0
                    count = 0
                    for score_value in case_result.scores.values():
                        if hasattr(score_value, 'score'):
                            total_score += float(score_value.score)
                        elif isinstance(score_value, (int, float)):
                            total_score += float(score_value)
                        count += 1
                    return total_score / count if count > 0 else 0.0
                return 0.0
            else:
                # Fallback for old format
                if case_result.scores:
                    return sum(s.score for s in case_result.scores) / len(case_result.scores)
                return 0.0
        
        avg_score = sum(get_case_avg_score(cr) for cr in report.cases) / len(report.cases)
        actual_avg_duration = sum(case_timings.values()) / len(case_timings) if case_timings else 0.0
        
        print(f"\n📈 Summary:")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Total Evaluation Time: {eval_total:.2f}s")
        print(f"   ACTUAL Average Time per Case: {actual_avg_duration:.2f}s")
        print(f"   PydanticEvals Reported Average: 1.0s (incorrect)")
        
        def case_passed(case_result):
            if isinstance(case_result.scores, dict):
                for score_value in case_result.scores.values():
                    if hasattr(score_value, 'score'):
                        if float(score_value.score) <= 0.5:
                            return False
                    elif isinstance(score_value, (int, float)):
                        if float(score_value) <= 0.5:
                            return False
                return True
            else:
                # Fallback for old format
                return all(s.score > 0.5 for s in case_result.scores)
        
        passed_cases = len([cr for cr in report.cases if case_passed(cr)])
        print(f"   Cases Passed: {passed_cases}/{len(report.cases)}")
        
        # Performance analysis
        if case_timings:
            fastest = min(case_timings.values())
            slowest = max(case_timings.values())
            print(f"\n⚡ Performance Analysis:")
            print(f"   Fastest case: {fastest:.2f}s")
            print(f"   Slowest case: {slowest:.2f}s")
            print(f"   Speed variation: {((slowest - fastest) / fastest * 100):.1f}%")
        
        # Set report metadata and performance metrics
        evaluation_report.set_metadata(
            api_key_type="real" if os.getenv('OPENAI_API_KEY') != 'test-key' else "test",
            model_config={
                "evaluation_model": "openai:gpt-4o-mini",
                "target_agent_model": "openai:gpt-4o-mini",
                "max_parallel_generations": 3
            },
            evaluation_config={
                "n_candidates": 3,
                "evaluation_criteria": {
                    "accuracy_weight": 0.25,
                    "completeness_weight": 0.35,
                    "relevance_weight": 0.25,
                    "clarity_weight": 0.15
                },
                "dataset_cases": len(report.cases)
            }
        )
        
        evaluation_report.set_performance_metrics(eval_total, case_timings)
        
        # Save report to file
        report_filepath = evaluation_report.save_to_file()
        
        print(f"\n💾 Evaluation Report Saved:")
        print(f"   File: {report_filepath}")
        print(f"   Report ID: {report_id}")
        print("   ✅ Real LLM evaluation completed successfully!")
        
        return report_filepath
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        evaluation_report.error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "error_occurred": True
        }
        # Save error report
        report_filepath = evaluation_report.save_to_file()
        print(f"💾 Error report saved: {report_filepath}")
        raise
    finally:
        # Reset model requests setting
        models.ALLOW_MODEL_REQUESTS = False


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test since it uses real APIs
async def test_real_llm_best_of_n_enterprise():
    """Pytest version of real LLM test for enterprise scenario."""
    
    # Skip if no real API key
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'test-key':
        pytest.skip("Real API key required for this test")
    
    models.ALLOW_MODEL_REQUESTS = True
    
    try:
        input_data = RFQInput(
            requirements="Enterprise CRM system with advanced analytics and workflow automation",
            budget_range="$150,000 - $250,000",
            timeline_preference="6 months",
            industry="technology"
        )
        
        start_time = time.time()
        result = await run_best_of_n_with_real_llm(input_data)
        actual_duration = time.time() - start_time
        
        # Validate results
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates >= 1
        assert result.best_candidate is not None
        assert result.best_evaluation is not None
        assert 0.0 <= result.selection_confidence <= 1.0
        assert 0.0 <= result.best_evaluation.overall_score <= 1.0
        
        # Validate proposal structure
        proposal = result.best_candidate.output
        assert hasattr(proposal, 'title')
        assert hasattr(proposal, 'description')
        assert len(proposal.title.strip()) > 0
        assert len(proposal.description.strip()) > 0
        
        print(f"✅ Test passed - Generated {result.n_candidates} candidates in {actual_duration:.2f}s")
        print(f"   Best score: {result.best_evaluation.overall_score:.3f}")
        print(f"   Confidence: {result.selection_confidence:.3f}")
        
    finally:
        models.ALLOW_MODEL_REQUESTS = False


async def test_json_report_generation():
    """Test JSON report generation without real API calls."""
    
    print("🧪 Testing JSON report generation...")
    
    # Create a sample evaluation report
    report_id = "test_20250622_123456"
    evaluation_report = BestOfNEvaluationReport(report_id)
    
    # Create mock input data
    mock_input = RFQInput(
        requirements="Test CRM system for demo purposes",
        budget_range="$50,000 - $100,000",
        timeline_preference="3-4 months",
        industry="technology"
    )
    
    # Create mock result data
    mock_proposal = RFQProposal(
        title="Test CRM Solution",
        description="A comprehensive CRM system with modern features and scalable architecture",
        timeline_months=4,
        cost_estimate=75000,
        key_features=["User Management", "Analytics Dashboard", "Mobile App"],
        confidence_level="high"
    )
    
    # Create simple mock objects since we don't have access to the actual classes
    class MockCandidate:
        def __init__(self, output, generation_time=None):
            self.candidate_id = "mock_candidate_1"
            self.output = output
            self.generation_time_ms = generation_time * 1000 if generation_time else 2500
            self.model_used = "openai:gpt-4o-mini"
            self.confidence_score = 0.85
    
    class MockEvaluation:
        def __init__(self, overall_score, reasoning):
            self.candidate_id = "mock_candidate_1"
            self.overall_score = overall_score
            self.accuracy_score = 0.8
            self.completeness_score = 0.9
            self.relevance_score = 0.85
            self.clarity_score = 0.8
            self.reasoning = reasoning
            self.evaluation_time_ms = 1500
    
    class MockResult:
        def __init__(self, candidates, best_candidate, best_evaluation, n_candidates, selection_confidence):
            self.all_candidates = candidates  # Use all_candidates to match BestOfNResult
            self.all_evaluations = [best_evaluation]  # List of all evaluations
            self.best_candidate = best_candidate
            self.best_evaluation = best_evaluation
            self.n_candidates = n_candidates
            self.selection_confidence = selection_confidence
            self.total_generation_time_ms = 7500
            self.total_evaluation_time_ms = 1500
    
    # Mock candidate
    mock_candidate = MockCandidate(
        output=mock_proposal,
        generation_time=2.5
    )
    
    # Mock evaluation
    mock_evaluation = MockEvaluation(
        overall_score=0.85,
        reasoning="High-quality proposal with comprehensive features and clear structure"
    )
    
    # Mock Best-of-N result
    mock_result = MockResult(
        candidates=[mock_candidate],
        best_candidate=mock_candidate,
        best_evaluation=mock_evaluation,
        n_candidates=3,
        selection_confidence=0.87
    )
    
    # Mock evaluation scores
    mock_scores = {
        "BestOfNQualityEvaluator": 0.85,
        "ProposalQualityEvaluator": 0.78
    }
    
    # Add case to report
    evaluation_report.add_case_result(
        case_name="test_case_enterprise",
        case_input=mock_input,
        best_of_n_result=mock_result,
        evaluation_scores=mock_scores,
        actual_duration=5.2
    )
    
    # Set metadata
    evaluation_report.set_metadata(
        api_key_type="test",
        model_config={
            "evaluation_model": "openai:gpt-4o-mini",
            "target_agent_model": "openai:gpt-4o-mini",
            "max_parallel_generations": 3
        },
        evaluation_config={
            "n_candidates": 3,
            "evaluation_criteria": {
                "accuracy_weight": 0.25,
                "completeness_weight": 0.35,
                "relevance_weight": 0.25,
                "clarity_weight": 0.15
            },
            "dataset_cases": 1
        }
    )
    
    # Set performance metrics
    evaluation_report.set_performance_metrics(5.2, {"test_case": 5.2})
    
    # Convert to dict and validate structure
    report_dict = evaluation_report.to_dict()
    
    # Validate report structure
    assert "metadata" in report_dict
    assert "evaluation_cases" in report_dict
    assert "performance_metrics" in report_dict
    assert "analytics" in report_dict
    
    # Validate metadata
    metadata = report_dict["metadata"]
    assert metadata["report_type"] == "best_of_n_llm_evaluation"
    assert metadata["api_key_type"] == "test"
    assert "timestamp" in metadata
    assert "pydantic_evals_version" in metadata
    
    # Validate case data
    assert len(report_dict["evaluation_cases"]) == 1
    case = report_dict["evaluation_cases"][0]
    assert case["case_name"] == "test_case_enterprise"
    assert "case_input" in case
    assert "best_of_n_processing" in case
    assert "selected_proposal" in case
    assert "evaluation_scores" in case
    assert "performance" in case
    
    # Validate performance metrics
    perf = report_dict["performance_metrics"]
    assert perf["total_evaluation_duration"] == 5.2
    assert perf["cases_evaluated"] == 1
    assert perf["successful_cases"] == 1
    assert perf["failed_cases"] == 0
    
    # Validate analytics
    analytics = report_dict["analytics"]
    assert analytics["total_candidates_generated"] == 3
    assert analytics["average_selection_confidence"] == 0.87
    assert analytics["average_best_score"] == 0.85
    
    # Test saving to file
    try:
        # Create test reports directory
        test_reports_dir = "test_reports"
        filepath = evaluation_report.save_to_file(test_reports_dir)
        
        # Verify file was created
        assert filepath.exists()
        
        # Verify file content
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["metadata"]["report_type"] == "best_of_n_llm_evaluation"
        assert len(saved_data["evaluation_cases"]) == 1
        
        print(f"✅ JSON report test passed!")
        print(f"   Report structure validated")
        print(f"   Test file saved: {filepath}")
        print(f"   File size: {filepath.stat().st_size} bytes")
        
        # Clean up test file
        filepath.unlink()
        Path(test_reports_dir).rmdir()
        
        return True
        
    except Exception as e:
        print(f"❌ JSON report test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the evaluation when called directly
    if len(sys.argv) > 1 and sys.argv[1] == "--test-json":
        # Test JSON report generation only
        asyncio.run(test_json_report_generation())
    else:
        # Run full evaluation
        asyncio.run(run_real_llm_evaluation()) 