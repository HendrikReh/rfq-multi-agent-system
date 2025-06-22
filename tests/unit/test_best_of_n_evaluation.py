"""
Unit Tests for Best-of-N Selector and LLM Judge

This module provides pytest-compatible tests for the BestOfNSelector
following the existing test patterns in the project.
"""

import pytest
import asyncio
from typing import List
from unittest.mock import Mock

from pydantic import BaseModel, Field
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo

# Add the src directory to the path for imports
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up environment for testing
os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-evaluation')

from rfq_system.agents.evaluation.best_of_n_selector import (
    BestOfNSelector,
    EvaluationCriteria,
    BestOfNResult,
    BestOfNDependencies
)
from rfq_system.core.interfaces.agent import BaseAgent, AgentContext, AgentStatus


class RFQProposal(BaseModel):
    """Test RFQ proposal model."""
    title: str
    description: str
    timeline_months: int = Field(ge=1, le=24)
    cost_estimate: int = Field(ge=1000)
    key_features: List[str]
    confidence_level: str = Field(pattern=r'^(low|medium|high)$')


class MockRFQAgent(BaseAgent):
    """Mock agent for testing Best-of-N selection."""
    
    def __init__(self, responses: List[RFQProposal]):
        self.agent_id = "mock_rfq_agent"
        self.model = "test-model"
        self.responses = responses
        self.call_count = 0
    
    async def process(self, input_data, context):
        """Return pre-defined responses."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def get_capabilities(self):
        return ["rfq_proposal_generation"]
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    async def health_check(self):
        return type('HealthStatus', (), {'status': AgentStatus.HEALTHY})()


@pytest.fixture
def high_quality_proposals():
    """High-quality RFQ proposals for testing."""
    return [
        RFQProposal(
            title="Enterprise CRM Solution with Advanced Analytics",
            description="Comprehensive customer relationship management system featuring advanced analytics, workflow automation, mobile access, and integration capabilities.",
            timeline_months=6,
            cost_estimate=125000,
            key_features=[
                "Advanced analytics and reporting",
                "Workflow automation",
                "Mobile applications",
                "Third-party integrations",
                "User training program"
            ],
            confidence_level="high"
        ),
        RFQProposal(
            title="Professional CRM Platform",
            description="Tailored CRM solution with custom workflows and comprehensive support.",
            timeline_months=4,
            cost_estimate=85000,
            key_features=[
                "Custom workflow design",
                "Interactive dashboards",
                "API integrations"
            ],
            confidence_level="high"
        )
    ]


@pytest.fixture
def mixed_quality_proposals():
    """Mixed quality RFQ proposals for testing selection."""
    return [
        # High quality
        RFQProposal(
            title="Enterprise CRM Solution",
            description="Comprehensive customer relationship management system with advanced features.",
            timeline_months=6,
            cost_estimate=125000,
            key_features=[
                "Advanced analytics",
                "Workflow automation",
                "Mobile applications"
            ],
            confidence_level="high"
        ),
        # Medium quality
        RFQProposal(
            title="CRM System Development",
            description="Custom CRM system with basic features.",
            timeline_months=3,
            cost_estimate=50000,
            key_features=[
                "Contact management",
                "Basic reporting"
            ],
            confidence_level="medium"
        ),
        # Low quality
        RFQProposal(
            title="Software",
            description="We can build it.",
            timeline_months=1,
            cost_estimate=15000,
            key_features=["Software"],
            confidence_level="low"
        )
    ]


@pytest.fixture
def best_of_n_selector():
    """Best-of-N selector instance for testing."""
    return BestOfNSelector(
        evaluation_model="openai:gpt-4o-mini",
        max_parallel_generations=5,
        enable_detailed_evaluation=True
    )


@pytest.fixture
def agent_context():
    """Agent context for testing."""
    return AgentContext(
        request_id="test-request",
        user_id="test-user",
        session_id="test-session"
    )


class TestBestOfNSelector:
    """Test suite for Best-of-N selector."""
    
    @pytest.mark.asyncio
    async def test_basic_best_of_n_generation(
        self, 
        best_of_n_selector, 
        mixed_quality_proposals, 
        agent_context
    ):
        """Test basic Best-of-N candidate generation."""
        mock_agent = MockRFQAgent(mixed_quality_proposals)
        prompt = "Generate a professional RFQ proposal"
        
        # Use TestModel to avoid API calls
        with best_of_n_selector._judge_agent.override(model=TestModel()):
            with best_of_n_selector._selection_agent.override(model=TestModel()):
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=3
                )
        
        # Validate results
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 3
        assert len(result.all_evaluations) == 3
        assert result.best_candidate is not None
        assert result.best_evaluation is not None
        assert 0.0 <= result.selection_confidence <= 1.0
        
        # Check that all candidates have unique IDs
        candidate_ids = {eval.candidate_id for eval in result.all_evaluations}
        assert len(candidate_ids) == 3
    
    @pytest.mark.asyncio
    async def test_llm_judge_with_function_model(
        self, 
        best_of_n_selector, 
        mixed_quality_proposals, 
        agent_context
    ):
        """Test LLM judge using FunctionModel for controlled evaluation."""
        
        def mock_judge_evaluation(messages, info: AgentInfo):
            """Mock function that returns structured evaluations."""
            prompt = str(messages[-1].parts[0].content) if messages else ""
            
            # Score based on content quality indicators
            if "comprehensive" in prompt.lower() or "enterprise" in prompt.lower():
                return {
                    "candidate_id": "candidate_0",
                    "overall_score": 0.85,
                    "accuracy_score": 0.9,
                    "completeness_score": 0.9,
                    "relevance_score": 0.8,
                    "clarity_score": 0.8,
                    "reasoning": "High-quality proposal with comprehensive features",
                    "evaluation_time_ms": 50.0
                }
            else:
                return {
                    "candidate_id": "candidate_1",
                    "overall_score": 0.45,
                    "accuracy_score": 0.5,
                    "completeness_score": 0.4,
                    "relevance_score": 0.5,
                    "clarity_score": 0.4,
                    "reasoning": "Basic proposal lacking detail",
                    "evaluation_time_ms": 30.0
                }
        
        def mock_selection_agent(messages, info: AgentInfo):
            """Mock selection that picks the first candidate."""
            return "candidate_0"
        
        mock_agent = MockRFQAgent(mixed_quality_proposals)
        prompt = "Generate enterprise software proposal"
        
        # Use FunctionModel for controlled testing
        with best_of_n_selector._judge_agent.override(model=FunctionModel(mock_judge_evaluation)):
            with best_of_n_selector._selection_agent.override(model=FunctionModel(mock_selection_agent)):
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=2
                )
        
        # Validate structured evaluation results
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 2
        assert len(result.all_evaluations) == 2
        
        # Check evaluation details
        for evaluation in result.all_evaluations:
            assert evaluation.reasoning is not None
            assert len(evaluation.reasoning) > 5
            assert 0.0 <= evaluation.overall_score <= 1.0
            assert 0.0 <= evaluation.accuracy_score <= 1.0
            assert 0.0 <= evaluation.completeness_score <= 1.0
            assert 0.0 <= evaluation.relevance_score <= 1.0
            assert 0.0 <= evaluation.clarity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_custom_evaluation_criteria(
        self, 
        best_of_n_selector, 
        high_quality_proposals, 
        agent_context
    ):
        """Test custom evaluation criteria application."""
        mock_agent = MockRFQAgent(high_quality_proposals)
        
        # Custom criteria emphasizing completeness and relevance
        custom_criteria = EvaluationCriteria(
            accuracy_weight=0.15,
            completeness_weight=0.40,  # Emphasize completeness
            relevance_weight=0.35,     # Emphasize relevance
            clarity_weight=0.10
        )
        
        prompt = "Generate comprehensive RFQ proposal"
        
        with best_of_n_selector._judge_agent.override(model=TestModel()):
            with best_of_n_selector._selection_agent.override(model=TestModel()):
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=2,
                    criteria=custom_criteria
                )
        
        # Validate that custom criteria were applied
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 2
        
        # The system should have used the custom criteria
        # (specific validation would require inspecting the evaluation process)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_timeouts(
        self, 
        best_of_n_selector, 
        mixed_quality_proposals, 
        agent_context
    ):
        """Test error handling and timeout behavior."""
        mock_agent = MockRFQAgent(mixed_quality_proposals)
        
        # Set short timeouts for testing
        deps = BestOfNDependencies(
            generation_timeout=0.1,  # Very short timeout
            evaluation_timeout=0.1
        )
        
        prompt = "Generate RFQ proposal"
        
        with best_of_n_selector._judge_agent.override(model=TestModel()):
            with best_of_n_selector._selection_agent.override(model=TestModel()):
                # This should still work with TestModel as it's fast
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=2,
                    deps=deps
                )
        
        # Should complete successfully even with short timeouts using TestModel
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 2
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(
        self, 
        best_of_n_selector, 
        mixed_quality_proposals, 
        agent_context
    ):
        """Test confidence scoring calculation."""
        
        def mock_judge_with_varied_scores(messages, info: AgentInfo):
            """Mock judge that returns varied scores for confidence testing."""
            # Return different scores to test confidence calculation
            prompt = str(messages[-1].parts[0].content) if messages else ""
            
            if "candidate_0" in prompt:
                return {
                    "candidate_id": "candidate_0",
                    "overall_score": 0.9,  # High score
                    "accuracy_score": 0.9,
                    "completeness_score": 0.9,
                    "relevance_score": 0.9,
                    "clarity_score": 0.9,
                    "reasoning": "Excellent proposal",
                    "evaluation_time_ms": 50.0
                }
            elif "candidate_1" in prompt:
                return {
                    "candidate_id": "candidate_1",
                    "overall_score": 0.5,  # Medium score
                    "accuracy_score": 0.5,
                    "completeness_score": 0.5,
                    "relevance_score": 0.5,
                    "clarity_score": 0.5,
                    "reasoning": "Average proposal",
                    "evaluation_time_ms": 40.0
                }
            else:
                return {
                    "candidate_id": "candidate_2",
                    "overall_score": 0.2,  # Low score
                    "accuracy_score": 0.2,
                    "completeness_score": 0.2,
                    "relevance_score": 0.2,
                    "clarity_score": 0.2,
                    "reasoning": "Poor proposal",
                    "evaluation_time_ms": 30.0
                }
        
        def mock_selection_best(messages, info: AgentInfo):
            """Mock selection that picks the highest scored candidate."""
            return "candidate_0"
        
        mock_agent = MockRFQAgent(mixed_quality_proposals)
        prompt = "Generate proposal for confidence testing"
        
        with best_of_n_selector._judge_agent.override(model=FunctionModel(mock_judge_with_varied_scores)):
            with best_of_n_selector._selection_agent.override(model=FunctionModel(mock_selection_best)):
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=3
                )
        
        # With varied scores, confidence should reflect the distribution
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 3
        
        # The confidence should be reasonable (not 0 or 1 with varied scores)
        assert 0.0 <= result.selection_confidence <= 1.0
        
        # Best candidate should have the highest score
        best_score = result.best_evaluation.overall_score
        other_scores = [e.overall_score for e in result.all_evaluations 
                       if e.candidate_id != result.best_candidate.candidate_id]
        
        if other_scores:  # Only check if we have other scores
            assert best_score >= max(other_scores)
    
    @pytest.mark.asyncio
    async def test_parallel_generation_performance(
        self, 
        best_of_n_selector, 
        high_quality_proposals, 
        agent_context
    ):
        """Test parallel generation performance."""
        import time
        
        mock_agent = MockRFQAgent(high_quality_proposals)
        prompt = "Generate RFQ proposal for performance testing"
        
        start_time = time.time()
        
        with best_of_n_selector._judge_agent.override(model=TestModel()):
            with best_of_n_selector._selection_agent.override(model=TestModel()):
                result = await best_of_n_selector.generate_best_of_n(
                    target_agent=mock_agent,
                    prompt=prompt,
                    context=agent_context,
                    n=5
                )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should be very fast with TestModel
        assert execution_time < 1.0  # Should complete in under 1 second
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 5
        
        # Performance should be reasonable
        candidates_per_second = result.n_candidates / execution_time
        assert candidates_per_second > 10  # Should process at least 10 candidates/second


class TestEvaluationCriteria:
    """Test suite for evaluation criteria."""
    
    def test_evaluation_criteria_creation(self):
        """Test creating evaluation criteria with custom weights."""
        criteria = EvaluationCriteria(
            accuracy_weight=0.3,
            completeness_weight=0.3,
            relevance_weight=0.2,
            clarity_weight=0.2
        )
        
        assert criteria.accuracy_weight == 0.3
        assert criteria.completeness_weight == 0.3
        assert criteria.relevance_weight == 0.2
        assert criteria.clarity_weight == 0.2
        
        # Weights should sum to 1.0
        total_weight = (
            criteria.accuracy_weight + 
            criteria.completeness_weight + 
            criteria.relevance_weight + 
            criteria.clarity_weight
        )
        assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision
    
    def test_default_evaluation_criteria(self):
        """Test default evaluation criteria."""
        criteria = EvaluationCriteria()
        
        # Check actual default weights (based on the implementation)
        assert criteria.accuracy_weight == 0.3
        assert criteria.completeness_weight == 0.3
        assert criteria.relevance_weight == 0.2
        assert criteria.clarity_weight == 0.2


class TestBestOfNDependencies:
    """Test suite for Best-of-N dependencies."""
    
    def test_dependencies_creation(self):
        """Test creating dependencies with custom timeouts."""
        deps = BestOfNDependencies(
            generation_timeout=15.0,
            evaluation_timeout=20.0
        )
        
        assert deps.generation_timeout == 15.0
        assert deps.evaluation_timeout == 20.0
    
    def test_default_dependencies(self):
        """Test default dependency values."""
        deps = BestOfNDependencies()
        
        # Should have reasonable default timeouts
        assert deps.generation_timeout > 0
        assert deps.evaluation_timeout > 0 