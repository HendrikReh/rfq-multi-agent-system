"""
Best-of-N Selection Agent

This module implements Best-of-N selection patterns following PydanticAI best practices,
including agent delegation and LLM judge evaluation for selecting the best output
from multiple candidate generations.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import Usage, UsageLimits

from ...core.interfaces.agent import BaseAgent, AgentContext
from ...core.models.rfq import RFQProcessingResult


T = TypeVar('T')
R = TypeVar('R')


class CandidateOutput(BaseModel):
    """Represents a single candidate output for evaluation."""
    candidate_id: str
    output: Any
    generation_time_ms: float
    model_used: str
    confidence_score: Optional[float] = None


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating candidate outputs."""
    accuracy_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    completeness_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    clarity_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    def validate_weights(self) -> bool:
        """Ensure weights sum to 1.0."""
        total = self.accuracy_weight + self.completeness_weight + self.relevance_weight + self.clarity_weight
        return abs(total - 1.0) < 0.001


class EvaluationResult(BaseModel):
    """Result of evaluating a candidate output."""
    candidate_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evaluation_time_ms: float


class BestOfNResult(BaseModel):
    """Result of Best-of-N selection process."""
    best_candidate: CandidateOutput
    best_evaluation: EvaluationResult
    all_candidates: List[CandidateOutput]
    all_evaluations: List[EvaluationResult]
    total_generation_time_ms: float
    total_evaluation_time_ms: float
    n_candidates: int
    selection_confidence: float = Field(ge=0.0, le=1.0)


class GenerationFailed(BaseModel):
    """Represents a failed generation attempt."""
    error_message: str
    candidate_id: str


@dataclass
class BestOfNDependencies:
    """Dependencies for Best-of-N selection."""
    usage_limits: Optional[UsageLimits] = None
    evaluation_model: str = "openai:gpt-4o"
    generation_timeout: float = 30.0
    evaluation_timeout: float = 15.0


class BestOfNSelector:
    """
    Best-of-N selector implementing PydanticAI agent delegation patterns.
    
    This class generates N candidate outputs using a target agent, then uses
    an LLM judge to evaluate and select the best candidate based on specified criteria.
    """
    
    def __init__(
        self,
        evaluation_model: str = "openai:gpt-4o",
        max_parallel_generations: int = 5,
        enable_detailed_evaluation: bool = True
    ):
        self.evaluation_model = evaluation_model
        self.max_parallel_generations = max_parallel_generations
        self.enable_detailed_evaluation = enable_detailed_evaluation
        
        # Create the LLM judge agent for evaluation
        self._judge_agent = Agent(
            model=evaluation_model,
            result_type=EvaluationResult,
            system_prompt=self._get_judge_system_prompt()
        )
        
        # Create agent for selecting best candidate
        self._selection_agent = Agent(
            model=evaluation_model,
            result_type=str,  # Returns candidate_id of best choice
            system_prompt=self._get_selection_system_prompt()
        )
    
    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for the LLM judge agent."""
        return """You are an expert evaluator tasked with scoring outputs based on multiple criteria.

Your job is to evaluate a candidate output against the original prompt and provide detailed scoring.

Evaluation Criteria:
- Accuracy: How factually correct and precise is the output?
- Completeness: Does the output fully address all aspects of the prompt?
- Relevance: How well does the output match the intent and context?
- Clarity: Is the output clear, well-structured, and easy to understand?

For each criterion, provide a score from 0.0 to 1.0, and calculate an overall weighted score.
Always provide clear reasoning for your evaluation."""
    
    def _get_selection_system_prompt(self) -> str:
        """Get system prompt for the selection agent."""
        return """You are tasked with selecting the best candidate from multiple evaluated options.

You will receive:
1. The original prompt/request
2. Multiple candidates with their evaluation scores and reasoning
3. Evaluation criteria and weights

Your job is to select the candidate_id of the best overall option, considering:
- Evaluation scores across all criteria
- Quality of reasoning in evaluations
- Overall fit for the original request

Return only the candidate_id of your chosen best option."""
    
    async def generate_best_of_n(
        self,
        target_agent: BaseAgent,
        prompt: str,
        context: AgentContext,
        n: int = 3,
        criteria: Optional[EvaluationCriteria] = None,
        deps: Optional[BestOfNDependencies] = None
    ) -> BestOfNResult:
        """
        Generate N candidates and select the best using LLM judge evaluation.
        
        Args:
            target_agent: The agent to generate candidates with
            prompt: The input prompt for generation
            context: Agent context for generation
            n: Number of candidates to generate
            criteria: Evaluation criteria (uses defaults if None)
            deps: Dependencies for generation and evaluation
            
        Returns:
            BestOfNResult with the best candidate and evaluation details
        """
        if criteria is None:
            criteria = EvaluationCriteria()
        
        if deps is None:
            deps = BestOfNDependencies()
        
        if not criteria.validate_weights():
            raise ValueError("Evaluation criteria weights must sum to 1.0")
        
        start_time = time.time()
        
        # Step 1: Generate N candidates in parallel
        candidates = await self._generate_candidates(
            target_agent, prompt, context, n, deps
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        if not candidates:
            raise RuntimeError("Failed to generate any valid candidates")
        
        # Step 2: Evaluate all candidates using LLM judge
        eval_start_time = time.time()
        evaluations = await self._evaluate_candidates(
            candidates, prompt, criteria, deps
        )
        evaluation_time = (time.time() - eval_start_time) * 1000
        
        # Step 3: Select the best candidate
        best_candidate, best_evaluation = await self._select_best_candidate(
            candidates, evaluations, prompt, criteria
        )
        
        # Step 4: Calculate selection confidence
        selection_confidence = self._calculate_selection_confidence(evaluations)
        
        return BestOfNResult(
            best_candidate=best_candidate,
            best_evaluation=best_evaluation,
            all_candidates=candidates,
            all_evaluations=evaluations,
            total_generation_time_ms=generation_time,
            total_evaluation_time_ms=evaluation_time,
            n_candidates=len(candidates),
            selection_confidence=selection_confidence
        )
    
    async def _generate_candidates(
        self,
        target_agent: BaseAgent,
        prompt: str,
        context: AgentContext,
        n: int,
        deps: BestOfNDependencies
    ) -> List[CandidateOutput]:
        """Generate N candidate outputs using the target agent."""
        # Create semaphore to limit parallel generations
        semaphore = asyncio.Semaphore(min(n, self.max_parallel_generations))
        
        async def generate_single_candidate(candidate_id: str) -> Union[CandidateOutput, GenerationFailed]:
            async with semaphore:
                start_time = time.time()
                try:
                    # Use asyncio.wait_for for timeout protection
                    result = await asyncio.wait_for(
                        target_agent.process(prompt, context),
                        timeout=deps.generation_timeout
                    )
                    
                    generation_time = (time.time() - start_time) * 1000
                    
                    return CandidateOutput(
                        candidate_id=candidate_id,
                        output=result,
                        generation_time_ms=generation_time,
                        model_used=getattr(target_agent, 'model', 'unknown')
                    )
                    
                except asyncio.TimeoutError:
                    return GenerationFailed(
                        error_message=f"Generation timeout after {deps.generation_timeout}s",
                        candidate_id=candidate_id
                    )
                except Exception as e:
                    return GenerationFailed(
                        error_message=str(e),
                        candidate_id=candidate_id
                    )
        
        # Generate all candidates in parallel
        tasks = [
            generate_single_candidate(f"candidate_{i}")
            for i in range(n)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful candidates
        candidates = []
        for result in results:
            if isinstance(result, CandidateOutput):
                candidates.append(result)
            elif isinstance(result, Exception):
                print(f"Generation failed with exception: {result}")
        
        return candidates
    
    async def _evaluate_candidates(
        self,
        candidates: List[CandidateOutput],
        original_prompt: str,
        criteria: EvaluationCriteria,
        deps: BestOfNDependencies
    ) -> List[EvaluationResult]:
        """Evaluate all candidates using the LLM judge."""
        async def evaluate_single_candidate(candidate: CandidateOutput) -> EvaluationResult:
            start_time = time.time()
            
            evaluation_prompt = f"""
Original Prompt: {original_prompt}

Candidate Output: {candidate.output}

Evaluation Criteria Weights:
- Accuracy: {criteria.accuracy_weight}
- Completeness: {criteria.completeness_weight}
- Relevance: {criteria.relevance_weight}
- Clarity: {criteria.clarity_weight}

Please evaluate this candidate output against the original prompt using the specified criteria.
Provide scores from 0.0 to 1.0 for each criterion and calculate the overall weighted score.
"""
            
            try:
                result = await asyncio.wait_for(
                    self._judge_agent.run(evaluation_prompt, usage_limits=deps.usage_limits),
                    timeout=deps.evaluation_timeout
                )
                
                evaluation_time = (time.time() - start_time) * 1000
                evaluation = result.data
                evaluation.candidate_id = candidate.candidate_id
                evaluation.evaluation_time_ms = evaluation_time
                
                # Calculate overall score using weights
                evaluation.overall_score = (
                    evaluation.accuracy_score * criteria.accuracy_weight +
                    evaluation.completeness_score * criteria.completeness_weight +
                    evaluation.relevance_score * criteria.relevance_weight +
                    evaluation.clarity_score * criteria.clarity_weight
                )
                
                return evaluation
                
            except Exception as e:
                # Return default evaluation on failure
                evaluation_time = (time.time() - start_time) * 1000
                return EvaluationResult(
                    candidate_id=candidate.candidate_id,
                    overall_score=0.0,
                    accuracy_score=0.0,
                    completeness_score=0.0,
                    relevance_score=0.0,
                    clarity_score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}",
                    evaluation_time_ms=evaluation_time
                )
        
        # Evaluate all candidates in parallel
        evaluations = await asyncio.gather(*[
            evaluate_single_candidate(candidate) for candidate in candidates
        ])
        
        return evaluations
    
    async def _select_best_candidate(
        self,
        candidates: List[CandidateOutput],
        evaluations: List[EvaluationResult],
        original_prompt: str,
        criteria: EvaluationCriteria
    ) -> tuple[CandidateOutput, EvaluationResult]:
        """Select the best candidate using the selection agent."""
        if self.enable_detailed_evaluation:
            # Use LLM to make final selection
            selection_prompt = f"""
Original Prompt: {original_prompt}

Evaluation Criteria Weights:
- Accuracy: {criteria.accuracy_weight}
- Completeness: {criteria.completeness_weight}  
- Relevance: {criteria.relevance_weight}
- Clarity: {criteria.clarity_weight}

Candidate Evaluations:
"""
            
            for i, (candidate, evaluation) in enumerate(zip(candidates, evaluations)):
                selection_prompt += f"""
Candidate {evaluation.candidate_id}:
- Overall Score: {evaluation.overall_score:.3f}
- Accuracy: {evaluation.accuracy_score:.3f}
- Completeness: {evaluation.completeness_score:.3f}
- Relevance: {evaluation.relevance_score:.3f}
- Clarity: {evaluation.clarity_score:.3f}
- Reasoning: {evaluation.reasoning}
- Output: {str(candidate.output)[:200]}...

"""
            
            selection_prompt += "\nSelect the candidate_id of the best overall option."
            
            try:
                result = await self._selection_agent.run(selection_prompt)
                selected_id = result.data.strip()
                
                # Find the selected candidate and evaluation
                for candidate, evaluation in zip(candidates, evaluations):
                    if evaluation.candidate_id == selected_id:
                        return candidate, evaluation
                        
            except Exception as e:
                print(f"Selection agent failed: {e}, falling back to highest score")
        
        # Fallback: select candidate with highest overall score
        best_evaluation = max(evaluations, key=lambda e: e.overall_score)
        best_candidate = next(
            c for c in candidates if c.candidate_id == best_evaluation.candidate_id
        )
        
        return best_candidate, best_evaluation
    
    def _calculate_selection_confidence(self, evaluations: List[EvaluationResult]) -> float:
        """Calculate confidence in the selection based on score distribution."""
        if len(evaluations) <= 1:
            return 1.0
        
        scores = [e.overall_score for e in evaluations]
        scores.sort(reverse=True)
        
        # Confidence based on gap between best and second-best scores
        if len(scores) >= 2:
            score_gap = scores[0] - scores[1]
            # Normalize gap to confidence (larger gap = higher confidence)
            confidence = min(1.0, 0.5 + score_gap)
        else:
            confidence = scores[0]  # Single score case
        
        return confidence


# Tool function for agent delegation pattern
@dataclass
class BestOfNToolDeps:
    """Dependencies for the best-of-N tool."""
    selector: BestOfNSelector
    target_agent: BaseAgent
    context: AgentContext


# Tool function for agent delegation pattern - defined at module level
async def best_of_n_generation_tool(
    ctx: RunContext[BestOfNToolDeps],
    prompt: str,
    n: int = 3,
    accuracy_weight: float = 0.3,
    completeness_weight: float = 0.3,
    relevance_weight: float = 0.2,
    clarity_weight: float = 0.2
) -> BestOfNResult:
    """Generate N candidates and select the best using LLM judge evaluation."""
    criteria = EvaluationCriteria(
        accuracy_weight=accuracy_weight,
        completeness_weight=completeness_weight,
        relevance_weight=relevance_weight,
        clarity_weight=clarity_weight
    )
    
    result = await ctx.deps.selector.generate_best_of_n(
        target_agent=ctx.deps.target_agent,
        prompt=prompt,
        context=ctx.deps.context,
        n=n,
        criteria=criteria
    )
    
    return result


class BestOfNAgent:
    """Wrapper class for the Best-of-N agent."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        self.agent = Agent(
            model=model,
            deps_type=BestOfNToolDeps,
            system_prompt="""You are an agent that can use Best-of-N selection to generate high-quality outputs.

When asked to generate content, use the best_of_n_generation tool to create multiple candidates and select the best one.

You can specify:
- Number of candidates (n)
- Evaluation criteria weights
- Generation and evaluation timeouts

Always use this tool for important or complex generation tasks where quality matters."""
        )
        
        # Add the tool to the agent
        self.agent.tool(best_of_n_generation_tool, name="best_of_n_generation")
    
    async def run(self, prompt: str, deps: BestOfNToolDeps):
        """Run the agent with the given prompt and dependencies."""
        return await self.agent.run(prompt, deps=deps)
    
    def override(self, **kwargs):
        """Override agent configuration."""
        return self.agent.override(**kwargs) 