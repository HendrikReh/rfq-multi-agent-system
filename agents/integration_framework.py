"""
Integration Framework for Complete RFQ Multi-Agent System

This framework integrates all agents into a cohesive system with:
- Comprehensive workflow orchestration
- Performance monitoring and evaluation
- Agent health checks and optimization
- Real-time analytics and reporting
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .competitive_intelligence_agent import CompetitiveIntelligenceAgent, CompetitiveAnalysis
from .contract_terms_agent import ContractTermsAgent, ContractTerms
from .customer_intent_agent import CustomerIntentAgent
from .customer_response_agent import CustomerResponseAgent
from .enhanced_orchestrator import EnhancedRFQOrchestrator, GraphBasedRFQController, MemoryEnhancedAgent
from .evaluation_intelligence_agent import EvaluationIntelligenceAgent
from .interaction_decision_agent import InteractionDecisionAgent
from .models import (
    CustomerIntent,
    RFQDependencies,
    RFQProcessingResult,
    RFQRequirements,
    SystemPerformance,
)
from .pricing_strategy_agent import PricingStrategyAgent
from .proposal_writer_agent import ProposalWriterAgent, ProposalDocument
from .question_generation_agent import QuestionGenerationAgent
from .rfq_parser import RFQParser
from .risk_assessment_agent import RiskAssessmentAgent, RiskAssessment
from .utils import get_model_name


class AgentHealthStatus(BaseModel):
    """Health status for individual agents."""
    agent_name: str
    status: str = Field(description="HEALTHY, DEGRADED, or FAILED")
    response_time: float
    error_count: int = 0
    last_successful_run: Optional[datetime] = None
    performance_score: float = Field(ge=0.0, le=1.0)


class SystemHealthReport(BaseModel):
    """Overall system health report."""
    overall_status: str
    agent_statuses: List[AgentHealthStatus]
    total_agents: int
    healthy_agents: int
    degraded_agents: int
    failed_agents: int
    average_response_time: float
    system_uptime: float
    recommendations: List[str] = Field(default_factory=list)


class ComprehensiveRFQResult(BaseModel):
    """Enhanced RFQ processing result with all agent outputs."""
    basic_result: RFQProcessingResult
    competitive_analysis: Optional[CompetitiveAnalysis] = None
    risk_assessment: Optional[RiskAssessment] = None
    contract_terms: Optional[ContractTerms] = None
    proposal_document: Optional[ProposalDocument] = None
    processing_metrics: Dict[str, float] = Field(default_factory=dict)
    agent_contributions: Dict[str, str] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    recommendation: str


class IntegratedRFQSystem:
    """
    Comprehensive RFQ processing system integrating all agents.
    
    Features:
    - Complete workflow orchestration
    - Parallel agent execution where appropriate
    - Performance monitoring and optimization
    - Health checks and error recovery
    - Advanced analytics and reporting
    """
    
    def __init__(self):
        # Agent health tracking - initialize first
        self.agent_health = {}
        self.system_start_time = datetime.now()
        
        # Initialize all agents
        self._init_all_agents()
        
        # Initialize monitoring and evaluation
        self._init_monitoring_system()
    
    def _init_all_agents(self):
        """Initialize all RFQ processing agents."""
        # Core agents
        self.rfq_parser = RFQParser()
        self.intent_agent = CustomerIntentAgent()
        self.decision_agent = InteractionDecisionAgent()
        self.question_agent = QuestionGenerationAgent()
        self.pricing_agent = PricingStrategyAgent()
        self.evaluation_agent = EvaluationIntelligenceAgent()
        self.customer_response_agent = CustomerResponseAgent()
        
        # Enhanced orchestrators
        self.enhanced_orchestrator = EnhancedRFQOrchestrator()
        self.graph_controller = GraphBasedRFQController()
        self.memory_agent = MemoryEnhancedAgent()
        
        # Specialized agents
        self.competitive_agent = CompetitiveIntelligenceAgent()
        self.contract_agent = ContractTermsAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.proposal_agent = ProposalWriterAgent()
        
        # Meta-analysis agent for system optimization
        self.meta_agent = Agent(
            get_model_name("rfq_orchestrator"),
            output_type=str,
            system_prompt="""
            You are a meta-analysis agent that optimizes multi-agent system performance.
            
            Analyze agent outputs, identify optimization opportunities, and provide
            recommendations for improving system effectiveness, efficiency, and accuracy.
            
            Consider:
            - Agent coordination effectiveness
            - Response quality and consistency
            - Processing time optimization
            - Error patterns and prevention
            - Customer satisfaction indicators
            """
        )
    
    def _init_monitoring_system(self):
        """Initialize performance monitoring and analytics."""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "agent_utilization": {},
            "error_patterns": {}
        }
        
        # Initialize health status for all agents
        agent_names = [
            "enhanced_orchestrator",
            "competitive_intelligence",
            "risk_assessment", 
            "contract_terms",
            "proposal_writer"
        ]
        
        for agent_name in agent_names:
            self.agent_health[agent_name] = AgentHealthStatus(
                agent_name=agent_name,
                status="HEALTHY",
                response_time=0.0,
                performance_score=1.0
            )
    
    async def process_rfq_comprehensive(
        self,
        customer_message: str,
        deps: RFQDependencies,
        include_competitive_analysis: bool = True,
        include_risk_assessment: bool = True,
        include_contract_terms: bool = True,
        include_proposal: bool = True,
        execution_mode: str = "parallel"  # "parallel" or "sequential"
    ) -> ComprehensiveRFQResult:
        """
        Process RFQ using all available agents for comprehensive analysis.
        
        Args:
            customer_message: Customer's RFQ message
            deps: RFQ processing dependencies
            include_competitive_analysis: Whether to include competitive analysis
            include_risk_assessment: Whether to include risk assessment
            include_contract_terms: Whether to include contract terms
            include_proposal: Whether to include proposal generation
            execution_mode: "parallel" for speed or "sequential" for dependency management
            
        Returns:
            ComprehensiveRFQResult: Complete analysis from all agents
        """
        start_time = datetime.now()
        self.performance_metrics["total_requests"] += 1
        
        try:
            # Step 1: Core RFQ processing (always required)
            basic_result = await self.enhanced_orchestrator.process_rfq_enhanced(
                customer_message, deps
            )
            
            # Step 2: Extract core data for specialized agents
            requirements = basic_result.requirements
            customer_intent = basic_result.customer_intent
            
            # Step 3: Execute specialized agents based on configuration
            specialized_tasks = []
            
            if execution_mode == "parallel":
                # Execute all specialized agents in parallel
                if include_competitive_analysis:
                    specialized_tasks.append(
                        self._execute_with_health_check(
                            "competitive_analysis",
                            self.competitive_agent.analyze_competitive_landscape(
                                requirements, customer_intent
                            )
                        )
                    )
                
                if include_risk_assessment:
                    specialized_tasks.append(
                        self._execute_with_health_check(
                            "risk_assessment",
                            self.risk_agent.assess_risks(requirements, customer_intent)
                        )
                    )
                
                if include_contract_terms:
                    specialized_tasks.append(
                        self._execute_with_health_check(
                            "contract_terms",
                            self.contract_agent.develop_contract_terms(
                                requirements, customer_intent
                            )
                        )
                    )
                
                # Execute parallel tasks
                specialized_results = await asyncio.gather(*specialized_tasks, return_exceptions=True)
                
                # Process results
                competitive_analysis = None
                risk_assessment = None
                contract_terms = None
                
                for i, result in enumerate(specialized_results):
                    if not isinstance(result, Exception):
                        if i == 0 and include_competitive_analysis:
                            competitive_analysis = result
                        elif i == 1 and include_risk_assessment:
                            risk_assessment = result
                        elif i == 2 and include_contract_terms:
                            contract_terms = result
                
                # Generate proposal if requested and quote is available
                proposal_document = None
                if include_proposal and basic_result.quote:
                    proposal_document = await self._execute_with_health_check(
                        "proposal_generation",
                        self.proposal_agent.generate_proposal(
                            requirements, customer_intent, basic_result.quote
                        )
                    )
            
            else:  # Sequential execution
                # Execute agents sequentially with dependency management
                competitive_analysis = None
                risk_assessment = None
                contract_terms = None
                proposal_document = None
                
                if include_competitive_analysis:
                    competitive_analysis = await self.competitive_agent.analyze_competitive_landscape(
                        requirements, customer_intent
                    )
                
                if include_risk_assessment:
                    risk_assessment = await self.risk_agent.assess_risks(
                        requirements, customer_intent
                    )
                
                if include_contract_terms:
                    contract_terms = await self.contract_agent.develop_contract_terms(
                        requirements, customer_intent
                    )
                
                if include_proposal and basic_result.quote:
                    # Use previous analyses to enhance proposal
                    competitive_context = ""
                    if competitive_analysis:
                        competitive_context = f"Competitive Analysis: {competitive_analysis.recommended_strategy}"
                    
                    proposal_document = await self.proposal_agent.generate_proposal(
                        requirements, customer_intent, basic_result.quote,
                        competitive_context=competitive_context
                    )
            
            # Step 4: Calculate comprehensive metrics and confidence
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence score based on available information
            confidence_score = self._calculate_confidence_score(
                basic_result, competitive_analysis, risk_assessment, contract_terms
            )
            
            # Generate system recommendation
            recommendation = await self._generate_system_recommendation(
                basic_result, competitive_analysis, risk_assessment, contract_terms
            )
            
            # Step 5: Create comprehensive result
            comprehensive_result = ComprehensiveRFQResult(
                basic_result=basic_result,
                competitive_analysis=competitive_analysis,
                risk_assessment=risk_assessment,
                contract_terms=contract_terms,
                proposal_document=proposal_document,
                processing_metrics={
                    "total_processing_time": processing_time,
                    "agents_executed": sum([
                        1,  # basic processing
                        1 if competitive_analysis else 0,
                        1 if risk_assessment else 0,
                        1 if contract_terms else 0,
                        1 if proposal_document else 0
                    ]),
                    "execution_mode": execution_mode
                },
                agent_contributions={
                    "core_processing": "Enhanced RFQ Orchestrator",
                    "competitive_intelligence": "Competitive Intelligence Agent" if competitive_analysis else "Not executed",
                    "risk_assessment": "Risk Assessment Agent" if risk_assessment else "Not executed",
                    "contract_terms": "Contract Terms Agent" if contract_terms else "Not executed",
                    "proposal_generation": "Proposal Writer Agent" if proposal_document else "Not executed"
                },
                confidence_score=confidence_score,
                recommendation=recommendation
            )
            
            # Update metrics
            self.performance_metrics["successful_requests"] += 1
            self._update_performance_metrics(processing_time)
            
            return comprehensive_result
            
        except Exception as e:
            self.performance_metrics["failed_requests"] += 1
            raise e
    
    async def _execute_with_health_check(self, agent_name: str, coroutine):
        """Execute agent with health monitoring."""
        start_time = datetime.now()
        
        try:
            result = await coroutine
            
            # Update health status
            response_time = (datetime.now() - start_time).total_seconds()
            self.agent_health[agent_name] = AgentHealthStatus(
                agent_name=agent_name,
                status="HEALTHY",
                response_time=response_time,
                last_successful_run=datetime.now(),
                performance_score=min(1.0, 1.0 / max(0.1, response_time))  # Simple performance score
            )
            
            return result
            
        except Exception as e:
            # Update health status for failure
            if agent_name in self.agent_health:
                self.agent_health[agent_name].error_count += 1
                self.agent_health[agent_name].status = "FAILED"
            
            raise e
    
    def _calculate_confidence_score(
        self,
        basic_result: RFQProcessingResult,
        competitive_analysis: Optional[CompetitiveAnalysis],
        risk_assessment: Optional[RiskAssessment],
        contract_terms: Optional[ContractTerms]
    ) -> float:
        """Calculate overall confidence score based on available analyses."""
        
        base_confidence = basic_result.interaction_decision.confidence_level / 5.0
        
        # Boost confidence based on additional analyses
        confidence_boost = 0.0
        
        if competitive_analysis:
            confidence_boost += competitive_analysis.win_probability * 0.2
        
        if risk_assessment:
            # Lower risk increases confidence
            risk_factor = (10.0 - risk_assessment.risk_score) / 10.0
            confidence_boost += risk_factor * 0.15
        
        if contract_terms:
            confidence_boost += 0.1  # Having contract terms adds confidence
        
        return min(1.0, base_confidence + confidence_boost)
    
    async def _generate_system_recommendation(
        self,
        basic_result: RFQProcessingResult,
        competitive_analysis: Optional[CompetitiveAnalysis],
        risk_assessment: Optional[RiskAssessment],
        contract_terms: Optional[ContractTerms]
    ) -> str:
        """Generate comprehensive system recommendation."""
        
        context = f"""
        Basic RFQ Result: {basic_result.interaction_decision.reasoning}
        
        Competitive Analysis: {competitive_analysis.recommended_strategy if competitive_analysis else 'Not available'}
        
        Risk Assessment: {risk_assessment.recommendation if risk_assessment else 'Not available'}
        
        Contract Considerations: {'Available' if contract_terms else 'Not available'}
        
        Provide a comprehensive recommendation for how to proceed with this RFQ opportunity.
        """
        
        result = await self.meta_agent.run(context)
        return result.output
    
    def _update_performance_metrics(self, processing_time: float):
        """Update system performance metrics."""
        # Update average response time
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics["average_response_time"] = new_avg
    
    async def get_system_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        
        agent_statuses = list(self.agent_health.values())
        
        healthy_count = sum(1 for status in agent_statuses if status.status == "HEALTHY")
        degraded_count = sum(1 for status in agent_statuses if status.status == "DEGRADED")
        failed_count = sum(1 for status in agent_statuses if status.status == "FAILED")
        
        overall_status = "HEALTHY"
        if failed_count > 0:
            overall_status = "DEGRADED" if failed_count < len(agent_statuses) / 2 else "FAILED"
        elif degraded_count > 0:
            overall_status = "DEGRADED"
        
        avg_response_time = sum(status.response_time for status in agent_statuses) / max(1, len(agent_statuses))
        
        uptime_hours = (datetime.now() - self.system_start_time).total_seconds() / 3600
        
        recommendations = []
        if failed_count > 0:
            recommendations.append(f"Investigate {failed_count} failed agents")
        if avg_response_time > 5.0:
            recommendations.append("Consider optimizing slow agents")
        if degraded_count > 0:
            recommendations.append("Monitor degraded agents closely")
        
        return SystemHealthReport(
            overall_status=overall_status,
            agent_statuses=agent_statuses,
            total_agents=len(agent_statuses),
            healthy_agents=healthy_count,
            degraded_agents=degraded_count,
            failed_agents=failed_count,
            average_response_time=avg_response_time,
            system_uptime=uptime_hours,
            recommendations=recommendations
        )
    
    async def optimize_system_performance(self) -> Dict[str, str]:
        """Analyze system performance and provide optimization recommendations."""
        
        health_report = await self.get_system_health_report()
        
        optimization_context = f"""
        System Health: {health_report.overall_status}
        Performance Metrics: {self.performance_metrics}
        Agent Health: {[status.dict() for status in health_report.agent_statuses]}
        
        Analyze system performance and provide specific optimization recommendations.
        """
        
        result = await self.meta_agent.run(optimization_context)
        
        return {
            "analysis": result.output,
            "health_status": health_report.overall_status,
            "key_metrics": {
                "success_rate": self.performance_metrics["successful_requests"] / max(1, self.performance_metrics["total_requests"]),
                "avg_response_time": self.performance_metrics["average_response_time"],
                "system_uptime_hours": health_report.system_uptime
            }
        }


# Export the integrated system
__all__ = [
    "IntegratedRFQSystem",
    "ComprehensiveRFQResult",
    "SystemHealthReport",
    "AgentHealthStatus"
] 