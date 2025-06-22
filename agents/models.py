"""
Shared data models for RFQ processing agents.

This module contains all Pydantic models and enums used across the different agents
in the RFQ orchestrator system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


# Enums
class ConversationState(str, Enum):
    """Possible states of the RFQ conversation."""
    INITIAL = "initial"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    CLARIFICATION = "clarification"
    PRICING = "pricing"
    NEGOTIATION = "negotiation"
    QUOTE_GENERATION = "quote_generation"
    COMPLETED = "completed"


class CustomerSentiment(str, Enum):
    """Customer sentiment analysis results."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    URGENT = "urgent"
    PRICE_SENSITIVE = "price_sensitive"


class RequirementsCompleteness(str, Enum):
    """Assessment of how complete the requirements are."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    UNCLEAR = "unclear"


# Pydantic Models
class RFQRequirements(BaseModel):
    """Parsed RFQ requirements."""
    product_type: str
    quantity: Optional[int] = None
    specifications: Dict[str, str] = Field(default_factory=dict)
    delivery_date: Optional[datetime] = None
    timeline: Optional[str] = None
    budget_range: Optional[str] = None
    technical_requirements: List[str] = Field(default_factory=list)
    compliance_requirements: List[str] = Field(default_factory=list)
    special_requirements: List[str] = Field(default_factory=list)
    completeness: RequirementsCompleteness = RequirementsCompleteness.UNCLEAR
    missing_info: List[str] = Field(default_factory=list)


class CustomerIntent(BaseModel):
    """Customer intent and sentiment analysis."""
    primary_intent: str
    sentiment: CustomerSentiment
    urgency_level: int = Field(ge=1, le=5, description="Urgency from 1-5")
    price_sensitivity: int = Field(ge=1, le=5, description="Price sensitivity from 1-5")
    decision_factors: List[str] = Field(default_factory=list)
    readiness_to_buy: int = Field(ge=1, le=5, description="How ready they are to purchase from 1-5")


class ClarifyingQuestion(BaseModel):
    """Strategic clarifying question."""
    question: str
    category: str
    priority: int = Field(ge=1, le=5)
    expected_response_type: str
    reasoning: str = Field(description="Why this question is important")


class InteractionDecision(BaseModel):
    """Decision on how to proceed with the customer interaction."""
    should_ask_questions: bool
    should_generate_quote: bool
    next_action: str
    reasoning: str
    confidence_level: int = Field(ge=1, le=5, description="Confidence in having enough info")


class PricingStrategy(BaseModel):
    """Intelligent pricing strategy."""
    strategy_type: str
    base_price: float
    discount_percentage: float = 0.0
    markup_percentage: float = 0.0
    justification: str
    competitive_factors: List[str] = Field(default_factory=list)


class Quote(BaseModel):
    """Generated quote."""
    quote_id: str
    items: List[Dict[str, Union[str, float, int]]]
    total_price: float
    delivery_terms: str
    validity_period: str
    special_conditions: List[str] = Field(default_factory=list)


class SystemPerformance(BaseModel):
    """System performance evaluation."""
    response_time: float
    accuracy_score: float
    customer_satisfaction_prediction: float
    improvement_suggestions: List[str] = Field(default_factory=list)


class RFQProcessingResult(BaseModel):
    """Complete result of RFQ processing."""
    status: str
    conversation_state: str
    requirements: RFQRequirements
    customer_intent: CustomerIntent
    interaction_decision: InteractionDecision
    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)
    pricing_strategy: Optional[PricingStrategy] = None
    quote: Optional[Quote] = None
    performance: Optional[SystemPerformance] = None
    next_steps: List[str] = Field(default_factory=list)
    message_to_customer: str


# Enhanced Agent Models
class CompetitiveAnalysis(BaseModel):
    """Competitive analysis results."""
    market_position: str
    competitor_analysis: List[str]
    win_probability: float = Field(ge=0.0, le=1.0)
    differentiation_strategy: List[str]
    recommended_approach: str


class RiskAssessment(BaseModel):
    """Risk assessment results."""
    overall_risk_score: int = Field(ge=1, le=10)
    risk_level: str  # low, medium, high
    risk_categories: Dict[str, int] = Field(default_factory=dict)
    mitigation_strategies: List[str]
    go_no_go_recommendation: str


class ContractTerms(BaseModel):
    """Contract terms and conditions."""
    payment_terms: str
    delivery_terms: str
    liability_limitations: List[str]
    compliance_requirements: List[str]
    termination_clauses: Optional[str] = None
    intellectual_property_terms: Optional[str] = None
    service_level_agreements: Optional[str] = None


class ProposalDocument(BaseModel):
    """Professional proposal document."""
    executive_summary: str
    technical_approach: str
    pricing_justification: str
    implementation_timeline: str
    risk_mitigation: Optional[str] = None
    success_metrics: Optional[str] = None
    next_steps: Optional[str] = None


class AgentHealthStatus(BaseModel):
    """Health status of an individual agent."""
    agent_name: str
    status: str  # healthy, degraded, unhealthy
    last_response_time: float
    error_count: int = 0
    last_error: Optional[str] = None


class SystemHealthReport(BaseModel):
    """Overall system health report."""
    overall_status: str  # healthy, degraded, unhealthy
    total_agents: int
    healthy_agents: int
    response_time_avg: float = 0.0
    uptime_percentage: float = 100.0
    agent_statuses: List[AgentHealthStatus] = Field(default_factory=list)


class ComprehensiveRFQResult(BaseModel):
    """Comprehensive RFQ processing result with all agent outputs."""
    rfq_requirements: RFQRequirements
    customer_intent: CustomerIntent
    interaction_decision: InteractionDecision
    competitive_analysis: Optional[CompetitiveAnalysis] = None
    risk_assessment: Optional[RiskAssessment] = None
    contract_terms: Optional[ContractTerms] = None
    proposal_document: Optional[ProposalDocument] = None
    pricing_strategy: Optional[PricingStrategy] = None
    quote: Optional[Quote] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time: float = 0.0
    execution_mode: str = "sequential"


class AgentCoordinationContext(BaseModel):
    """Context for agent coordination and delegation."""
    primary_agent: str
    delegated_agents: List[str] = Field(default_factory=list)
    coordination_strategy: str = "sequential"
    shared_context: Dict[str, str] = Field(default_factory=dict)


# Dependencies
@dataclass
class RFQDependencies:
    """Dependencies for the RFQ processing system."""
    customer_id: str = "default_customer"
    session_id: str = "default_session"
    conversation_history: List[str] = None
    current_state: ConversationState = ConversationState.INITIAL
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = [] 