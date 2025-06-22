"""
Core RFQ (Request for Quote) data models.

This module contains the primary data models for RFQ processing,
including requirements, processing results, and related structures.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RequirementsCompleteness(str, Enum):
    """Assessment of how complete the requirements are."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    UNCLEAR = "unclear"


class RFQStatus(str, Enum):
    """RFQ processing status."""
    RECEIVED = "received"
    PROCESSING = "processing"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    ANALYSIS = "analysis"
    PRICING = "pricing"
    PROPOSAL_GENERATION = "proposal_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class RFQPriority(str, Enum):
    """RFQ processing priority."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RFQRequirements(BaseModel):
    """Structured RFQ requirements extracted from customer requests."""
    
    # Core requirements
    product_type: str = Field(description="Type of product or service requested")
    quantity: Optional[int] = Field(None, description="Quantity requested")
    budget_range: Optional[str] = Field(None, description="Budget range or constraints")
    delivery_date: Optional[datetime] = Field(None, description="Required delivery date")
    timeline: Optional[str] = Field(None, description="Project timeline")
    
    # Technical specifications
    technical_requirements: List[str] = Field(
        default_factory=list, 
        description="Technical requirements and specifications"
    )
    specifications: Dict[str, str] = Field(
        default_factory=dict,
        description="Detailed product specifications"
    )
    
    # Compliance and legal
    compliance_requirements: List[str] = Field(
        default_factory=list,
        description="Compliance and regulatory requirements"
    )
    special_requirements: List[str] = Field(
        default_factory=list,
        description="Special or unique requirements"
    )
    
    # Assessment metadata
    completeness: RequirementsCompleteness = RequirementsCompleteness.UNCLEAR
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing information needed"
    )
    confidence_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Confidence in requirements extraction"
    )
    
    # Source information
    source_documents: List[str] = Field(
        default_factory=list,
        description="Source documents or references"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="When requirements were extracted"
    )


class ClarifyingQuestion(BaseModel):
    """Strategic clarifying question to gather more information."""
    
    question: str = Field(description="The question to ask")
    category: str = Field(description="Category of the question")
    priority: int = Field(ge=1, le=5, description="Priority from 1-5")
    expected_response_type: str = Field(description="Expected type of response")
    reasoning: str = Field(description="Why this question is important")
    required_for_completion: bool = Field(
        default=False,
        description="Whether this question is required for completion"
    )


class InteractionDecision(BaseModel):
    """Decision on how to proceed with customer interaction."""
    
    should_ask_questions: bool = Field(description="Whether to ask clarifying questions")
    should_generate_quote: bool = Field(description="Whether to generate a quote")
    next_action: str = Field(description="Recommended next action")
    reasoning: str = Field(description="Reasoning for the decision")
    confidence_level: int = Field(
        ge=1, le=5, 
        description="Confidence in having enough information"
    )
    estimated_completion_time: Optional[str] = Field(
        None,
        description="Estimated time to complete the RFQ"
    )


class PricingStrategy(BaseModel):
    """Intelligent pricing strategy for the RFQ."""
    
    strategy_type: str = Field(description="Type of pricing strategy")
    base_price: float = Field(ge=0, description="Base price calculation")
    discount_percentage: float = Field(default=0.0, ge=0, le=100)
    markup_percentage: float = Field(default=0.0, ge=0)
    final_price: float = Field(ge=0, description="Final calculated price")
    
    # Strategy details
    justification: str = Field(description="Justification for pricing strategy")
    competitive_factors: List[str] = Field(
        default_factory=list,
        description="Competitive factors considered"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Risk factors affecting pricing"
    )
    
    # Pricing breakdown
    cost_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of costs"
    )
    profit_margin: float = Field(default=0.0, description="Profit margin percentage")


class Quote(BaseModel):
    """Generated quote for the RFQ."""
    
    quote_id: str = Field(description="Unique quote identifier")
    rfq_id: Optional[str] = Field(None, description="Associated RFQ ID")
    
    # Quote items
    items: List[Dict[str, Union[str, float, int]]] = Field(
        description="List of quoted items"
    )
    
    # Pricing
    subtotal: float = Field(ge=0, description="Subtotal before taxes and fees")
    tax_amount: float = Field(default=0.0, ge=0, description="Tax amount")
    total_price: float = Field(ge=0, description="Total quote price")
    
    # Terms
    delivery_terms: str = Field(description="Delivery terms and conditions")
    payment_terms: str = Field(description="Payment terms")
    validity_period: str = Field(description="Quote validity period")
    
    # Additional terms
    special_conditions: List[str] = Field(
        default_factory=list,
        description="Special conditions or terms"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made in the quote"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None)
    version: int = Field(default=1, description="Quote version number")


class SystemPerformance(BaseModel):
    """System performance evaluation metrics."""
    
    response_time_ms: float = Field(ge=0, description="Response time in milliseconds")
    accuracy_score: float = Field(ge=0, le=1, description="Processing accuracy score")
    customer_satisfaction_prediction: float = Field(
        ge=0, le=1,
        description="Predicted customer satisfaction"
    )
    
    # Performance breakdown
    agent_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual agent performance scores"
    )
    bottlenecks: List[str] = Field(
        default_factory=list,
        description="Identified performance bottlenecks"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )


class RFQProcessingResult(BaseModel):
    """Complete result of RFQ processing."""
    
    # Processing metadata
    rfq_id: str = Field(description="Unique RFQ identifier")
    status: RFQStatus = Field(description="Processing status")
    priority: RFQPriority = Field(default=RFQPriority.MEDIUM)
    
    # Core processing results
    requirements: RFQRequirements = Field(description="Extracted requirements")
    interaction_decision: InteractionDecision = Field(description="Interaction decision")
    
    # Optional results (depending on processing stage)
    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)
    pricing_strategy: Optional[PricingStrategy] = Field(None)
    quote: Optional[Quote] = Field(None)
    
    # Performance and quality
    performance: Optional[SystemPerformance] = Field(None)
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall confidence in the result"
    )
    
    # Processing details
    processing_time_ms: float = Field(default=0.0, ge=0)
    agents_involved: List[str] = Field(
        default_factory=list,
        description="List of agents involved in processing"
    )
    execution_mode: str = Field(
        default="sequential",
        description="Execution mode used (sequential, parallel, etc.)"
    )
    
    # Customer communication
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    message_to_customer: str = Field(
        default="",
        description="Message to send to customer"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)
    
    # Error handling
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during processing"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during processing"
    ) 