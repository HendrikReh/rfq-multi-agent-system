"""
Customer-related data models for the RFQ system.

This module contains models for customer information, intent analysis,
sentiment tracking, and interaction history.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerSentiment(str, Enum):
    """Customer sentiment classification."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    UNKNOWN = "unknown"


class CustomerUrgency(str, Enum):
    """Customer urgency level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerType(str, Enum):
    """Customer type classification."""
    ENTERPRISE = "enterprise"
    SMB = "smb"
    STARTUP = "startup"
    GOVERNMENT = "government"
    NON_PROFIT = "non_profit"
    INDIVIDUAL = "individual"


class DecisionFactor(BaseModel):
    """Factor influencing customer decision-making."""
    
    factor: str = Field(description="The decision factor")
    importance: int = Field(ge=1, le=5, description="Importance level 1-5")
    customer_priority: str = Field(description="How customer prioritizes this factor")
    our_advantage: bool = Field(
        default=False,
        description="Whether we have advantage in this factor"
    )


class CustomerIntent(BaseModel):
    """Analysis of customer intent and decision factors."""
    
    # Primary intent
    primary_intent: str = Field(description="Primary customer intent")
    intent_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in intent analysis"
    )
    
    # Customer characteristics
    customer_type: CustomerType = CustomerType.INDIVIDUAL
    urgency_level: CustomerUrgency = CustomerUrgency.MEDIUM
    sentiment: CustomerSentiment = CustomerSentiment.NEUTRAL
    
    # Decision factors
    decision_factors: List[DecisionFactor] = Field(
        default_factory=list,
        description="Factors influencing customer decisions"
    )
    
    # Budget and timeline sensitivity
    budget_sensitivity: int = Field(
        ge=1, le=5,
        default=3,
        description="Customer budget sensitivity (1=very flexible, 5=very strict)"
    )
    timeline_sensitivity: int = Field(
        ge=1, le=5,
        default=3,
        description="Customer timeline sensitivity (1=very flexible, 5=very strict)"
    )
    
    # Communication preferences
    preferred_communication_style: str = Field(
        default="professional",
        description="Preferred communication style"
    )
    technical_level: int = Field(
        ge=1, le=5,
        default=3,
        description="Customer technical knowledge level"
    )
    
    # Behavioral indicators
    price_shopping: bool = Field(
        default=False,
        description="Whether customer appears to be price shopping"
    )
    ready_to_buy: bool = Field(
        default=False,
        description="Whether customer appears ready to make a purchase"
    )
    
    # Risk factors
    risk_indicators: List[str] = Field(
        default_factory=list,
        description="Potential risk indicators"
    )
    
    # Opportunities
    upsell_opportunities: List[str] = Field(
        default_factory=list,
        description="Potential upselling opportunities"
    )
    
    # Analysis metadata
    analyzed_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str = Field(default="1.0")


class CustomerProfile(BaseModel):
    """Comprehensive customer profile."""
    
    # Basic information
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")
    company_name: Optional[str] = Field(None, description="Company name")
    contact_name: Optional[str] = Field(None, description="Primary contact name")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone")
    
    # Company details
    industry: Optional[str] = Field(None, description="Customer industry")
    company_size: Optional[str] = Field(None, description="Company size")
    annual_revenue: Optional[str] = Field(None, description="Annual revenue range")
    
    # Geographic information
    country: Optional[str] = Field(None, description="Country")
    region: Optional[str] = Field(None, description="Region or state")
    timezone: Optional[str] = Field(None, description="Customer timezone")
    
    # Relationship history
    previous_interactions: int = Field(
        default=0,
        description="Number of previous interactions"
    )
    previous_purchases: int = Field(
        default=0,
        description="Number of previous purchases"
    )
    total_purchase_value: float = Field(
        default=0.0,
        description="Total historical purchase value"
    )
    
    # Current analysis
    current_intent: Optional[CustomerIntent] = Field(
        None,
        description="Current intent analysis"
    )
    
    # Preferences and notes
    preferences: Dict[str, str] = Field(
        default_factory=dict,
        description="Customer preferences and notes"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class InteractionHistory(BaseModel):
    """Record of customer interactions."""
    
    interaction_id: str = Field(description="Unique interaction identifier")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    session_id: str = Field(description="Session identifier")
    
    # Interaction details
    interaction_type: str = Field(description="Type of interaction")
    channel: str = Field(description="Communication channel")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Content
    customer_message: str = Field(description="Customer message")
    system_response: str = Field(description="System response")
    
    # Analysis
    sentiment_at_time: CustomerSentiment = CustomerSentiment.NEUTRAL
    intent_at_time: str = Field(default="", description="Intent at time of interaction")
    
    # Outcomes
    resolution_status: str = Field(
        default="in_progress",
        description="Resolution status"
    )
    satisfaction_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Customer satisfaction score"
    )
    
    # Agent information
    handling_agent: str = Field(description="Agent that handled interaction")
    response_time_ms: float = Field(default=0.0, description="Response time")
    
    # Follow-up
    requires_followup: bool = Field(default=False)
    followup_date: Optional[datetime] = Field(None)
    followup_notes: str = Field(default="")


class CustomerFeedback(BaseModel):
    """Customer feedback and satisfaction data."""
    
    feedback_id: str = Field(description="Unique feedback identifier")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    session_id: str = Field(description="Associated session")
    
    # Feedback content
    rating: int = Field(ge=1, le=5, description="Overall rating 1-5")
    comment: str = Field(default="", description="Customer comment")
    
    # Specific ratings
    response_quality: Optional[int] = Field(None, ge=1, le=5)
    response_speed: Optional[int] = Field(None, ge=1, le=5)
    helpfulness: Optional[int] = Field(None, ge=1, le=5)
    accuracy: Optional[int] = Field(None, ge=1, le=5)
    
    # Improvement areas
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Customer suggestions for improvement"
    )
    
    # Metadata
    submitted_at: datetime = Field(default_factory=datetime.now)
    feedback_type: str = Field(default="post_interaction")
    
    # Analysis
    sentiment: CustomerSentiment = CustomerSentiment.NEUTRAL
    key_themes: List[str] = Field(
        default_factory=list,
        description="Key themes extracted from feedback"
    ) 