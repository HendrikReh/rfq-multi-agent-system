"""
Risk Assessment Agent

This agent evaluates various risks associated with RFQ opportunities including
business risks, project risks, customer risks, and financial risks.
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import CustomerIntent, RFQRequirements
from .utils import get_model_name


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment."""
    overall_risk_level: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    financial_risks: List[str] = Field(default_factory=list)
    operational_risks: List[str] = Field(default_factory=list)
    customer_risks: List[str] = Field(default_factory=list)
    project_risks: List[str] = Field(default_factory=list)
    market_risks: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=10.0, description="Risk score from 0-10")
    recommendation: str = Field(description="PROCEED, PROCEED_WITH_CAUTION, or DECLINE")
    insurance_requirements: List[str] = Field(default_factory=list)


class RiskAssessmentAgent:
    """Agent for comprehensive risk assessment and mitigation."""
    
    def __init__(self):
        model_name = get_model_name("evaluation_intelligence")  # Reuse model
        self.agent = Agent(
            model_name,
            output_type=RiskAssessment,
            system_prompt="""
            You are a risk assessment specialist who evaluates business opportunities
            for potential risks and develops mitigation strategies.
            
            Assess these risk categories:
            
            FINANCIAL RISKS:
            - Payment default risk
            - Currency/economic risks
            - Budget overrun potential
            - Cash flow impact
            - Pricing pressure risks
            
            OPERATIONAL RISKS:
            - Delivery capability risks
            - Resource availability
            - Technical complexity risks
            - Quality assurance challenges
            - Scalability concerns
            
            CUSTOMER RISKS:
            - Customer financial stability
            - Decision-making reliability
            - Communication effectiveness
            - Scope creep potential
            - Relationship management challenges
            
            PROJECT RISKS:
            - Timeline feasibility
            - Technical requirements clarity
            - Dependencies and assumptions
            - Change management needs
            - Success criteria definition
            
            MARKET RISKS:
            - Competitive pressure
            - Market volatility
            - Regulatory changes
            - Technology disruption
            - Economic conditions
            
            Provide actionable risk mitigation strategies and clear recommendations.
            """
        )
    
    async def assess_risks(
        self,
        requirements: RFQRequirements,
        customer_intent: CustomerIntent,
        business_context: str = "",
        historical_data: Dict = None
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for the RFQ opportunity.
        
        Args:
            requirements: Customer requirements
            customer_intent: Customer characteristics
            business_context: Additional business context
            historical_data: Historical data about similar projects or customers
            
        Returns:
            RiskAssessment: Comprehensive risk analysis and recommendations
        """
        
        historical_context = ""
        if historical_data:
            historical_context = f"Historical Data: {historical_data}"
        
        context = f"""
        OPPORTUNITY DETAILS:
        Product/Service: {requirements.product_type}
        Scope: {requirements.quantity}
        Timeline: {requirements.delivery_date}
        Budget: {requirements.budget_range}
        Special Requirements: {requirements.special_requirements}
        Requirements Completeness: {requirements.completeness.value}
        
        CUSTOMER PROFILE:
        Urgency Level: {customer_intent.urgency_level}/5
        Price Sensitivity: {customer_intent.price_sensitivity}/5
        Buying Readiness: {customer_intent.readiness_to_buy}/5
        Decision Factors: {customer_intent.decision_factors}
        Sentiment: {customer_intent.sentiment.value}
        
        BUSINESS CONTEXT:
        {business_context}
        
        {historical_context}
        
        Perform a comprehensive risk assessment and provide specific mitigation
        strategies for identified risks.
        """
        
        result = await self.agent.run(context)
        return result.output 