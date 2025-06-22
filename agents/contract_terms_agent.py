"""
Contract Terms Agent

This agent handles contract terms, legal considerations, compliance requirements,
and risk assessment for RFQ responses.
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import CustomerIntent, RFQRequirements
from .utils import get_model_name


class ContractTerms(BaseModel):
    """Contract terms and conditions."""
    payment_terms: str
    delivery_terms: str
    warranty_terms: str
    liability_limitations: List[str] = Field(default_factory=list)
    compliance_requirements: List[str] = Field(default_factory=list)
    termination_clauses: List[str] = Field(default_factory=list)
    intellectual_property: str
    risk_assessment: str
    recommended_clauses: List[str] = Field(default_factory=list)


class ContractTermsAgent:
    """Agent for contract terms and legal considerations."""
    
    def __init__(self):
        model_name = get_model_name("pricing_strategy")  # Reuse model
        self.agent = Agent(
            model_name,
            output_type=ContractTerms,
            system_prompt="""
            You are a contract specialist who develops appropriate terms and conditions
            for business agreements while managing legal and business risks.
            
            Your expertise covers:
            
            PAYMENT TERMS:
            - Industry-standard payment schedules
            - Risk-appropriate payment terms
            - Incentives for early payment
            - Protection against late payment
            
            DELIVERY & PERFORMANCE:
            - Realistic delivery commitments
            - Performance milestones and metrics
            - Change order procedures
            - Force majeure considerations
            
            RISK MANAGEMENT:
            - Appropriate liability limitations
            - Insurance requirements
            - Indemnification clauses
            - Dispute resolution mechanisms
            
            COMPLIANCE:
            - Industry-specific regulations
            - Data protection requirements
            - Security and confidentiality
            - Quality standards and certifications
            
            Balance customer needs with business protection while maintaining
            competitive positioning and relationship building.
            """
        )
    
    async def develop_contract_terms(
        self,
        requirements: RFQRequirements,
        customer_intent: CustomerIntent,
        business_context: str = ""
    ) -> ContractTerms:
        """
        Develop appropriate contract terms for the RFQ.
        
        Args:
            requirements: Customer requirements
            customer_intent: Customer characteristics
            business_context: Additional business context
            
        Returns:
            ContractTerms: Recommended contract terms and conditions
        """
        
        context = f"""
        PROJECT REQUIREMENTS:
        Product/Service: {requirements.product_type}
        Quantity/Scope: {requirements.quantity}
        Timeline: {requirements.delivery_date}
        Special Requirements: {requirements.special_requirements}
        Budget: {requirements.budget_range}
        
        CUSTOMER PROFILE:
        Urgency: {customer_intent.urgency_level}/5
        Price Sensitivity: {customer_intent.price_sensitivity}/5
        Decision Factors: {customer_intent.decision_factors}
        
        BUSINESS CONTEXT:
        {business_context}
        
        Develop comprehensive contract terms that protect our interests while
        remaining competitive and customer-friendly.
        """
        
        result = await self.agent.run(context)
        return result.output 