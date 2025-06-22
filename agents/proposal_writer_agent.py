"""
Proposal Writer Agent

This agent generates professional proposal documents, presentations,
and supporting materials for RFQ responses.
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import CustomerIntent, Quote, RFQRequirements
from .utils import get_model_name


class ProposalDocument(BaseModel):
    """Professional proposal document."""
    executive_summary: str
    problem_statement: str
    proposed_solution: str
    technical_approach: str
    project_timeline: str
    team_qualifications: str
    pricing_section: str
    terms_and_conditions: str
    next_steps: str
    appendices: List[str] = Field(default_factory=list)
    presentation_outline: List[str] = Field(default_factory=list)


class ProposalWriterAgent:
    """Agent for generating professional proposal documents."""
    
    def __init__(self):
        model_name = get_model_name("question_generation")  # Reuse creative model
        self.agent = Agent(
            model_name,
            output_type=ProposalDocument,
            system_prompt="""
            You are a professional proposal writer who creates compelling,
            well-structured business proposals that win deals.
            
            Your proposals should include:
            
            EXECUTIVE SUMMARY:
            - Clear value proposition
            - Key benefits and outcomes
            - Investment summary
            - Call to action
            
            PROBLEM & SOLUTION:
            - Demonstrate understanding of customer needs
            - Present tailored solution approach
            - Highlight unique differentiators
            - Address specific requirements
            
            TECHNICAL APPROACH:
            - Detailed methodology
            - Implementation plan
            - Quality assurance measures
            - Risk mitigation strategies
            
            TEAM & QUALIFICATIONS:
            - Relevant experience and expertise
            - Past success stories
            - Team member qualifications
            - Company credentials
            
            PRICING & VALUE:
            - Clear, transparent pricing
            - Value justification
            - ROI demonstration
            - Flexible options when appropriate
            
            Write in professional, persuasive language that builds confidence
            and demonstrates expertise while remaining customer-focused.
            """
        )
    
    async def generate_proposal(
        self,
        requirements: RFQRequirements,
        customer_intent: CustomerIntent,
        quote: Quote,
        company_info: str = "",
        competitive_context: str = ""
    ) -> ProposalDocument:
        """
        Generate a comprehensive proposal document.
        
        Args:
            requirements: Customer requirements
            customer_intent: Customer characteristics
            quote: Generated quote information
            company_info: Company capabilities and background
            competitive_context: Competitive landscape information
            
        Returns:
            ProposalDocument: Professional proposal document
        """
        
        context = f"""
        CUSTOMER REQUIREMENTS:
        Product/Service: {requirements.product_type}
        Quantity/Scope: {requirements.quantity}
        Timeline: {requirements.delivery_date}
        Budget: {requirements.budget_range}
        Special Requirements: {requirements.special_requirements}
        
        CUSTOMER PROFILE:
        Primary Intent: {customer_intent.primary_intent}
        Urgency: {customer_intent.urgency_level}/5
        Price Sensitivity: {customer_intent.price_sensitivity}/5
        Decision Factors: {customer_intent.decision_factors}
        
        QUOTE DETAILS:
        Quote ID: {quote.quote_id}
        Total Price: ${quote.total_price:,.2f}
        Items: {quote.items}
        Delivery Terms: {quote.delivery_terms}
        Validity: {quote.validity_period}
        
        COMPANY INFORMATION:
        {company_info}
        
        COMPETITIVE CONTEXT:
        {competitive_context}
        
        Create a compelling, professional proposal that addresses the customer's
        specific needs and positions our solution as the optimal choice.
        """
        
        result = await self.agent.run(context)
        return result.output 