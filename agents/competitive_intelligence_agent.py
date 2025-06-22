"""
Competitive Intelligence Agent

This agent analyzes competitive landscape, market conditions, and positioning
to help optimize quotes and strategies against competitors.
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import CustomerIntent, RFQRequirements
from .utils import get_model_name


class CompetitiveAnalysis(BaseModel):
    """Competitive analysis results."""
    market_position: str = Field(description="Our position in the market")
    competitor_threats: List[str] = Field(default_factory=list)
    competitive_advantages: List[str] = Field(default_factory=list)
    pricing_benchmarks: Dict[str, float] = Field(default_factory=dict)
    win_probability: float = Field(ge=0.0, le=1.0, description="Probability of winning this deal")
    recommended_strategy: str
    differentiation_points: List[str] = Field(default_factory=list)


class CompetitiveIntelligenceAgent:
    """Agent for competitive analysis and market intelligence."""
    
    def __init__(self):
        model_name = get_model_name("customer_intent")  # Reuse existing model type
        self.agent = Agent(
            model_name,
            output_type=CompetitiveAnalysis,
            system_prompt="""
            You are a competitive intelligence expert who analyzes market conditions
            and competitive landscape to optimize sales strategies.
            
            Your analysis should consider:
            
            MARKET POSITIONING:
            - Our strengths vs competitors
            - Market share and reputation
            - Brand perception and trust factors
            - Service quality differentiators
            
            COMPETITIVE THREATS:
            - Known competitors likely to bid
            - Their typical pricing strategies
            - Their strengths and weaknesses
            - Customer relationships they may have
            
            PRICING ANALYSIS:
            - Market rate benchmarks
            - Competitor pricing patterns
            - Price sensitivity in this segment
            - Value-based pricing opportunities
            
            WIN STRATEGY:
            - Key differentiators to emphasize
            - Competitive advantages to highlight
            - Potential objections to address
            - Relationship-building opportunities
            
            Consider customer urgency, budget constraints, and decision factors
            when developing competitive strategy.
            """
        )
    
    async def analyze_competitive_landscape(
        self,
        requirements: RFQRequirements,
        customer_intent: CustomerIntent,
        market_context: str = ""
    ) -> CompetitiveAnalysis:
        """
        Analyze competitive landscape for this specific RFQ.
        
        Args:
            requirements: Customer requirements
            customer_intent: Customer intent and characteristics
            market_context: Additional market context information
            
        Returns:
            CompetitiveAnalysis: Comprehensive competitive analysis
        """
        
        context = f"""
        CUSTOMER REQUIREMENTS:
        Product Type: {requirements.product_type}
        Quantity: {requirements.quantity}
        Budget Range: {requirements.budget_range}
        Timeline: {requirements.delivery_date}
        Special Requirements: {requirements.special_requirements}
        
        CUSTOMER CHARACTERISTICS:
        Urgency Level: {customer_intent.urgency_level}/5
        Price Sensitivity: {customer_intent.price_sensitivity}/5
        Decision Factors: {customer_intent.decision_factors}
        Buying Readiness: {customer_intent.readiness_to_buy}/5
        
        MARKET CONTEXT:
        {market_context}
        
        Analyze the competitive landscape and provide strategic recommendations
        for winning this specific deal.
        """
        
        result = await self.agent.run(context)
        return result.output 