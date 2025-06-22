"""
Pricing Strategy Agent

This agent develops intelligent pricing strategies based on customer analysis and market conditions.
It considers customer sentiment, urgency, and competitive factors to optimize pricing.
"""

from pydantic_ai import Agent

from .models import CustomerIntent, PricingStrategy, RFQRequirements
from .utils import get_model_name


class PricingStrategyAgent:
    """Pricing Strategy agent for developing intelligent pricing strategies."""
    
    def __init__(self):
        model_name = get_model_name("pricing_strategy")
        self.agent = Agent(
            model_name,
            result_type=PricingStrategy,
            system_prompt="""
            You develop intelligent pricing strategies based on customer analysis and market conditions.
            Consider customer sentiment, urgency, and competitive factors.
            Use value-based pricing that reflects:
            - Market rates for similar products/services
            - Customer's perceived value and budget
            - Competitive positioning
            - Risk factors and complexity
            """
        )
    
    async def develop_strategy(self, requirements: RFQRequirements, intent: CustomerIntent) -> PricingStrategy:
        """
        Develop pricing strategy based on requirements and customer intent.
        
        Args:
            requirements: Parsed RFQ requirements
            intent: Customer intent analysis
            
        Returns:
            PricingStrategy: Intelligent pricing strategy with justification
        """
        context = f"Requirements: {requirements.model_dump()}\nIntent: {intent.model_dump()}"
        result = await self.agent.run(context)
        return result.output 