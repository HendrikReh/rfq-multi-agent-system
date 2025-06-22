"""
Interaction Decision Agent

This agent analyzes the current state of requirements and customer intent to decide
whether to ask clarifying questions or proceed with quote generation.
"""

from pydantic_ai import Agent

from .models import CustomerIntent, InteractionDecision, RFQRequirements
from .utils import get_model_name


class InteractionDecisionAgent:
    """Agent that decides how to proceed with customer interaction."""
    
    def __init__(self):
        model_name = get_model_name("interaction_decision")
        self.agent = Agent(
            model_name,
            result_type=InteractionDecision,
            system_prompt="""
            You are an expert sales interaction strategist. Your role is to analyze customer
            requirements and intent to decide the best next step in the sales process.
            
            Consider these factors when making decisions:
            - Completeness of requirements (missing critical information?)
            - Customer readiness and urgency
            - Confidence level in providing accurate quotes
            - Strategic value of gathering more information
            
            Guidelines:
            - If requirements are UNCLEAR or MINIMAL, ask clarifying questions
            - If customer shows high urgency but requirements are incomplete, prioritize key questions
            - If requirements are COMPLETE and customer shows buying intent, proceed to quote
            - If customer seems price-sensitive, ensure budget information is clear
            - Always explain your reasoning clearly
            """
        )
    
    async def decide_next_action(self, requirements: RFQRequirements, intent: CustomerIntent) -> InteractionDecision:
        """
        Decide whether to ask questions or proceed with quote generation.
        
        Args:
            requirements: Current parsed requirements
            intent: Customer intent analysis
            
        Returns:
            InteractionDecision: Decision on how to proceed
        """
        context = f"""
        Requirements Analysis:
        - Product Type: {requirements.product_type}
        - Quantity: {requirements.quantity}
        - Budget Range: {requirements.budget_range}
        - Completeness: {requirements.completeness.value}
        - Missing Info: {requirements.missing_info}
        - Special Requirements: {requirements.special_requirements}
        
        Customer Intent:
        - Primary Intent: {intent.primary_intent}
        - Sentiment: {intent.sentiment.value}
        - Urgency Level: {intent.urgency_level}/5
        - Price Sensitivity: {intent.price_sensitivity}/5
        - Readiness to Buy: {intent.readiness_to_buy}/5
        - Decision Factors: {intent.decision_factors}
        
        Based on this information, decide whether to:
        1. Ask clarifying questions to gather more information
        2. Proceed directly to quote generation
        
        Consider the completeness of requirements, customer urgency, and confidence in providing accurate quotes.
        """
        
        result = await self.agent.run(context)
        return result.output 