"""
Customer Intent Agent

This agent analyzes customer intent, sentiment, and decision factors from their messages.
It determines emotional state, urgency level, price sensitivity, readiness to buy, and key decision drivers.
"""

from pydantic_ai import Agent

from .models import CustomerIntent, CustomerSentiment
from .utils import get_model_name


class CustomerIntentAgent:
    """Customer Intent agent for analyzing customer sentiment, intent, and buying readiness."""
    
    def __init__(self):
        model_name = get_model_name("customer_intent")
        self.agent = Agent(
            model_name,
            result_type=CustomerIntent,
            system_prompt="""
            You are an expert sales psychologist and customer intent analyst. Analyze customer 
            messages to understand their intent, emotional state, and buying readiness.
            
            Analyze these key aspects:
            
            SENTIMENT ANALYSIS:
            - POSITIVE: Enthusiastic, optimistic, eager to proceed
            - NEUTRAL: Professional, factual, information-seeking
            - NEGATIVE: Frustrated, skeptical, or expressing concerns
            - URGENT: Time-pressured, needs quick resolution
            - PRICE_SENSITIVE: Focused on cost, budget constraints, comparing prices
            
            URGENCY LEVEL (1-5):
            1 = No rush, exploratory
            2 = Some timeline, but flexible
            3 = Moderate urgency, reasonable timeline
            4 = High urgency, tight timeline
            5 = Critical urgency, immediate need
            
            PRICE SENSITIVITY (1-5):
            1 = Price not a primary concern, value-focused
            2 = Price considered but not limiting
            3 = Balanced price/value consideration
            4 = Price is a significant factor
            5 = Price is the primary decision factor
            
            READINESS TO BUY (1-5):
            1 = Early research phase, not ready to purchase
            2 = Gathering information, comparing options
            3 = Evaluating solutions, getting closer to decision
            4 = Ready to move forward, needs final details
            5 = Immediate buying intent, ready to commit
            
            Look for indicators like:
            - Language urgency ("need ASAP", "urgent", "deadline")
            - Budget mentions ("budget of $X", "cost-effective", "affordable")
            - Decision authority ("I can approve", "need to check with team")
            - Timeline pressure ("by end of month", "immediately")
            - Comparison shopping ("comparing options", "best price")
            """
        )
    
    async def analyze(self, customer_message: str) -> CustomerIntent:
        """
        Analyze customer intent, sentiment, and buying readiness from their message.
        
        Args:
            customer_message: The customer's message text
            
        Returns:
            CustomerIntent: Comprehensive analysis of customer's intent and readiness
        """
        enhanced_prompt = f"""
        Analyze this customer message for intent, sentiment, urgency, price sensitivity, and buying readiness.
        
        Customer Message: "{customer_message}"
        
        Pay special attention to:
        - Emotional tone and sentiment
        - Urgency indicators and timeline pressure
        - Price/budget sensitivity signals
        - Decision-making authority and readiness
        - Specific decision factors they mention or imply
        
        Provide a comprehensive analysis of their buying intent and readiness level.
        """
        
        result = await self.agent.run(enhanced_prompt)
        return result.output 