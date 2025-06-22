"""
Question Generation Agent

This agent generates strategic clarifying questions to better understand customer needs.
It focuses on the most critical missing information that would impact quote accuracy and customer satisfaction.
"""

from typing import List

from pydantic_ai import Agent

from .models import ClarifyingQuestion, CustomerIntent, RFQRequirements
from .utils import get_model_name


class QuestionGenerationAgent:
    """Question Generation agent for creating strategic, prioritized clarifying questions."""
    
    def __init__(self):
        model_name = get_model_name("question_generation")
        self.agent = Agent(
            model_name,
            result_type=List[ClarifyingQuestion],
            system_prompt="""
            You are an expert sales consultant who specializes in asking the right questions
            to understand customer needs and provide accurate quotes.
            
            Generate strategic clarifying questions that:
            1. Address the most critical missing information first
            2. Are tailored to the customer's urgency and buying readiness
            3. Help build trust and demonstrate expertise
            4. Guide the customer toward a successful purchase decision
            
            QUESTION PRIORITIES (1-5):
            5 = Critical for accurate quoting (quantity, specifications, timeline)
            4 = Important for pricing strategy (budget, decision process)
            3 = Valuable for customer satisfaction (preferences, concerns)
            2 = Helpful for relationship building (background, goals)
            1 = Nice to know but not essential
            
            QUESTION CATEGORIES:
            - "requirements": Core product/service specifications
            - "budget": Financial constraints and expectations
            - "timeline": Delivery and implementation schedules
            - "decision_process": Who decides and approval process
            - "technical": Technical specifications and integration needs
            - "business": Business goals and success criteria
            - "competitive": Alternative solutions being considered
            
            CUSTOMER ADAPTATION:
            - High urgency customers: Focus on essential questions only (priority 4-5)
            - Price-sensitive customers: Include budget clarification questions
            - High readiness customers: Focus on implementation details
            - Low readiness customers: Focus on value and benefits
            
            Always provide clear reasoning for why each question is important.
            Limit to 3-5 questions maximum to avoid overwhelming the customer.
            """
        )
    
    async def generate_questions(self, requirements: RFQRequirements, intent: CustomerIntent) -> List[ClarifyingQuestion]:
        """
        Generate strategic clarifying questions based on missing requirements and customer intent.
        
        Args:
            requirements: Parsed RFQ requirements with completeness assessment
            intent: Customer intent analysis including urgency and readiness
            
        Returns:
            List[ClarifyingQuestion]: Strategic questions prioritized by importance
        """
        context = f"""
        REQUIREMENTS ANALYSIS:
        - Product Type: {requirements.product_type}
        - Quantity: {requirements.quantity}
        - Budget Range: {requirements.budget_range}
        - Completeness Level: {requirements.completeness.value}
        - Missing Information: {requirements.missing_info}
        - Special Requirements: {requirements.special_requirements}
        
        CUSTOMER INTENT:
        - Primary Intent: {intent.primary_intent}
        - Sentiment: {intent.sentiment.value}
        - Urgency Level: {intent.urgency_level}/5
        - Price Sensitivity: {intent.price_sensitivity}/5
        - Readiness to Buy: {intent.readiness_to_buy}/5
        - Decision Factors: {intent.decision_factors}
        
        Based on this analysis, generate 3-5 strategic clarifying questions that:
        1. Address the most critical missing information for accurate quoting
        2. Are appropriate for the customer's urgency level and buying readiness
        3. Help move the sales process forward effectively
        
        Prioritize questions that will have the biggest impact on quote accuracy and customer satisfaction.
        For high-urgency customers, focus only on essential questions (priority 4-5).
        For price-sensitive customers, ensure budget clarification is included.
        
        Each question should include clear reasoning for why it's important to ask.
        """
        
        result = await self.agent.run(context)
        return result.output 