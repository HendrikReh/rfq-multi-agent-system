"""
Customer Response Agent

This agent simulates realistic customer responses to clarifying questions and system messages.
It helps demonstrate the complete interactive RFQ workflow by generating appropriate
customer replies based on different customer personas and scenarios.
"""

from typing import List

from pydantic_ai import Agent

from .models import ClarifyingQuestion, CustomerIntent
from .utils import get_model_name


class CustomerResponseAgent:
    """Agent that simulates realistic customer responses for demo purposes."""
    
    def __init__(self):
        model_name = get_model_name("customer_response")
        self.agent = Agent(
            model_name,
            result_type=str,
            system_prompt="""
            You are simulating a realistic customer responding to an RFQ (Request for Quote) system.
            Your role is to provide authentic, business-appropriate responses that match the customer's
            profile, urgency level, and communication style.
            
            Customer Response Guidelines:
            
            RESPONSE STYLE BASED ON URGENCY:
            - High urgency (4-5): Brief, direct responses focused on speed
            - Medium urgency (2-3): Balanced responses with reasonable detail
            - Low urgency (1): More detailed, exploratory responses
            
            RESPONSE STYLE BASED ON PRICE SENSITIVITY:
            - High price sensitivity (4-5): Focus on budget constraints, value, comparisons
            - Medium price sensitivity (2-3): Balanced cost/value considerations
            - Low price sensitivity (1): Focus on quality, features, service
            
            RESPONSE CONTENT:
            - Answer questions directly and realistically
            - Provide business context when appropriate
            - Include relevant details that a real customer would mention
            - Show decision-making process when relevant
            - Express concerns or priorities naturally
            
            BUSINESS AUTHENTICITY:
            - Use appropriate business language for the customer type
            - Include realistic constraints (timeline, budget, approval process)
            - Mention stakeholders when relevant (team, management, IT department)
            - Show understanding of business implications
            
            Keep responses conversational but professional, and ensure they feel like
            genuine customer communications rather than scripted answers.
            """
        )
    
    async def respond_to_questions(
        self, 
        questions: List[ClarifyingQuestion], 
        customer_intent: CustomerIntent,
        customer_persona: str,
        business_context: str = ""
    ) -> str:
        """
        Generate a realistic customer response to clarifying questions.
        
        Args:
            questions: List of clarifying questions from the system
            customer_intent: Customer's intent and characteristics
            customer_persona: Description of the customer type/persona
            business_context: Additional business context for the response
            
        Returns:
            str: Realistic customer response addressing the questions
        """
        questions_text = "\n".join([f"{i+1}. {q.question}" for i, q in enumerate(questions)])
        
        context = f"""
        CUSTOMER PROFILE:
        - Persona: {customer_persona}
        - Primary Intent: {customer_intent.primary_intent}
        - Sentiment: {customer_intent.sentiment.value}
        - Urgency Level: {customer_intent.urgency_level}/5
        - Price Sensitivity: {customer_intent.price_sensitivity}/5
        - Buying Readiness: {customer_intent.readiness_to_buy}/5
        - Decision Factors: {customer_intent.decision_factors}
        
        BUSINESS CONTEXT:
        {business_context}
        
        QUESTIONS TO ANSWER:
        {questions_text}
        
        Generate a realistic customer response that:
        1. Addresses each question appropriately for this customer type
        2. Matches the urgency and price sensitivity levels
        3. Includes realistic business details and constraints
        4. Shows the customer's decision-making process
        5. Feels authentic and conversational
        
        The response should be written as if the customer is replying via email or chat.
        """
        
        result = await self.agent.run(context)
        return result.output
    
    async def respond_to_quote(
        self,
        quote_message: str,
        customer_intent: CustomerIntent,
        customer_persona: str,
        response_type: str = "interested"
    ) -> str:
        """
        Generate a realistic customer response to a quote.
        
        Args:
            quote_message: The quote message from the system
            customer_intent: Customer's intent and characteristics
            customer_persona: Description of the customer type/persona
            response_type: Type of response (interested, negotiating, accepting, declining)
            
        Returns:
            str: Realistic customer response to the quote
        """
        context = f"""
        CUSTOMER PROFILE:
        - Persona: {customer_persona}
        - Primary Intent: {customer_intent.primary_intent}
        - Sentiment: {customer_intent.sentiment.value}
        - Urgency Level: {customer_intent.urgency_level}/5
        - Price Sensitivity: {customer_intent.price_sensitivity}/5
        - Buying Readiness: {customer_intent.readiness_to_buy}/5
        
        QUOTE RECEIVED:
        {quote_message}
        
        RESPONSE TYPE: {response_type}
        
        Generate a realistic customer response to this quote that:
        1. Matches the customer's price sensitivity and urgency
        2. Shows appropriate business decision-making process
        3. Includes realistic questions or concerns for this customer type
        4. Reflects the specified response type ({response_type})
        5. Feels authentic and business-appropriate
        
        Response types:
        - interested: Shows interest, asks follow-up questions
        - negotiating: Requests adjustments to price or terms
        - accepting: Accepts the quote and wants to proceed
        - declining: Politely declines with reasons
        
        Write as if the customer is responding via email or chat.
        """
        
        result = await self.agent.run(context)
        return result.output
    
    async def generate_follow_up_inquiry(
        self,
        customer_persona: str,
        inquiry_type: str = "general"
    ) -> str:
        """
        Generate a realistic follow-up customer inquiry.
        
        Args:
            customer_persona: Description of the customer type/persona
            inquiry_type: Type of inquiry (general, urgent, budget_conscious, detailed)
            
        Returns:
            str: Realistic customer inquiry message
        """
        context = f"""
        CUSTOMER PERSONA: {customer_persona}
        INQUIRY TYPE: {inquiry_type}
        
        Generate a realistic customer inquiry that would start an RFQ process.
        
        Inquiry types:
        - general: Vague initial inquiry with minimal details
        - urgent: Time-pressured request with urgency indicators
        - budget_conscious: Price-focused inquiry with budget concerns
        - detailed: Comprehensive request with specific requirements
        
        The inquiry should:
        1. Match the customer persona and inquiry type
        2. Feel like a genuine business communication
        3. Include appropriate level of detail for the type
        4. Use realistic business language and context
        5. Show the customer's priorities and concerns
        
        Write as if the customer is initiating contact via email or chat.
        """
        
        result = await self.agent.run(context)
        return result.output 