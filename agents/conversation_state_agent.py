"""
Conversation State Agent

This agent tracks conversation flow and determines the current stage of the RFQ process.
It analyzes conversation history to understand where we are in the sales process.
"""

from typing import List

from pydantic_ai import Agent

from .models import ConversationState
from .utils import get_model_name


class ConversationStateAgent:
    """Conversation State agent for tracking conversation flow."""
    
    def __init__(self):
        model_name = get_model_name("conversation_state")
        self.agent = Agent(
            model_name,
            result_type=ConversationState,
            system_prompt="""
            You determine the current state of the RFQ conversation.
            Analyze the conversation flow and determine what stage we're in:
            - INITIAL: First contact or greeting
            - REQUIREMENTS_GATHERING: Collecting basic needs
            - CLARIFICATION: Asking follow-up questions
            - PRICING: Discussing costs and pricing
            - NEGOTIATION: Adjusting terms or prices
            - QUOTE_GENERATION: Finalizing the quote
            - COMPLETED: Process finished
            """
        )
    
    async def determine_state(self, conversation_history: List[str]) -> ConversationState:
        """
        Determine current conversation state based on message history.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            ConversationState: Current state of the conversation
        """
        context = "\n".join(conversation_history[-5:])  # Last 5 messages
        result = await self.agent.run(f"Conversation history: {context}")
        return result.output 