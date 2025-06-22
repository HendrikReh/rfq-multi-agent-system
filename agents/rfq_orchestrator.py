"""
RFQ Orchestrator Agent

Main orchestrator agent that coordinates all RFQ processing agents.
This agent manages an interactive workflow that asks clarifying questions when needed
and generates quotes when sufficient information is available.
"""

from datetime import datetime
from typing import Any, Dict, List, Union

from pydantic_ai import Agent

from .conversation_state_agent import ConversationStateAgent
from .customer_intent_agent import CustomerIntentAgent
from .evaluation_intelligence_agent import EvaluationIntelligenceAgent
from .interaction_decision_agent import InteractionDecisionAgent
from .models import (
    ClarifyingQuestion,
    ConversationState,
    CustomerIntent,
    CustomerSentiment,
    InteractionDecision,
    PricingStrategy,
    Quote,
    RFQDependencies,
    RFQProcessingResult,
    RFQRequirements,
)
from .pricing_strategy_agent import PricingStrategyAgent
from .question_generation_agent import QuestionGenerationAgent
from .rfq_parser import RFQParser
from .utils import get_model_name


class RFQOrchestrator:
    """
    Main orchestrator agent that coordinates all RFQ processing agents.
    
    This agent manages an intelligent, interactive workflow:
    1. Parsing RFQ requirements and assessing completeness
    2. Analyzing customer intent and buying readiness
    3. Deciding whether to ask clarifying questions or proceed with quotes
    4. Generating strategic questions when more information is needed
    5. Creating accurate quotes when sufficient information is available
    6. Evaluating system performance and customer satisfaction
    """
    
    def __init__(self):
        # Initialize the main orchestrator agent
        model_name = get_model_name("rfq_orchestrator")
        self.agent = Agent(
            model_name,
            deps_type=RFQDependencies,
            result_type=str,
            system_prompt="""
            You are the RFQ Orchestrator, an expert sales system coordinator that manages
            intelligent customer interactions for quote processing.
            
            Your role is to:
            1. Coordinate multiple specialized agents for optimal customer experience
            2. Make intelligent decisions about when to ask questions vs. provide quotes
            3. Ensure customer needs are fully understood before quoting
            4. Maintain professional, helpful communication throughout the process
            5. Optimize for both accuracy and customer satisfaction
            
            Always prioritize understanding customer needs completely before generating quotes.
            Be strategic about question timing based on customer urgency and buying readiness.
            """
        )
        
        # Initialize specialized agents
        self.rfq_parser = RFQParser()
        self.state_agent = ConversationStateAgent()
        self.intent_agent = CustomerIntentAgent()
        self.decision_agent = InteractionDecisionAgent()
        self.question_agent = QuestionGenerationAgent()
        self.pricing_agent = PricingStrategyAgent()
        self.evaluation_agent = EvaluationIntelligenceAgent()
    
    async def process_rfq(self, customer_message: str, deps: RFQDependencies) -> RFQProcessingResult:
        """
        Main method to process an RFQ through the intelligent interactive workflow.
        
        Args:
            customer_message: The customer's RFQ message
            deps: RFQ processing dependencies
            
        Returns:
            RFQProcessingResult: Complete processing results with next steps
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Parse RFQ requirements and assess completeness
            requirements = await self.rfq_parser.parse(customer_message)
            
            # Step 2: Determine conversation state
            current_state = await self.state_agent.determine_state(deps.conversation_history)
            
            # Step 3: Analyze customer intent and buying readiness
            intent = await self.intent_agent.analyze(customer_message)
            
            # Step 4: Decide whether to ask questions or proceed with quote
            decision = await self.decision_agent.decide_next_action(requirements, intent)
            
            # Step 5: Generate response based on decision
            if decision.should_ask_questions:
                # Generate strategic clarifying questions
                questions = await self.question_agent.generate_questions(requirements, intent)
                
                # Create customer message for clarifying questions
                customer_message = self._create_question_message(questions, intent)
                
                # Prepare result with questions
                result = RFQProcessingResult(
                    status="questions_needed",
                    conversation_state=current_state.value,
                    requirements=requirements,
                    customer_intent=intent,
                    interaction_decision=decision,
                    clarifying_questions=questions,
                    next_steps=self._determine_question_next_steps(questions),
                    message_to_customer=customer_message
                )
                
            else:
                # Proceed with quote generation
                pricing_strategy = await self.pricing_agent.develop_strategy(requirements, intent)
                quote = self._generate_quote(requirements, pricing_strategy)
                
                # Evaluate system performance
                processing_time = (datetime.now() - start_time).total_seconds()
                performance = await self.evaluation_agent.evaluate(processing_time, quote)
                
                # Create customer message for quote
                customer_message = self._create_quote_message(quote, pricing_strategy)
                
                # Prepare result with quote
                result = RFQProcessingResult(
                    status="quote_generated",
                    conversation_state=current_state.value,
                    requirements=requirements,
                    customer_intent=intent,
                    interaction_decision=decision,
                    pricing_strategy=pricing_strategy,
                    quote=quote,
                    performance=performance,
                    next_steps=self._determine_quote_next_steps(current_state),
                    message_to_customer=customer_message
                )
            
            return result
            
        except Exception as e:
            # Create default customer intent for error cases
            default_intent = CustomerIntent(
                primary_intent="unknown",
                sentiment=CustomerSentiment.NEUTRAL,
                urgency_level=1,
                price_sensitivity=1,
                readiness_to_buy=1,
                decision_factors=[]
            )
            
            return RFQProcessingResult(
                status="error",
                conversation_state="error",
                requirements=RFQRequirements(product_type="unknown"),
                customer_intent=intent if 'intent' in locals() else default_intent,
                interaction_decision=InteractionDecision(
                    should_ask_questions=False,
                    should_generate_quote=False,
                    next_action="retry",
                    reasoning=f"System error: {str(e)}",
                    confidence_level=1
                ),
                next_steps=["Please retry with a clearer request", "Contact support if the issue persists"],
                message_to_customer=f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again or contact our support team."
            )
    
    def _create_question_message(self, questions: List[ClarifyingQuestion], intent) -> str:
        """Create a professional message with clarifying questions."""
        urgency_note = ""
        if intent.urgency_level >= 4:
            urgency_note = "I understand this is urgent, so I'll keep this brief. "
        
        message = f"Thank you for your interest! {urgency_note}To provide you with the most accurate quote, I have a few key questions:\n\n"
        
        for i, question in enumerate(questions, 1):
            message += f"{i}. {question.question}\n"
        
        message += "\nOnce I have this information, I'll be able to provide you with a detailed, accurate quote tailored to your specific needs."
        
        return message
    
    def _create_quote_message(self, quote: Quote, strategy: PricingStrategy) -> str:
        """Create a professional message presenting the quote."""
        message = f"Thank you for providing the details! I'm pleased to present your customized quote:\n\n"
        message += f"**Quote ID:** {quote.quote_id}\n\n"
        
        message += "**Items:**\n"
        for item in quote.items:
            message += f"• {item['description']}: {item['quantity']} × ${item['unit_price']:,.2f} = ${item['total']:,.2f}\n"
        
        message += f"\n**Total Investment:** ${quote.total_price:,.2f}\n\n"
        message += f"**Delivery:** {quote.delivery_terms}\n"
        message += f"**Quote Valid:** {quote.validity_period}\n\n"
        
        if quote.special_conditions:
            message += "**Terms & Conditions:**\n"
            for condition in quote.special_conditions:
                message += f"• {condition}\n"
        
        message += f"\n**Why this pricing:** {strategy.justification}\n\n"
        message += "I'm here to answer any questions or discuss how we can move forward. Would you like to proceed with this quote?"
        
        return message
    
    def _generate_quote(self, requirements: RFQRequirements, strategy: PricingStrategy) -> Quote:
        """Generate a quote based on requirements and pricing strategy."""
        base_price = strategy.base_price
        final_price = base_price * (1 - strategy.discount_percentage / 100)
        quantity = requirements.quantity or 1
        
        # Create detailed line items
        items = []
        
        # Main product/service
        unit_price = final_price / quantity
        items.append({
            "description": requirements.product_type,
            "quantity": quantity,
            "unit_price": round(unit_price, 2),
            "total": round(final_price, 2)
        })
        
        # Add special requirements as separate line items
        additional_cost = 0
        for req in requirements.special_requirements:
            if "24/7 support" in req:
                support_cost = final_price * 0.15  # 15% of base for premium support
                items.append({
                    "description": "24/7 Premium Support (Annual)",
                    "quantity": 1,
                    "unit_price": round(support_cost, 2),
                    "total": round(support_cost, 2)
                })
                additional_cost += support_cost
            elif "custom integration" in req:
                integration_cost = 15000  # Fixed cost for custom integration
                items.append({
                    "description": "Custom Integration Services",
                    "quantity": 1,
                    "unit_price": integration_cost,
                    "total": integration_cost
                })
                additional_cost += integration_cost
        
        total_price = final_price + additional_cost
        
        return Quote(
            quote_id=f"RFQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            items=items,
            total_price=round(total_price, 2),
            delivery_terms="Standard delivery within 30 business days",
            validity_period="Valid for 30 days from quote date",
            special_conditions=[
                "Payment terms: Net 30",
                "Includes standard warranty",
                "Training materials included"
            ] + requirements.special_requirements
        )
    
    def _determine_question_next_steps(self, questions: List[ClarifyingQuestion]) -> List[str]:
        """Determine next steps when asking clarifying questions."""
        return [
            "Wait for customer responses to clarifying questions",
            "Analyze customer answers to update requirements",
            "Proceed with quote generation once sufficient information is gathered",
            "Follow up if customer doesn't respond within reasonable timeframe"
        ]
    
    def _determine_quote_next_steps(self, state: ConversationState) -> List[str]:
        """Determine next steps after providing a quote."""
        return [
            "Follow up on customer's quote review",
            "Answer any questions about the proposal",
            "Negotiate terms if customer requests modifications",
            "Prepare final agreement if customer accepts",
            "Schedule implementation planning meeting"
        ] 