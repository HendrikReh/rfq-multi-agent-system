"""
Complete RFQ Flow Demo

This demo showcases the complete interactive RFQ processing workflow including:
1. Customer initiates inquiry
2. System asks clarifying questions
3. Customer responds to questions
4. System generates quote
5. Customer responds to quote

The demo uses the CustomerResponseAgent to simulate realistic customer interactions,
demonstrating the full conversational flow from initial inquiry to quote acceptance.
"""

import asyncio
from datetime import datetime

from agents import (
    ConversationState,
    CustomerResponseAgent,
    RFQDependencies,
    RFQOrchestrator,
    ScenarioRecorder,
)
from agents.utils import get_all_agent_models


class RFQFlowSimulator:
    """Simulates complete RFQ flows with realistic customer interactions."""
    
    def __init__(self):
        self.orchestrator = RFQOrchestrator()
        self.customer_agent = CustomerResponseAgent()
        self.recorder = ScenarioRecorder()
    
    async def simulate_complete_flow(
        self,
        scenario_id: int,
        scenario_name: str,
        customer_persona: str,
        initial_inquiry_type: str,
        business_context: str = "",
        quote_response_type: str = "interested"
    ):
        """
        Simulate a complete RFQ flow from initial inquiry to quote response.
        
        Args:
            scenario_id: Unique identifier for this scenario
            scenario_name: Human-readable name for the scenario
            customer_persona: Description of the customer type
            initial_inquiry_type: Type of initial inquiry (general, urgent, budget_conscious, detailed)
            business_context: Additional business context for the customer
            quote_response_type: How customer responds to quote (interested, negotiating, accepting, declining)
        """
        print(f"🎭 CUSTOMER PERSONA: {customer_persona}")
        print(f"📋 BUSINESS CONTEXT: {business_context}")
        print("=" * 80)
        
        # Step 1: Generate initial customer inquiry
        print("📧 STEP 1: Customer Initial Inquiry")
        print("-" * 40)
        
        initial_inquiry = await self.customer_agent.generate_follow_up_inquiry(
            customer_persona=customer_persona,
            inquiry_type=initial_inquiry_type
        )
        
        print(f"Customer: \"{initial_inquiry}\"")
        print()
        
        # Step 2: System processes initial inquiry
        print("🤖 STEP 2: System Analysis & Response")
        print("-" * 40)
        
        deps = RFQDependencies(
            customer_id="demo_customer",
            session_id="demo_session",
            conversation_history=[initial_inquiry],
            current_state=ConversationState.INITIAL
        )
        
        result = await self.orchestrator.process_rfq(initial_inquiry, deps)
        
        print(f"Decision: {'Ask Questions' if result.interaction_decision.should_ask_questions else 'Generate Quote'}")
        print(f"Confidence: {result.interaction_decision.confidence_level}/5")
        print(f"Reasoning: {result.interaction_decision.reasoning}")
        print()
        print(f"System: \"{result.message_to_customer}\"")
        print()
        
        # Initialize tracking variables for recording
        customer_responses = []
        final_result = None
        quote_response = None
        
        # Step 3: Handle system response
        if result.clarifying_questions:
            # Customer responds to clarifying questions
            print("📧 STEP 3: Customer Responds to Questions")
            print("-" * 40)
            
            customer_response = await self.customer_agent.respond_to_questions(
                questions=result.clarifying_questions,
                customer_intent=result.customer_intent,
                customer_persona=customer_persona,
                business_context=business_context
            )
            
            print(f"Customer: \"{customer_response}\"")
            print()
            
            # Track customer response
            customer_responses.append(customer_response)
            
            # Step 4: System processes customer response and generates quote
            print("🤖 STEP 4: System Processes Response & Generates Quote")
            print("-" * 40)
            
            updated_deps = RFQDependencies(
                customer_id="demo_customer",
                session_id="demo_session",
                conversation_history=[initial_inquiry, customer_response],
                current_state=ConversationState.REQUIREMENTS_GATHERING
            )
            
            final_result = await self.orchestrator.process_rfq(customer_response, updated_deps)
            
            print(f"Decision: {'Ask More Questions' if final_result.interaction_decision.should_ask_questions else 'Generate Quote'}")
            print(f"Confidence: {final_result.interaction_decision.confidence_level}/5")
            print()
            
            if final_result.quote:
                print(f"System: \"{final_result.message_to_customer}\"")
                print()
                
                # Step 5: Customer responds to quote
                print("📧 STEP 5: Customer Responds to Quote")
                print("-" * 40)
                
                quote_response = await self.customer_agent.respond_to_quote(
                    quote_message=final_result.message_to_customer,
                    customer_intent=final_result.customer_intent,
                    customer_persona=customer_persona,
                    response_type=quote_response_type
                )
                
                print(f"Customer: \"{quote_response}\"")
                print()
                
                # Display final metrics
                if final_result.performance:
                    print("📊 FINAL METRICS")
                    print("-" * 40)
                    print(f"Response Time: {final_result.performance.response_time:.2f}s")
                    print(f"Accuracy Score: {final_result.performance.accuracy_score:.2f}")
                    print(f"Satisfaction Prediction: {final_result.performance.customer_satisfaction_prediction:.2f}")
            
        elif result.quote:
            # Direct quote generated - customer responds
            print("📧 STEP 3: Customer Responds to Immediate Quote")
            print("-" * 40)
            
            quote_response = await self.customer_agent.respond_to_quote(
                quote_message=result.message_to_customer,
                customer_intent=result.customer_intent,
                customer_persona=customer_persona,
                response_type=quote_response_type
            )
            
            print(f"Customer: \"{quote_response}\"")
            print()
        
        # Record the complete scenario
        try:
            # Collect current agent model configuration
            agent_models = get_all_agent_models()
            
            filepath = self.recorder.record_scenario(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                customer_persona=customer_persona,
                business_context=business_context,
                initial_inquiry=initial_inquiry,
                initial_result=result,
                customer_responses=customer_responses,
                final_result=final_result,
                quote_response=quote_response,
                agent_models=agent_models,
                metadata={
                    "initial_inquiry_type": initial_inquiry_type,
                    "quote_response_type": quote_response_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"📁 Scenario recorded: {filepath}")
        except Exception as e:
            print(f"❌ Failed to record scenario: {e}")


async def demo_complete_rfq_flows():
    """Demonstrate multiple complete RFQ flows with different customer types."""
    
    print("🚀 COMPLETE RFQ FLOW SIMULATION")
    print("Demonstrating end-to-end customer interactions with realistic responses")
    print("=" * 80)
    print()
    
    simulator = RFQFlowSimulator()
    
    # Define different customer scenarios
    scenarios = [
        {
            "name": "Startup CTO - Budget Conscious",
            "persona": "CTO of a 50-person startup, technically savvy, budget-constrained, needs to justify costs to board",
            "inquiry_type": "budget_conscious",
            "business_context": "Fast-growing SaaS startup, recently raised Series A, need to scale team collaboration tools",
            "quote_response": "negotiating"
        },
        {
            "name": "Enterprise IT Director - Urgent Need",
            "persona": "IT Director at Fortune 500 company, experienced buyer, has approval authority, values quality and support",
            "inquiry_type": "urgent",
            "business_context": "Large enterprise expanding internationally, existing system failing, board pressure for quick resolution",
            "quote_response": "accepting"
        },
        {
            "name": "SMB Owner - Exploring Options",
            "persona": "Small business owner, not very technical, price-sensitive, careful decision maker",
            "inquiry_type": "general",
            "business_context": "25-person marketing agency, first time buying enterprise software, wants to understand options",
            "quote_response": "interested"
        },
        {
            "name": "Corporate Procurement - Detailed RFP",
            "persona": "Corporate procurement specialist, very thorough, follows formal process, risk-averse",
            "inquiry_type": "detailed",
            "business_context": "Large corporation with formal procurement process, multiple stakeholders, compliance requirements",
            "quote_response": "declining"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🎯 SCENARIO {i}: {scenario['name']}")
        print("=" * 80)
        
        try:
            await simulator.simulate_complete_flow(
                scenario_id=i,
                scenario_name=scenario["name"],
                customer_persona=scenario["persona"],
                initial_inquiry_type=scenario["inquiry_type"],
                business_context=scenario["business_context"],
                quote_response_type=scenario["quote_response"]
            )
            
        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")
            # Record error scenario
            try:
                # Collect current agent model configuration for error scenarios too
                agent_models = get_all_agent_models()
                
                error_filepath = simulator.recorder.record_error_scenario(
                    scenario_id=i,
                    scenario_name=scenario["name"],
                    customer_persona=scenario["persona"],
                    business_context=scenario["business_context"],
                    initial_inquiry="Error occurred before initial inquiry",
                    error=e,
                    agent_models=agent_models,
                    metadata={
                        "initial_inquiry_type": scenario["inquiry_type"],
                        "quote_response_type": scenario["quote_response"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                print(f"📁 Error scenario recorded: {error_filepath}")
            except Exception as record_error:
                print(f"❌ Failed to record error scenario: {record_error}")
        
        print("\n" + "="*80 + "\n")
    
    print("✨ COMPLETE RFQ FLOW SIMULATION FINISHED")
    print("All scenarios demonstrate the intelligent, adaptive nature of the multi-agent system")


async def demo_question_refinement_flow():
    """Demonstrate how the system refines questions based on customer responses."""
    
    print("🔄 QUESTION REFINEMENT FLOW DEMO")
    print("Showing how the system adapts questions based on customer responses")
    print("=" * 80)
    print()
    
    simulator = RFQFlowSimulator()
    
    # Simulate a customer who provides partial answers, requiring follow-up
    customer_persona = "Mid-level manager at tech company, somewhat technical, moderate urgency"
    business_context = "Growing tech company, need better project management tools"
    
    print(f"🎭 CUSTOMER: {customer_persona}")
    print(f"📋 CONTEXT: {business_context}")
    print("=" * 80)
    
    # Initial vague inquiry
    initial_inquiry = "We need some project management software for our team. Can you help?"
    
    print("📧 INITIAL INQUIRY")
    print(f"Customer: \"{initial_inquiry}\"")
    print()
    
    # Process initial inquiry
    deps = RFQDependencies(
        customer_id="refinement_demo",
        session_id="refinement_session",
        conversation_history=[initial_inquiry],
        current_state=ConversationState.INITIAL
    )
    
    result = await simulator.orchestrator.process_rfq(initial_inquiry, deps)
    
    print("🤖 SYSTEM ASKS CLARIFYING QUESTIONS")
    print(f"System: \"{result.message_to_customer}\"")
    print()
    
    if result.clarifying_questions:
        # Customer provides partial response
        print("📧 CUSTOMER PROVIDES PARTIAL ANSWERS")
        partial_response = """Thanks for the questions! Here's what I can tell you:

1. We need project management software for about 30 people
2. Our budget is around $15,000 annually
3. We need it within the next 2 months
4. Not sure about integrations yet - would need to check with IT
5. I can make recommendations but final approval needs to go through my director

Is this enough to get started with a quote?"""
        
        print(f"Customer: \"{partial_response}\"")
        print()
        
        # Process partial response
        updated_deps = RFQDependencies(
            customer_id="refinement_demo",
            session_id="refinement_session",
            conversation_history=[initial_inquiry, partial_response],
            current_state=ConversationState.REQUIREMENTS_GATHERING
        )
        
        final_result = await simulator.orchestrator.process_rfq(partial_response, updated_deps)
        
        print("🤖 SYSTEM DECIDES NEXT STEP")
        print(f"Decision: {'Ask Follow-up Questions' if final_result.interaction_decision.should_ask_questions else 'Generate Quote'}")
        print(f"Confidence: {final_result.interaction_decision.confidence_level}/5")
        print(f"Reasoning: {final_result.interaction_decision.reasoning}")
        print()
        print(f"System: \"{final_result.message_to_customer}\"")


async def main():
    """Run all demo scenarios."""
    
    # Main complete flow demo
    await demo_complete_rfq_flows()
    
    print("\n" + "="*100 + "\n")
    
    # Question refinement demo
    await demo_question_refinement_flow()


if __name__ == "__main__":
    asyncio.run(main()) 