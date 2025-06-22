"""
RFQ Orchestrator Demo

This demo showcases the intelligent multi-agent RFQ processing system that:
1. Analyzes customer requirements and assesses completeness
2. Makes strategic decisions about when to ask clarifying questions
3. Generates targeted questions when more information is needed
4. Provides accurate quotes when sufficient information is available

Run with different modes:
- python main.py : Basic interactive demo showing system decisions
- python main.py --complete : Complete flow simulation with customer responses
"""

import argparse
import asyncio

from agents import ConversationState, RFQDependencies, RFQOrchestrator
from agents.utils import print_model_configuration


async def demo_interactive_rfq_processing():
    """Demonstrate the interactive RFQ processing workflow."""
    
    print("=== RFQ Orchestrator Interactive Demo ===\n")
    
    # Show current model configuration
    print_model_configuration()
    print("\n" + "="*80 + "\n")
    
    # Initialize the orchestrator
    orchestrator = RFQOrchestrator()
    
    # Test scenarios with different levels of information completeness
    test_scenarios = [
        {
            "name": "Vague Initial Request",
            "message": "Hi, I need some software for my business. Can you help?",
            "description": "Very limited information - should trigger clarifying questions"
        },
        {
            "name": "Partial Information Request", 
            "message": "We need 50 enterprise software licenses for our team. We're looking at a budget around $30k and need it deployed by end of Q1. What can you offer?",
            "description": "Some key details provided - may still need clarification"
        },
        {
            "name": "Detailed Urgent Request",
            "message": "URGENT: Need 100 enterprise software licenses ASAP for our expanding sales team. Budget is $75k max. Must include 24/7 support and custom integration with our CRM. Timeline is critical - need deployment within 2 weeks. I have approval authority to move forward immediately.",
            "description": "Comprehensive information with high urgency - should generate quote"
        },
        {
            "name": "Price-Sensitive Request",
            "message": "Looking for the most cost-effective solution for 25 software licenses. We're a startup with limited budget but growing fast. Need basic features and good support. Comparing multiple vendors - price is our main concern.",
            "description": "Price-focused customer - should ask budget clarification questions"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"--- Scenario {i}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Customer Message: \"{scenario['message']}\"\n")
        
        # Create dependencies for this scenario
        deps = RFQDependencies(
            customer_id=f"customer_{i}",
            session_id=f"session_{i}",
            conversation_history=[scenario['message']],
            current_state=ConversationState.INITIAL
        )
        
        try:
            # Process the RFQ
            result = await orchestrator.process_rfq(scenario['message'], deps)
            
            # Display results
            print("🤖 SYSTEM ANALYSIS:")
            print(f"Status: {result.status}")
            print(f"Requirements Completeness: {result.requirements.completeness.value}")
            print(f"Customer Urgency: {result.customer_intent.urgency_level}/5")
            print(f"Price Sensitivity: {result.customer_intent.price_sensitivity}/5")
            print(f"Buying Readiness: {result.customer_intent.readiness_to_buy}/5")
            print(f"Decision: {'Ask Questions' if result.interaction_decision.should_ask_questions else 'Generate Quote'}")
            print(f"Confidence Level: {result.interaction_decision.confidence_level}/5")
            print(f"Reasoning: {result.interaction_decision.reasoning}\n")
            
            if result.clarifying_questions:
                print("❓ CLARIFYING QUESTIONS GENERATED:")
                for j, question in enumerate(result.clarifying_questions, 1):
                    print(f"{j}. {question.question}")
                    print(f"   Category: {question.category} | Priority: {question.priority}/5")
                    print(f"   Reasoning: {question.reasoning}\n")
            
            if result.quote:
                print("💰 QUOTE GENERATED:")
                print(f"Quote ID: {result.quote.quote_id}")
                print(f"Total Price: ${result.quote.total_price:,.2f}")
                print(f"Strategy: {result.pricing_strategy.strategy_type}")
                print(f"Justification: {result.pricing_strategy.justification}\n")
            
            print("📧 MESSAGE TO CUSTOMER:")
            print(f"\"{result.message_to_customer}\"\n")
            
            print("📋 NEXT STEPS:")
            for step in result.next_steps:
                print(f"• {step}")
            
            if result.performance:
                print(f"\n📊 PERFORMANCE METRICS:")
                print(f"Response Time: {result.performance.response_time:.2f}s")
                print(f"Accuracy Score: {result.performance.accuracy_score:.2f}")
                print(f"Satisfaction Prediction: {result.performance.customer_satisfaction_prediction:.2f}")
        
        except Exception as e:
            print(f"❌ Error processing scenario: {e}")
        
        print("\n" + "="*80 + "\n")
    
    # Demonstrate follow-up scenario
    print("--- Follow-up Scenario: Customer Answers Questions ---")
    print("Simulating customer responding to clarifying questions...\n")
    
    follow_up_message = """Thanks for the questions! Here are the details:
    
    1. We need project management software for a team of 50 people
    2. Budget is around $40,000 annually 
    3. Need deployment within 6 weeks
    4. Must integrate with Slack and Google Workspace
    5. I'm the IT Director and can make the final decision
    
    Please send me a detailed quote."""
    
    deps = RFQDependencies(
        customer_id="customer_followup",
        session_id="session_followup", 
        conversation_history=[
            "Hi, I need some software for my business. Can you help?",
            follow_up_message
        ],
        current_state=ConversationState.REQUIREMENTS_GATHERING
    )
    
    try:
        result = await orchestrator.process_rfq(follow_up_message, deps)
        
        print("🤖 SYSTEM ANALYSIS:")
        print(f"Status: {result.status}")
        print(f"Requirements Completeness: {result.requirements.completeness.value}")
        print(f"Decision: {'Ask Questions' if result.interaction_decision.should_ask_questions else 'Generate Quote'}")
        print(f"Confidence Level: {result.interaction_decision.confidence_level}/5\n")
        
        if result.quote:
            print("💰 FINAL QUOTE GENERATED:")
            print(f"Quote ID: {result.quote.quote_id}")
            print("Items:")
            for item in result.quote.items:
                print(f"• {item['description']}: {item['quantity']} × ${item['unit_price']:,.2f} = ${item['total']:,.2f}")
            print(f"\nTotal Investment: ${result.quote.total_price:,.2f}")
            print(f"Delivery: {result.quote.delivery_terms}")
            print(f"Valid Until: {result.quote.validity_period}\n")
        
        print("📧 CUSTOMER MESSAGE:")
        print(f"\"{result.message_to_customer}\"\n")
        
    except Exception as e:
        print(f"❌ Error processing follow-up: {e}")


async def run_complete_flow_demo():
    """Run the complete flow demo with customer simulation."""
    try:
        # Import here to avoid circular imports
        from demo_complete_flow import demo_complete_rfq_flows, demo_question_refinement_flow
        
        print("🚀 COMPLETE RFQ FLOW SIMULATION")
        print("=" * 80)
        print("Running complete end-to-end RFQ flows with simulated customer responses...")
        print("This demonstrates the full conversational workflow from inquiry to quote response.")
        print("=" * 80)
        print()
        
        # Run the complete flow demos
        await demo_complete_rfq_flows()
        
        print("\n" + "="*100 + "\n")
        
        # Run the question refinement demo
        await demo_question_refinement_flow()
        
    except ImportError as e:
        print(f"❌ Error: Could not import complete flow demo: {e}")
        print("Make sure demo_complete_flow.py is available and all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error running complete flow demo: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="RFQ Orchestrator Demo")
    parser.add_argument(
        "--complete", 
        action="store_true", 
        help="Run complete flow simulation with customer responses"
    )
    
    args = parser.parse_args()
    
    if args.complete:
        await run_complete_flow_demo()
    else:
        await demo_interactive_rfq_processing()


if __name__ == "__main__":
    asyncio.run(main())
