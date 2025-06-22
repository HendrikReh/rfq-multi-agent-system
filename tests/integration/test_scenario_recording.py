#!/usr/bin/env python3
"""
Test Scenario Recording

Demonstrates the scenario recording functionality with mock data,
showing how scenarios are saved to JSON files with the specified naming pattern.
"""

from datetime import datetime

from agents import (
    ClarifyingQuestion,
    ConversationState,
    CustomerIntent,
    CustomerSentiment,
    InteractionDecision,
    PricingStrategy,
    Quote,
    RequirementsCompleteness,
    RFQProcessingResult,
    RFQRequirements,
    ScenarioRecorder,
    SystemPerformance,
)
from agents.utils import get_all_agent_models


def create_mock_scenario_data(scenario_id: int) -> dict:
    """Create mock scenario data for testing."""
    
    scenarios = [
        {
            "name": "Test Startup CTO - Budget Conscious",
            "persona": "CTO of a 50-person startup, technically savvy, budget-constrained",
            "business_context": "Fast-growing SaaS startup, recently raised Series A",
            "initial_inquiry": "We need project management software for our growing team. Budget is tight but we need something reliable.",
            "customer_responses": [
                "Thanks for the questions! We have about 45 people, budget around $25k annually, need it within 8 weeks."
            ],
            "quote_response": "This looks reasonable. Can we negotiate on the support package pricing?"
        },
        {
            "name": "Test Enterprise IT Director - Urgent",
            "persona": "IT Director at Fortune 500 company, experienced buyer, has approval authority",
            "business_context": "Large enterprise expanding internationally, existing system failing",
            "initial_inquiry": "URGENT: Our current project management system is failing. Need 200 licenses ASAP with enterprise support.",
            "customer_responses": [],
            "quote_response": "Perfect! This meets our requirements. Please proceed with the implementation."
        },
        {
            "name": "Test SMB Owner - Exploring",
            "persona": "Small business owner, not very technical, price-sensitive",
            "business_context": "25-person marketing agency, first time buying enterprise software",
            "initial_inquiry": "Hi, I'm looking into project management tools for my small agency. Not sure what we need exactly.",
            "customer_responses": [
                "We have 25 people, mostly remote. Budget is around $10k per year. Need something simple to use."
            ],
            "quote_response": "This is helpful but seems a bit expensive. Do you have any smaller packages?"
        }
    ]
    
    # Use modulo to cycle through scenarios if more IDs than scenarios
    scenario = scenarios[(scenario_id - 1) % len(scenarios)]
    
    return scenario


def create_mock_rfq_result(scenario_data: dict, scenario_id: int) -> RFQProcessingResult:
    """Create a mock RFQ processing result."""
    
    # Create mock requirements
    requirements = RFQRequirements(
        product_type="Project Management Software",
        quantity=50 + (scenario_id * 25),  # Vary quantity by scenario
        completeness=RequirementsCompleteness.PARTIAL if scenario_data["customer_responses"] else RequirementsCompleteness.COMPLETE,
        missing_info=["Integration requirements", "User roles"] if scenario_data["customer_responses"] else []
    )
    
    # Create mock customer intent
    customer_intent = CustomerIntent(
        primary_intent="purchase_evaluation",
        sentiment=CustomerSentiment.POSITIVE,
        urgency_level=5 if "URGENT" in scenario_data["initial_inquiry"] else 3,
        price_sensitivity=4 if "budget" in scenario_data["initial_inquiry"].lower() else 2,
        readiness_to_buy=4 if "URGENT" in scenario_data["initial_inquiry"] else 3,
        decision_factors=["Price", "Features", "Support", "Timeline"]
    )
    
    # Create mock interaction decision
    should_ask_questions = bool(scenario_data["customer_responses"])
    interaction_decision = InteractionDecision(
        should_ask_questions=should_ask_questions,
        should_generate_quote=not should_ask_questions,
        next_action="generate_quote" if not should_ask_questions else "ask_questions",
        confidence_level=4 if not should_ask_questions else 3,
        reasoning="Customer provided sufficient information for quote generation" if not should_ask_questions else "Need more details about requirements"
    )
    
    # Create mock clarifying questions
    clarifying_questions = []
    if should_ask_questions:
        clarifying_questions = [
            ClarifyingQuestion(
                question="How many team members will be using the software?",
                category="requirements",
                priority=5,
                expected_response_type="number",
                reasoning="Need to determine license count for accurate pricing"
            ),
            ClarifyingQuestion(
                question="What's your approximate annual budget for this solution?",
                category="budget",
                priority=4,
                expected_response_type="currency",
                reasoning="Understanding budget helps tailor the proposal"
            ),
            ClarifyingQuestion(
                question="What's your preferred timeline for implementation?",
                category="timeline",
                priority=3,
                expected_response_type="date",
                reasoning="Timeline affects pricing and implementation approach"
            )
        ]
    
    # Create mock quote
    base_price = 500 * requirements.quantity
    quote = Quote(
        quote_id=f"Q-{datetime.now().strftime('%Y%m%d')}-{scenario_id:03d}",
        items=[
            {
                "description": "Enterprise Software Licenses",
                "quantity": requirements.quantity,
                "unit_price": 500,
                "total": base_price
            },
            {
                "description": "Professional Support (Annual)",
                "quantity": 1,
                "unit_price": base_price * 0.2,
                "total": base_price * 0.2
            },
            {
                "description": "Implementation Services",
                "quantity": 1,
                "unit_price": 5000,
                "total": 5000
            }
        ],
        total_price=base_price * 1.2 + 5000,
        validity_period="30 days",
        delivery_terms="Implementation within 4-6 weeks of contract signing"
    )
    
    # Create mock pricing strategy
    pricing_strategy = PricingStrategy(
        strategy_type="value_based",
        base_price=base_price,
        adjustments=[
            {"type": "volume_discount", "amount": -base_price * 0.05, "reason": "Volume discount for enterprise purchase"},
            {"type": "urgency_premium", "amount": base_price * 0.1 if customer_intent.urgency_level >= 4 else 0, "reason": "Expedited delivery premium"}
        ],
        justification=f"Pricing based on {requirements.quantity} licenses with enterprise support and professional services"
    )
    
    # Create mock performance metrics
    performance = SystemPerformance(
        response_time=1.5 + (scenario_id * 0.2),  # Vary response time
        accuracy_score=0.85 + (scenario_id * 0.05),
        customer_satisfaction_prediction=0.8 + (scenario_id * 0.03),
        improvement_suggestions=[
            "Consider gathering more detailed integration requirements",
            "Implement automated budget qualification",
            "Add industry-specific pricing models"
        ]
    )
    
    # Create the complete result
    result = RFQProcessingResult(
        status="completed",
        conversation_state=ConversationState.QUOTE_GENERATION.value if not should_ask_questions else ConversationState.REQUIREMENTS_GATHERING.value,
        requirements=requirements,
        customer_intent=customer_intent,
        interaction_decision=interaction_decision,
        clarifying_questions=clarifying_questions if should_ask_questions else [],
        pricing_strategy=pricing_strategy,
        quote=quote,
        performance=performance,
        next_steps=["Review quote details", "Schedule implementation call"] if not should_ask_questions else ["Provide additional information", "Schedule follow-up call"],
        message_to_customer=f"Thank you for your inquiry! {'Based on your requirements, here is our quote:' if not should_ask_questions else 'To provide an accurate quote, I need some additional information:'}"
    )
    
    return result


def test_scenario_recording():
    """Test the scenario recording functionality."""
    
    print("🧪 TESTING SCENARIO RECORDING")
    print("=" * 80)
    print("Creating mock scenarios and recording them to JSON files...")
    print()
    
    recorder = ScenarioRecorder()
    
    # Test 3 different scenarios
    for scenario_id in range(1, 4):
        print(f"📝 Creating test scenario {scenario_id}...")
        
        scenario_data = create_mock_scenario_data(scenario_id)
        initial_result = create_mock_rfq_result(scenario_data, scenario_id)
        
        # Create final result if there were customer responses
        final_result = None
        if scenario_data["customer_responses"]:
            # Modify the result to show quote generation after customer response
            final_scenario_data = scenario_data.copy()
            final_scenario_data["customer_responses"] = []  # No more questions needed
            final_result = create_mock_rfq_result(final_scenario_data, scenario_id)
            final_result.conversation_state = ConversationState.QUOTE_GENERATION.value
            final_result.interaction_decision.should_ask_questions = False
            final_result.interaction_decision.should_generate_quote = True
            final_result.interaction_decision.next_action = "generate_quote"
            final_result.interaction_decision.confidence_level = 5
            final_result.clarifying_questions = []
        
        try:
            # Get current agent model configuration
            agent_models = get_all_agent_models()
            
            filepath = recorder.record_scenario(
                scenario_id=scenario_id,
                scenario_name=scenario_data["name"],
                customer_persona=scenario_data["persona"],
                business_context=scenario_data["business_context"],
                initial_inquiry=scenario_data["initial_inquiry"],
                initial_result=initial_result,
                customer_responses=scenario_data["customer_responses"],
                final_result=final_result,
                quote_response=scenario_data["quote_response"],
                agent_models=agent_models,
                metadata={
                    "test_run": True,
                    "mock_data": True,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            print(f"✅ Scenario {scenario_id} recorded: {filepath}")
            
        except Exception as e:
            print(f"❌ Error recording scenario {scenario_id}: {e}")
    
    print()
    print("🎯 TESTING ERROR SCENARIO RECORDING")
    print("-" * 40)
    
    # Test error scenario recording
    try:
        # Get current agent model configuration for error scenario too
        agent_models = get_all_agent_models()
        
        test_error = ValueError("This is a test error for demonstration")
        error_filepath = recorder.record_error_scenario(
            scenario_id=99,
            scenario_name="Test Error Scenario",
            customer_persona="Test customer experiencing system error",
            business_context="Testing error handling in scenario recording",
            initial_inquiry="This inquiry triggered a system error",
            error=test_error,
            agent_models=agent_models,
            metadata={
                "test_run": True,
                "error_test": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"✅ Error scenario recorded: {error_filepath}")
        
    except Exception as e:
        print(f"❌ Error recording error scenario: {e}")
    
    print()
    print("📁 LISTING RECORDED SCENARIOS")
    print("-" * 40)
    
    # List all recorded scenarios
    scenario_files = recorder.list_scenario_files()
    if scenario_files:
        for filepath in sorted(scenario_files):
            summary = recorder.get_scenario_summary(filepath)
            timestamp = datetime.fromisoformat(summary['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            print(f"• {summary['scenario_name']} (ID: {summary['scenario_id']}) - {timestamp}")
            print(f"  File: {filepath}")
            quote_value = f"${summary['total_quote_value']:,.0f}" if summary['total_quote_value'] else "N/A"
            print(f"  Quote: {'✅' if summary['quote_generated'] else '❌'} | Value: {quote_value}")
            print()
    else:
        print("No scenario files found.")
    
    print("✨ SCENARIO RECORDING TEST COMPLETED")
    print("Use 'python view_scenarios.py' to view and analyze the recorded scenarios")


if __name__ == "__main__":
    test_scenario_recording() 