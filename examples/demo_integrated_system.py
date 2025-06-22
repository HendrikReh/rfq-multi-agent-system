"""
Comprehensive Demo of Integrated RFQ Multi-Agent System

This demo showcases the complete integrated system with all agents working together:
- Enhanced orchestration patterns
- Competitive intelligence
- Risk assessment
- Contract terms generation
- Proposal writing
- Health monitoring
- Performance optimization
"""

import asyncio
from datetime import datetime

from agents import ConversationState, RFQDependencies
from agents.integration_framework import IntegratedRFQSystem


async def demo_comprehensive_system():
    """Demonstrate the complete integrated RFQ system."""
    
    print("🚀 COMPREHENSIVE RFQ MULTI-AGENT SYSTEM DEMO")
    print("=" * 80)
    print("Showcasing complete LLM augmentation with 13+ specialized agents")
    print("=" * 80)
    print()
    
    # Initialize the integrated system
    system = IntegratedRFQSystem()
    
    # Demo scenarios showcasing different system capabilities
    scenarios = [
        {
            "name": "Enterprise Software RFQ - High Stakes",
            "message": "We need enterprise project management software for 500 users across 5 global offices. This is critical for our Q1 digital transformation initiative. Budget is $2M annually. Need deployment by March 1st with 24/7 support, SSO integration, and compliance with SOX requirements.",
            "description": "High-value, complex enterprise deal requiring full analysis",
            "execution_mode": "parallel",
            "include_all": True
        },
        {
            "name": "Startup MVP Development - Competitive Market",
            "message": "Looking for MVP development for our fintech startup. Need a mobile app with basic payment processing. We're comparing multiple vendors and have limited budget (~$50k). Timeline is flexible but prefer to launch within 4 months.",
            "description": "Competitive scenario requiring strategic positioning",
            "execution_mode": "sequential",
            "include_all": True
        },
        {
            "name": "Government Contract - High Risk",
            "message": "Federal agency seeking cybersecurity consulting services. Must have top secret clearance, FISMA compliance, and 5+ years government experience. Contract value $5M over 3 years. Strict procurement process and multiple competitors expected.",
            "description": "High-risk, regulated environment with complex requirements",
            "execution_mode": "parallel",
            "include_all": True
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 SCENARIO {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Execution Mode: {scenario['execution_mode'].title()}")
        print("-" * 60)
        print(f"Customer Message: \"{scenario['message']}\"")
        print()
        
        # Create dependencies
        deps = RFQDependencies(
            customer_id=f"enterprise_customer_{i}",
            session_id=f"session_{i}",
            conversation_history=[scenario['message']],
            current_state=ConversationState.INITIAL
        )
        
        # Process with comprehensive analysis
        start_time = datetime.now()
        
        try:
            result = await system.process_rfq_comprehensive(
                customer_message=scenario['message'],
                deps=deps,
                include_competitive_analysis=scenario['include_all'],
                include_risk_assessment=scenario['include_all'],
                include_contract_terms=scenario['include_all'],
                include_proposal=scenario['include_all'],
                execution_mode=scenario['execution_mode']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display comprehensive results
            print("🤖 COMPREHENSIVE SYSTEM ANALYSIS")
            print("-" * 40)
            
            # Basic RFQ Analysis
            basic = result.basic_result
            print(f"Requirements Completeness: {basic.requirements.completeness.value}")
            print(f"Customer Urgency: {basic.customer_intent.urgency_level}/5")
            print(f"Price Sensitivity: {basic.customer_intent.price_sensitivity}/5")
            print(f"Decision: {'Ask Questions' if basic.interaction_decision.should_ask_questions else 'Generate Quote'}")
            print(f"Base Confidence: {basic.interaction_decision.confidence_level}/5")
            print()
            
            # Competitive Intelligence
            if result.competitive_analysis:
                comp = result.competitive_analysis
                print("🎯 COMPETITIVE INTELLIGENCE")
                print(f"Market Position: {comp.market_position}")
                print(f"Win Probability: {comp.win_probability:.1%}")
                print(f"Strategy: {comp.recommended_strategy}")
                print(f"Key Advantages: {', '.join(comp.competitive_advantages[:3])}")
                print()
            
            # Risk Assessment
            if result.risk_assessment:
                risk = result.risk_assessment
                print("⚠️ RISK ASSESSMENT")
                print(f"Overall Risk Level: {risk.overall_risk_level}")
                print(f"Risk Score: {risk.risk_score}/10")
                print(f"Recommendation: {risk.recommendation}")
                print(f"Key Risks: {', '.join(risk.financial_risks[:2] + risk.operational_risks[:2])}")
                print()
            
            # Contract Terms
            if result.contract_terms:
                contract = result.contract_terms
                print("📄 CONTRACT TERMS")
                print(f"Payment Terms: {contract.payment_terms}")
                print(f"Delivery Terms: {contract.delivery_terms}")
                print(f"Warranty: {contract.warranty_terms}")
                print(f"Key Clauses: {len(contract.recommended_clauses)} recommended")
                print()
            
            # Proposal Document
            if result.proposal_document:
                proposal = result.proposal_document
                print("📝 PROPOSAL DOCUMENT")
                print(f"Executive Summary: {proposal.executive_summary[:100]}...")
                print(f"Technical Approach: {proposal.technical_approach[:100]}...")
                print(f"Sections Generated: {len([x for x in [proposal.executive_summary, proposal.technical_approach, proposal.pricing_section] if x])}")
                print()
            
            # System Metrics
            print("📊 SYSTEM PERFORMANCE")
            print(f"Total Processing Time: {processing_time:.2f}s")
            print(f"Agents Executed: {result.processing_metrics['agents_executed']}")
            print(f"Overall Confidence: {result.confidence_score:.1%}")
            print(f"Execution Mode: {result.processing_metrics['execution_mode']}")
            print()
            
            # Agent Contributions
            print("🤝 AGENT CONTRIBUTIONS")
            for role, agent in result.agent_contributions.items():
                status = "✅" if "Not executed" not in agent else "⏭️"
                print(f"{status} {role.replace('_', ' ').title()}: {agent}")
            print()
            
            # System Recommendation
            print("💡 SYSTEM RECOMMENDATION")
            print(f"\"{result.recommendation}\"")
            print()
            
        except Exception as e:
            print(f"❌ Error processing scenario: {e}")
        
        print("=" * 80)
        print()
    
    # System Health and Performance Analysis
    print("🏥 SYSTEM HEALTH ANALYSIS")
    print("-" * 40)
    
    health_report = await system.get_system_health_report()
    
    print(f"Overall System Status: {health_report.overall_status}")
    print(f"Total Agents: {health_report.total_agents}")
    print(f"Healthy: {health_report.healthy_agents} | Degraded: {health_report.degraded_agents} | Failed: {health_report.failed_agents}")
    print(f"Average Response Time: {health_report.average_response_time:.2f}s")
    print(f"System Uptime: {health_report.system_uptime:.2f} hours")
    print()
    
    if health_report.agent_statuses:
        print("Agent Performance:")
        for status in health_report.agent_statuses:
            health_icon = "🟢" if status.status == "HEALTHY" else "🟡" if status.status == "DEGRADED" else "🔴"
            print(f"{health_icon} {status.agent_name}: {status.response_time:.2f}s (Score: {status.performance_score:.2f})")
    
    if health_report.recommendations:
        print("\nRecommendations:")
        for rec in health_report.recommendations:
            print(f"• {rec}")
    
    print()
    
    # Performance Optimization Analysis
    print("🔧 PERFORMANCE OPTIMIZATION")
    print("-" * 40)
    
    optimization = await system.optimize_system_performance()
    
    print("Key Metrics:")
    for metric, value in optimization["key_metrics"].items():
        print(f"• {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\nOptimization Analysis:")
    print(f"\"{optimization['analysis']}\"")
    print()


async def demo_agent_delegation_patterns():
    """Demonstrate advanced PydanticAI agent delegation patterns."""
    
    print("🔗 AGENT DELEGATION PATTERNS DEMO")
    print("=" * 60)
    print("Showcasing PydanticAI multi-agent patterns:")
    print("1. Agent delegation via tools")
    print("2. Parallel agent execution")
    print("3. Graph-based control flow")
    print("4. Memory-enhanced processing")
    print("=" * 60)
    print()
    
    system = IntegratedRFQSystem()
    
    # Test enhanced orchestrator with delegation
    print("🎯 Enhanced Orchestrator with Agent Delegation")
    print("-" * 50)
    
    enhanced_message = "Need cloud infrastructure for AI workloads. High performance computing requirements."
    
    deps = RFQDependencies(
        customer_id="tech_customer",
        session_id="delegation_demo",
        conversation_history=[enhanced_message],
        current_state=ConversationState.INITIAL
    )
    
    try:
        # This will use the enhanced orchestrator with tool delegation
        result = await system.enhanced_orchestrator.process_rfq_enhanced(
            enhanced_message, deps
        )
        
        print(f"✅ Enhanced orchestration completed")
        print(f"Status: {result.status}")
        print(f"Decision: {result.interaction_decision.reasoning}")
        print()
        
    except Exception as e:
        print(f"❌ Enhanced orchestration error: {e}")
    
    # Test graph-based control flow
    print("🕸️ Graph-Based State Machine")
    print("-" * 50)
    
    try:
        from agents.enhanced_orchestrator import AgentCoordinationContext
        
        context = AgentCoordinationContext(
            session_id="graph_demo",
            customer_profile={},
            conversation_history=[enhanced_message],
            processing_stage="competitive_analysis"
        )
        
        next_state, response = await system.graph_controller.process_state_transition(
            "competitive_analysis", context, "We're comparing multiple cloud providers"
        )
        
        print(f"✅ State transition: competitive_analysis → {next_state}")
        print(f"Response: {response[:100]}...")
        print()
        
    except Exception as e:
        print(f"❌ Graph control error: {e}")
    
    # Test memory-enhanced processing
    print("🧠 Memory-Enhanced Agent Processing")
    print("-" * 50)
    
    try:
        memory_response = await system.memory_agent.process_with_memory(
            customer_id="repeat_customer",
            message="Looking for another cloud solution, similar to our previous project",
            context={"previous_projects": ["AWS migration", "Azure deployment"]}
        )
        
        print(f"✅ Memory-enhanced processing completed")
        print(f"Response: {memory_response[:150]}...")
        print()
        
    except Exception as e:
        print(f"❌ Memory processing error: {e}")


async def main():
    """Main demo function."""
    
    print("🎭 COMPLETE RFQ MULTI-AGENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete LLM augmentation of the RFQ system")
    print("with advanced PydanticAI patterns and comprehensive agent integration.")
    print("=" * 80)
    print()
    
    # Run comprehensive system demo
    await demo_comprehensive_system()
    
    print("\n" + "="*100 + "\n")
    
    # Run agent delegation patterns demo
    await demo_agent_delegation_patterns()
    
    print("\n🎉 DEMO COMPLETED")
    print("=" * 40)
    print("The system now includes:")
    print("✅ 13+ specialized agents with distinct responsibilities")
    print("✅ Advanced PydanticAI multi-agent orchestration patterns")
    print("✅ Parallel and sequential execution modes")
    print("✅ Comprehensive risk and competitive analysis")
    print("✅ Professional proposal generation")
    print("✅ Real-time health monitoring and optimization")
    print("✅ Memory-enhanced learning capabilities")
    print("✅ Graph-based workflow control")
    print("✅ Complete end-to-end RFQ processing")
    print()
    print("🚀 Ready for production deployment with full LLM augmentation!")


if __name__ == "__main__":
    asyncio.run(main()) 