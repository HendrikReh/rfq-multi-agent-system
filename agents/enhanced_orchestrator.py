"""
Enhanced RFQ Orchestrator using Advanced PydanticAI Multi-Agent Patterns

This orchestrator implements:
1. Agent delegation - agents using other agents via tools
2. Graph-based control flow for complex scenarios
3. Parallel agent execution where appropriate
4. Advanced memory management and context sharing
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from .models import (
    ConversationState,
    CustomerIntent,
    RFQDependencies,
    RFQProcessingResult,
    RFQRequirements,
)
from .utils import get_model_name


class AgentCoordinationContext(BaseModel):
    """Shared context for agent coordination."""
    session_id: str
    customer_profile: Dict
    conversation_history: List[str]
    processing_stage: str
    accumulated_insights: Dict = {}
    parallel_results: Dict = {}


class EnhancedRFQOrchestrator:
    """
    Enhanced orchestrator using advanced PydanticAI multi-agent patterns.
    
    Features:
    - Agent delegation with tools
    - Parallel agent execution
    - Graph-based workflow control
    - Advanced context sharing
    """
    
    def __init__(self):
        # Main orchestrator agent with tool access to other agents
        model_name = get_model_name("rfq_orchestrator")
        self.agent = Agent(
            model_name,
            deps_type=RFQDependencies,
            output_type=RFQProcessingResult,
            system_prompt="""
            You are an advanced RFQ orchestrator that coordinates multiple specialized agents
            using delegation patterns and intelligent workflow control.
            
            Your capabilities:
            1. Delegate tasks to specialized agents via tools
            2. Execute agents in parallel when appropriate
            3. Make intelligent routing decisions based on context
            4. Manage shared context and memory across agents
            5. Optimize workflow based on customer urgency and complexity
            
            Use the available tools to delegate to specialist agents and coordinate
            their outputs for optimal customer experience.
            """
        )
        
        # Initialize specialist agents for delegation
        self._init_specialist_agents()
        
        # Add tools for agent delegation
        self._register_agent_tools()
    
    def _init_specialist_agents(self):
        """Initialize specialist agents for delegation."""
        from .customer_intent_agent import CustomerIntentAgent
        from .pricing_strategy_agent import PricingStrategyAgent
        from .question_generation_agent import QuestionGenerationAgent
        from .rfq_parser import RFQParser
        
        self.rfq_parser = RFQParser()
        self.intent_agent = CustomerIntentAgent()
        self.pricing_agent = PricingStrategyAgent()
        self.question_agent = QuestionGenerationAgent()
    
    def _register_agent_tools(self):
        """Register specialist agents as tools for delegation."""
        
        @self.agent.tool
        async def parse_requirements(ctx: RunContext[RFQDependencies], message: str) -> RFQRequirements:
            """Parse customer requirements from message."""
            return await self.rfq_parser.parse(message)
        
        @self.agent.tool
        async def analyze_customer_intent(ctx: RunContext[RFQDependencies], message: str) -> CustomerIntent:
            """Analyze customer intent and sentiment."""
            return await self.intent_agent.analyze(message)
        
        @self.agent.tool
        async def generate_strategic_questions(
            ctx: RunContext[RFQDependencies], 
            requirements: str, 
            intent: str
        ) -> List[str]:
            """Generate strategic clarifying questions."""
            # Convert string representations back to objects for processing
            # In a real implementation, you'd use proper serialization
            questions = await self.question_agent.generate_questions(
                requirements, intent
            )
            return [q.question for q in questions]
        
        @self.agent.tool
        async def develop_pricing_strategy(
            ctx: RunContext[RFQDependencies],
            requirements: str,
            intent: str
        ) -> str:
            """Develop intelligent pricing strategy."""
            strategy = await self.pricing_agent.develop_strategy(requirements, intent)
            return f"Strategy: {strategy.strategy_type}, Base: ${strategy.base_price}, Justification: {strategy.justification}"
        
        @self.agent.tool
        async def execute_parallel_analysis(
            ctx: RunContext[RFQDependencies],
            message: str
        ) -> Dict[str, str]:
            """Execute multiple agents in parallel for comprehensive analysis."""
            import asyncio
            
            # Execute multiple agents concurrently
            tasks = [
                self.rfq_parser.parse(message),
                self.intent_agent.analyze(message)
            ]
            
            results = await asyncio.gather(*tasks)
            requirements, intent = results
            
            return {
                "requirements_completeness": requirements.completeness.value,
                "customer_urgency": str(intent.urgency_level),
                "price_sensitivity": str(intent.price_sensitivity),
                "buying_readiness": str(intent.readiness_to_buy)
            }
    
    async def process_rfq_enhanced(
        self, 
        customer_message: str, 
        deps: RFQDependencies
    ) -> RFQProcessingResult:
        """
        Process RFQ using enhanced multi-agent coordination.
        
        This method demonstrates:
        - Agent delegation via tools
        - Intelligent workflow routing
        - Parallel execution where beneficial
        """
        
        # Create coordination context
        context = AgentCoordinationContext(
            session_id=deps.session_id,
            customer_profile={},
            conversation_history=deps.conversation_history,
            processing_stage="initial_analysis"
        )
        
        # Use the main agent to coordinate the workflow
        prompt = f"""
        Process this RFQ using intelligent agent coordination:
        
        Customer Message: "{customer_message}"
        
        Steps to follow:
        1. Use execute_parallel_analysis to get comprehensive initial analysis
        2. Based on urgency and completeness, decide next action
        3. If questions needed, use generate_strategic_questions
        4. If ready for pricing, use develop_pricing_strategy
        5. Provide complete RFQ processing result
        
        Conversation History: {deps.conversation_history}
        """
        
        result = await self.agent.run(prompt, deps=deps)
        return result.data


class GraphBasedRFQController:
    """
    Graph-based state machine for complex RFQ workflows.
    
    Handles advanced scenarios like:
    - Multi-round negotiations
    - Complex approval workflows
    - Competitive bidding scenarios
    """
    
    def __init__(self):
        self.state_graph = {
            "initial": ["requirements_analysis", "urgent_fast_track"],
            "requirements_analysis": ["clarification_needed", "ready_for_pricing"],
            "clarification_needed": ["requirements_analysis", "partial_quote"],
            "ready_for_pricing": ["quote_generation", "competitive_analysis"],
            "quote_generation": ["quote_presented", "pricing_refinement"],
            "quote_presented": ["negotiation", "acceptance", "rejection"],
            "negotiation": ["quote_generation", "acceptance", "rejection"],
            "competitive_analysis": ["competitive_quote", "value_proposition"],
            "urgent_fast_track": ["express_quote", "priority_processing"]
        }
        
        # Initialize state-specific agents
        self._init_state_agents()
    
    def _init_state_agents(self):
        """Initialize agents for specific workflow states."""
        
        # Competitive analysis agent
        self.competitive_agent = Agent(
            get_model_name("customer_intent"),  # Reuse model
            result_type=str,
            system_prompt="""
            You analyze competitive scenarios and develop strategies to win deals.
            Consider market positioning, value propositions, and competitive advantages.
            """
        )
        
        # Negotiation agent
        self.negotiation_agent = Agent(
            get_model_name("pricing_strategy"),  # Reuse model
            result_type=str,
            system_prompt="""
            You handle price negotiations and contract terms discussions.
            Find win-win solutions while protecting business interests.
            """
        )
        
        # Express processing agent for urgent requests
        self.express_agent = Agent(
            get_model_name("rfq_orchestrator"),  # Use main model
            result_type=str,
            system_prompt="""
            You handle urgent RFQ requests with streamlined processing.
            Prioritize speed while maintaining accuracy and professionalism.
            """
        )
    
    async def process_state_transition(
        self, 
        current_state: str, 
        context: AgentCoordinationContext,
        customer_input: str
    ) -> tuple[str, str]:
        """
        Process state transition based on current state and input.
        
        Returns:
            tuple: (next_state, response_message)
        """
        
        if current_state == "competitive_analysis":
            response = await self.competitive_agent.run(
                f"Analyze competitive scenario: {customer_input}\nContext: {context.accumulated_insights}"
            )
            return "competitive_quote", response.data
        
        elif current_state == "negotiation":
            response = await self.negotiation_agent.run(
                f"Handle negotiation: {customer_input}\nHistory: {context.conversation_history}"
            )
            # Determine next state based on negotiation outcome
            if "accept" in response.data.lower():
                return "acceptance", response.data
            elif "counter" in response.data.lower():
                return "quote_generation", response.data
            else:
                return "negotiation", response.data
        
        elif current_state == "urgent_fast_track":
            response = await self.express_agent.run(
                f"Express processing: {customer_input}"
            )
            return "express_quote", response.data
        
        # Default transition logic
        possible_states = self.state_graph.get(current_state, [])
        if possible_states:
            # Simple logic - could be enhanced with ML-based state prediction
            return possible_states[0], f"Transitioning to {possible_states[0]}"
        
        return current_state, "Staying in current state"


# Memory-enhanced agent for learning from interactions
class MemoryEnhancedAgent:
    """
    Agent with persistent memory and learning capabilities.
    
    Features:
    - Conversation memory across sessions
    - Learning from successful/failed interactions
    - Customer preference tracking
    - Performance optimization based on history
    """
    
    def __init__(self):
        self.agent = Agent(
            get_model_name("rfq_orchestrator"),
            result_type=str,
            system_prompt="""
            You are a memory-enhanced RFQ agent that learns from interactions.
            
            Use your memory to:
            1. Remember customer preferences and history
            2. Learn from successful quote patterns
            3. Improve question quality based on past effectiveness
            4. Optimize pricing strategies based on acceptance rates
            
            Always consider historical context when making decisions.
            """
        )
        
        # In a real implementation, this would use a proper database
        self.conversation_memory = {}
        self.customer_preferences = {}
        self.success_patterns = {}
    
    async def process_with_memory(
        self, 
        customer_id: str, 
        message: str,
        context: Dict
    ) -> str:
        """Process request using historical memory and learning."""
        
        # Retrieve customer history
        customer_history = self.conversation_memory.get(customer_id, [])
        preferences = self.customer_preferences.get(customer_id, {})
        
        # Enhanced prompt with memory context
        memory_prompt = f"""
        Customer Message: {message}
        
        Customer History: {customer_history[-5:]}  # Last 5 interactions
        Known Preferences: {preferences}
        Successful Patterns: {self.success_patterns}
        
        Use this historical context to provide the most relevant and effective response.
        """
        
        result = await self.agent.run(memory_prompt)
        
        # Update memory
        self.conversation_memory.setdefault(customer_id, []).append({
            "message": message,
            "response": result.data,
            "timestamp": datetime.now().isoformat()
        })
        
        return result.data
    
    def update_success_pattern(self, pattern_type: str, details: Dict):
        """Update success patterns based on outcomes."""
        if pattern_type not in self.success_patterns:
            self.success_patterns[pattern_type] = []
        
        self.success_patterns[pattern_type].append({
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent patterns (last 100)
        self.success_patterns[pattern_type] = self.success_patterns[pattern_type][-100:]


# Export the enhanced orchestrators
__all__ = [
    "EnhancedRFQOrchestrator",
    "GraphBasedRFQController", 
    "MemoryEnhancedAgent",
    "AgentCoordinationContext"
] 