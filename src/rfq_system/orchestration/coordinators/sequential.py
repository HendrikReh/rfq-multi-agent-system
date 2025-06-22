"""
Sequential agent coordination for the RFQ system.

This module implements sequential execution patterns where agents
are executed one after another in a defined order.
"""

from typing import Any, Dict, List, Optional
from ...core.interfaces.agent import BaseAgent, AgentContext


class SequentialCoordinator:
    """
    Coordinates sequential execution of multiple agents.
    
    This coordinator executes agents one after another, passing
    results from one agent to the next in the chain.
    """
    
    def __init__(self):
        self.execution_order: List[str] = []
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent for sequential execution."""
        self.agents[name] = agent
        if name not in self.execution_order:
            self.execution_order.append(name)
    
    async def process_workflow(
        self, 
        input_data: Any, 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Process workflow sequentially through all registered agents."""
        results = {}
        current_data = input_data
        
        for agent_name in self.execution_order:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                result = await agent.process(current_data, context)
                results[agent_name] = result
                current_data = result
        
        return results 