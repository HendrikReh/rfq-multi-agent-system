"""
Base agent interfaces and protocols for the RFQ multi-agent system.

This module defines the core interfaces that all agents must implement,
following modern multi-agent architecture patterns and ensuring consistency
across the system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from datetime import datetime
from dataclasses import dataclass

from pydantic import BaseModel, Field


# Type variables for generic agent interface
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
ContextType = TypeVar('ContextType')


class AgentStatus(str, Enum):
    """Agent operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    STOPPED = "stopped"


class AgentCapability(str, Enum):
    """Standard agent capabilities."""
    # Core capabilities
    PROCESS_RFQ = "process_rfq"
    ANALYZE_CUSTOMER = "analyze_customer"
    GENERATE_QUESTIONS = "generate_questions"
    PRICE_STRATEGY = "price_strategy"
    
    # Specialized capabilities
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    CONTRACT_TERMS = "contract_terms"
    PROPOSAL_WRITING = "proposal_writing"
    
    # Orchestration capabilities
    DELEGATE_TASKS = "delegate_tasks"
    COORDINATE_AGENTS = "coordinate_agents"
    MANAGE_WORKFLOW = "manage_workflow"
    
    # Evaluation capabilities
    MONITOR_PERFORMANCE = "monitor_performance"
    ASSESS_QUALITY = "assess_quality"


class AgentHealthStatus(BaseModel):
    """Agent health information."""
    agent_id: str
    status: AgentStatus
    last_heartbeat: datetime
    response_time_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    capabilities: List[AgentCapability] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    last_24h_requests: int = 0
    uptime_percentage: float = 100.0


@dataclass
class AgentContext:
    """Context passed to agents during processing."""
    request_id: str
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    shared_memory: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.shared_memory is None:
            self.shared_memory = {}
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC, Generic[InputType, OutputType]):
    """
    Base interface for all agents in the RFQ system.
    
    This abstract base class defines the core interface that all agents
    must implement, ensuring consistency and enabling interoperability
    across the multi-agent system.
    
    Following Anthropic's multi-agent patterns, agents should be:
    - Stateless where possible
    - Capable of graceful error handling
    - Observable through health checks and metrics
    - Composable with other agents
    """
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self._status = AgentStatus.INITIALIZING
        self._metrics = AgentMetrics()
        self._last_heartbeat = datetime.now()
    
    @abstractmethod
    async def process(
        self, 
        input_data: InputType, 
        context: AgentContext
    ) -> OutputType:
        """
        Process input data and return result.
        
        This is the main entry point for agent processing. Implementations
        should handle errors gracefully and update metrics accordingly.
        
        Args:
            input_data: The input data to process
            context: Processing context with shared state
            
        Returns:
            The processed result
            
        Raises:
            AgentProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent and its dependencies."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        pass
    
    async def health_check(self) -> AgentHealthStatus:
        """
        Return current agent health status.
        
        This method should be implemented to provide real-time health
        information about the agent's operational status.
        """
        self._last_heartbeat = datetime.now()
        
        return AgentHealthStatus(
            agent_id=self.agent_id,
            status=self._status,
            last_heartbeat=self._last_heartbeat,
            response_time_ms=self._metrics.average_response_time_ms,
            error_count=self._metrics.failed_requests,
            capabilities=self.capabilities,
            metadata={"type": self.__class__.__name__}
        )
    
    def get_metrics(self) -> AgentMetrics:
        """Return current agent performance metrics."""
        return self._metrics
    
    def update_status(self, status: AgentStatus) -> None:
        """Update agent operational status."""
        self._status = status
    
    def record_request(self, success: bool, response_time_ms: float) -> None:
        """Record request metrics."""
        self._metrics.total_requests += 1
        if success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1
        
        # Update average response time
        total_successful = self._metrics.successful_requests
        if total_successful > 0:
            current_avg = self._metrics.average_response_time_ms
            self._metrics.average_response_time_ms = (
                (current_avg * (total_successful - 1) + response_time_ms) / total_successful
            )


class DelegatingAgent(BaseAgent[InputType, OutputType]):
    """
    Base class for agents that can delegate tasks to other agents.
    
    This follows Anthropic's pattern of lead agents that coordinate
    subagents for complex tasks.
    """
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, capabilities)
        self._subagents: Dict[str, BaseAgent] = {}
    
    def register_subagent(self, name: str, agent: BaseAgent) -> None:
        """Register a subagent for delegation."""
        self._subagents[name] = agent
    
    def get_subagent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered subagent by name."""
        return self._subagents.get(name)
    
    def list_subagents(self) -> List[str]:
        """List all registered subagent names."""
        return list(self._subagents.keys())
    
    @abstractmethod
    async def delegate_task(
        self, 
        task_name: str, 
        input_data: Any, 
        context: AgentContext
    ) -> Any:
        """Delegate a task to an appropriate subagent."""
        pass


class SpecializedAgent(BaseAgent[InputType, OutputType]):
    """
    Base class for domain-specific specialized agents.
    
    Specialized agents focus on specific domain tasks like competitive
    analysis, risk assessment, or proposal writing.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        domain: str, 
        capabilities: List[AgentCapability]
    ):
        super().__init__(agent_id, capabilities)
        self.domain = domain
    
    @abstractmethod
    def get_domain_expertise(self) -> Dict[str, Any]:
        """Return information about the agent's domain expertise."""
        pass


# Protocol for agent factories
class AgentFactory(ABC):
    """Abstract factory for creating agents."""
    
    @abstractmethod
    def create_agent(self, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
        """Create an agent of the specified type with given configuration."""
        pass
    
    @abstractmethod
    def list_supported_types(self) -> List[str]:
        """List all supported agent types."""
        pass 