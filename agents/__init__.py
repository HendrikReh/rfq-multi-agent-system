"""
RFQ Orchestrator Agents Package

This package contains all the specialized AI agents for comprehensive quote processing:

CORE AGENTS:
- RFQParser: Extracts structured requirements from customer requests
- ConversationStateAgent: Tracks conversation flow and determines current stage
- CustomerIntentAgent: Analyzes customer intent, sentiment, and decision factors
- QuestionGenerationAgent: Generates strategic clarifying questions
- PricingStrategyAgent: Develops intelligent pricing strategies
- EvaluationIntelligenceAgent: Monitors system performance and suggests improvements
- InteractionDecisionAgent: Makes strategic decisions about next steps
- CustomerResponseAgent: Simulates realistic customer responses for demos

ENHANCED ORCHESTRATION:
- RFQOrchestrator: Main coordinator that manages the multi-agent workflow
- EnhancedRFQOrchestrator: Advanced orchestrator with agent delegation
- GraphBasedRFQController: State machine for complex workflows
- MemoryEnhancedAgent: Agent with learning and memory capabilities

SPECIALIZED AGENTS:
- CompetitiveIntelligenceAgent: Analyzes competitive landscape and market positioning
- RiskAssessmentAgent: Evaluates business, project, and customer risks
- ContractTermsAgent: Handles legal terms and compliance requirements
- ProposalWriterAgent: Generates professional proposal documents

INTEGRATION & MONITORING:
- IntegratedRFQSystem: Comprehensive system integrating all agents
- ScenarioRecorder: Records and analyzes system performance
"""

# Core agents
from .conversation_state_agent import ConversationStateAgent
from .customer_intent_agent import CustomerIntentAgent
from .customer_response_agent import CustomerResponseAgent
from .evaluation_intelligence_agent import EvaluationIntelligenceAgent
from .interaction_decision_agent import InteractionDecisionAgent
from .pricing_strategy_agent import PricingStrategyAgent
from .question_generation_agent import QuestionGenerationAgent
from .rfq_orchestrator import RFQOrchestrator
from .rfq_parser import RFQParser
from .scenario_recorder import ScenarioRecorder

# Enhanced orchestration
from .enhanced_orchestrator import (
    EnhancedRFQOrchestrator,
    GraphBasedRFQController,
    MemoryEnhancedAgent,
    AgentCoordinationContext,
)

# Specialized agents
from .competitive_intelligence_agent import CompetitiveIntelligenceAgent, CompetitiveAnalysis
from .contract_terms_agent import ContractTermsAgent, ContractTerms
from .proposal_writer_agent import ProposalWriterAgent, ProposalDocument
from .risk_assessment_agent import RiskAssessmentAgent, RiskAssessment

# Integration framework
from .integration_framework import (
    IntegratedRFQSystem,
    ComprehensiveRFQResult,
    SystemHealthReport,
    AgentHealthStatus,
)

# Data models
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
    RequirementsCompleteness,
    SystemPerformance,
)

# Utilities
from .utils import get_model_name

__all__ = [
    # Core agents
    "RFQParser",
    "ConversationStateAgent", 
    "CustomerIntentAgent",
    "CustomerResponseAgent",
    "InteractionDecisionAgent",
    "QuestionGenerationAgent",
    "PricingStrategyAgent",
    "EvaluationIntelligenceAgent",
    "RFQOrchestrator",
    "ScenarioRecorder",
    
    # Enhanced orchestration
    "EnhancedRFQOrchestrator",
    "GraphBasedRFQController",
    "MemoryEnhancedAgent",
    "AgentCoordinationContext",
    
    # Specialized agents
    "CompetitiveIntelligenceAgent",
    "CompetitiveAnalysis",
    "ContractTermsAgent",
    "ContractTerms",
    "ProposalWriterAgent",
    "ProposalDocument",
    "RiskAssessmentAgent",
    "RiskAssessment",
    
    # Integration framework
    "IntegratedRFQSystem",
    "ComprehensiveRFQResult",
    "SystemHealthReport",
    "AgentHealthStatus",
    
    # Data models
    "RFQRequirements",
    "RequirementsCompleteness",
    "ConversationState",
    "CustomerIntent",
    "CustomerSentiment",
    "InteractionDecision",
    "ClarifyingQuestion",
    "PricingStrategy",
    "Quote",
    "SystemPerformance",
    "RFQProcessingResult",
    "RFQDependencies",
    
    # Utilities
    "get_model_name",
] 