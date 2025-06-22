```
        _             _            _           _                  _          _                           
       / /\      _   /\ \         /\ \        /\_\               /\ \       /\ \     _                   
      / / /    / /\ /  \ \       /  \ \      / / /  _            \ \ \     /  \ \   /\_\                 
     / / /    / / // /\ \ \     / /\ \ \    / / /  /\_\          /\ \_\   / /\ \ \_/ / /                 
    / / /_   / / // / /\ \ \   / / /\ \_\  / / /__/ / /         / /\/_/  / / /\ \___/ /                  
   / /_//_/\/ / // / /  \ \_\ / / /_/ / / / /\_____/ /         / / /    / / /  \/____/                   
  / _______/\/ // / /   / / // / /__\/ / / /\_______/         / / /    / / /    / / /                    
 / /  \____\  // / /   / / // / /_____/ / / /\ \ \           / / /    / / /    / / /                     
/_/ /\ \ /\ \// / /___/ / // / /\ \ \  / / /  \ \ \      ___/ / /__  / / /    / / /                      
\_\//_/ /_/ // / /____\/ // / /  \ \ \/ / /    \ \ \    /\__\/_/___\/ / /    / / /                       
    \_\/\_\/ \/_________/ \/_/  _ \_\/\/_/   _  \_\_\   \/_________/\/_/_    \/_/   _           _        
        /\ \       /\ \        /\ \         /\ \           /\ \        /\ \        / /\        / /\      
       /  \ \     /  \ \      /  \ \       /  \ \         /  \ \      /  \ \      / /  \      / /  \     
      / /\ \ \   / /\ \ \    / /\ \ \     / /\ \_\       / /\ \ \    / /\ \ \    / / /\ \__  / / /\ \__  
     / / /\ \_\ / / /\ \_\  / / /\ \ \   / / /\/_/      / / /\ \_\  / / /\ \_\  / / /\ \___\/ / /\ \___\ 
    / / /_/ / // / /_/ / / / / /  \ \_\ / / / ______   / / /_/ / / / /_/_ \/_/  \ \ \ \/___/\ \ \ \/___/ 
   / / /__\/ // / /__\/ / / / /   / / // / / /\_____\ / / /__\/ / / /____/\      \ \ \       \ \ \       
  / / /_____// / /_____/ / / /   / / // / /  \/____ // / /_____/ / /\____\/  _    \ \ \  _    \ \ \      
 / / /      / / /\ \ \  / / /___/ / // / /_____/ / // / /\ \ \  / / /______ /_/\__/ / / /_/\__/ / /      
/ / /      / / /  \ \ \/ / /____\/ // / /______\/ // / /  \ \ \/ / /_______\\ \/___/ /  \ \/___/ /       
\/_/       \/_/    \_\/\/_________/ \/___________/ \/_/    \_\/\/__________/ \_____\/    \_____\/        
                                                                                                         

```

# RFQ Multi-Agent System

**Home Project #1**

**Course:** https://maven.com/will-brown-kyle-corbitt/agents-mcp-rl

A multi-agent system that intelligently processes Request for Quote (RFQ) inquiries.
The system analyzes customer requirements, makes strategic decisions about when to ask clarifying questions, and generates (accurate)  quotes.

## Overview

This system demonstrates modern multi-agent architecture patterns inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system), providing:

- **Domain specific Agents** for comprehensive RFQ processing
- **Modular Architecture** with clear separation of concerns
- **Production Orchestration** with parallel execution and health monitoring
- **Comprehensive Testing** with unit, integration, and performance tests
- **Logfire Observability** with complete LLM conversation tracing and performance monitoring

## Agent Ecosystem

### Core Processing Agents (9 agents)
- **RFQParser** - Requirements extraction and validation
- **ConversationStateAgent** - State management and tracking
- **CustomerIntentAgent** - Sentiment analysis and buying readiness
- **InteractionDecisionAgent** - Strategic workflow decisions
- **QuestionGenerationAgent** - Context-aware clarifying questions
- **PricingStrategyAgent** - Intelligent pricing strategies
- **EvaluationIntelligenceAgent** - Performance monitoring
- **CustomerResponseAgent** - Customer simulation and testing
- **RFQOrchestrator** - Core workflow coordination

### Specialized Domain Agents (4 agents)
- **CompetitiveIntelligenceAgent** - Market positioning and win probability
- **RiskAssessmentAgent** - 10-point risk scoring across 5 categories
- **ContractTermsAgent** - Legal terms and compliance management
- **ProposalWriterAgent** - Professional document generation

### Evaluation & Quality Assurance
- **BestOfNSelector** - Multiple candidate generation with LLM judge evaluation
- **LLM Judge System** - Structured scoring across accuracy, completeness, relevance, clarity
- **Confidence Scoring** - Score distribution analysis and quality metrics

## RUN THE DEMO

```
uv run ./examples/demo_real_llm_evaluation.py
```
