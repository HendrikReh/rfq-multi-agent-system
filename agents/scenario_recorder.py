"""
Scenario Recorder

Utility for recording RFQ scenario runs to JSON files for analysis and tracking.
Saves detailed scenario data including customer interactions, system responses,
and performance metrics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    ClarifyingQuestion,
    CustomerIntent,
    InteractionDecision,
    Quote,
    RFQProcessingResult,
    RFQRequirements,
    SystemPerformance,
)


class ScenarioRecorder:
    """Records scenario runs to JSON files for analysis and tracking."""
    
    def __init__(self, reports_dir: str = "./reports"):
        """
        Initialize the scenario recorder.
        
        Args:
            reports_dir: Directory to save scenario reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
    
    def _generate_filename(self, scenario_id: int) -> str:
        """
        Generate filename with pattern: {date}_{time}_scenario_{scenario_id}.json
        
        Args:
            scenario_id: Unique identifier for the scenario
            
        Returns:
            str: Generated filename
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        return f"{date_str}_{time_str}_scenario_{scenario_id}.json"
    
    def _serialize_pydantic_model(self, obj: Any) -> Any:
        """
        Convert Pydantic models to serializable dictionaries.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable representation of the object
        """
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif isinstance(obj, list):
            return [self._serialize_pydantic_model(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._serialize_pydantic_model(value) for key, value in obj.items()}
        else:
            return obj
    
    def record_scenario(
        self,
        scenario_id: int,
        scenario_name: str,
        customer_persona: str,
        business_context: str,
        initial_inquiry: str,
        initial_result: RFQProcessingResult,
        customer_responses: List[str] = None,
        final_result: Optional[RFQProcessingResult] = None,
        quote_response: Optional[str] = None,
        error_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_models: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Record a complete scenario run to a JSON file.
        
        Args:
            scenario_id: Unique identifier for the scenario
            scenario_name: Human-readable name for the scenario
            customer_persona: Description of the customer persona
            business_context: Business context for the scenario
            initial_inquiry: Customer's initial inquiry
            initial_result: System's initial processing result
            customer_responses: List of customer responses during the flow
            final_result: Final processing result if different from initial
            quote_response: Customer's response to the quote
            error_info: Error information if scenario failed
            metadata: Additional metadata about the scenario
            agent_models: Dictionary mapping agent types to their model names
            
        Returns:
            str: Path to the saved JSON file
        """
        timestamp = datetime.now().isoformat()
        filename = self._generate_filename(scenario_id)
        filepath = self.reports_dir / filename
        
        # Build comprehensive scenario data
        scenario_data = {
            "metadata": {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "timestamp": timestamp,
                "filename": filename,
                "version": "1.0",
                **(metadata or {})
            },
            "customer_profile": {
                "persona": customer_persona,
                "business_context": business_context
            },
            "conversation_flow": {
                "initial_inquiry": initial_inquiry,
                "customer_responses": customer_responses or [],
                "quote_response": quote_response
            },
            "system_processing": {
                "initial_result": self._serialize_pydantic_model(initial_result),
                "final_result": self._serialize_pydantic_model(final_result) if final_result else None
            },
            "agent_models": agent_models or {},
            "analytics": self._extract_analytics(initial_result, final_result),
            "error_info": error_info
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenario_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(filepath)
    
    def _extract_analytics(
        self,
        initial_result: RFQProcessingResult,
        final_result: Optional[RFQProcessingResult] = None
    ) -> Dict[str, Any]:
        """
        Extract analytics and metrics from scenario results.
        
        Args:
            initial_result: Initial processing result
            final_result: Final processing result if available
            
        Returns:
            Dictionary with extracted analytics
        """
        result_to_analyze = final_result or initial_result
        
        analytics = {
            "decision_confidence": result_to_analyze.interaction_decision.confidence_level,
            "should_ask_questions": result_to_analyze.interaction_decision.should_ask_questions,
            "requirements_completeness": result_to_analyze.requirements.completeness.value,
            "customer_urgency": result_to_analyze.customer_intent.urgency_level,
            "customer_price_sensitivity": result_to_analyze.customer_intent.price_sensitivity,
            "customer_readiness_to_buy": result_to_analyze.customer_intent.readiness_to_buy,
            "questions_generated": len(result_to_analyze.clarifying_questions) if result_to_analyze.clarifying_questions else 0,
            "quote_generated": result_to_analyze.quote is not None,
            "total_quote_value": None,
            "performance_metrics": None
        }
        
        # Extract quote information
        if result_to_analyze.quote:
            analytics["total_quote_value"] = result_to_analyze.quote.total_price
            analytics["quote_line_items"] = len(result_to_analyze.quote.items)
        
        # Extract performance metrics
        if result_to_analyze.performance:
            analytics["performance_metrics"] = {
                "response_time": result_to_analyze.performance.response_time,
                "accuracy_score": result_to_analyze.performance.accuracy_score,
                "customer_satisfaction_prediction": result_to_analyze.performance.customer_satisfaction_prediction,
                "improvement_suggestions": result_to_analyze.performance.improvement_suggestions
            }
        
        # Extract question priorities if available
        if result_to_analyze.clarifying_questions:
            question_priorities = [q.priority for q in result_to_analyze.clarifying_questions]
            analytics["question_priorities"] = {
                "average_priority": sum(question_priorities) / len(question_priorities),
                "max_priority": max(question_priorities),
                "min_priority": min(question_priorities),
                "priority_distribution": {
                    str(i): question_priorities.count(i) for i in range(1, 6)
                }
            }
        
        return analytics
    
    def record_error_scenario(
        self,
        scenario_id: int,
        scenario_name: str,
        customer_persona: str,
        business_context: str,
        initial_inquiry: str,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None,
        agent_models: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Record a scenario that encountered an error.
        
        Args:
            scenario_id: Unique identifier for the scenario
            scenario_name: Human-readable name for the scenario
            customer_persona: Description of the customer persona
            business_context: Business context for the scenario
            initial_inquiry: Customer's initial inquiry
            error: Exception that occurred
            metadata: Additional metadata about the scenario
            agent_models: Dictionary mapping agent types to their model names
            
        Returns:
            str: Path to the saved JSON file
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_occurred": True
        }
        
        # Create a minimal result for error scenarios
        from .models import (
            ConversationState,
            CustomerSentiment,
            RequirementsCompleteness,
            RFQProcessingResult,
            CustomerIntent,
            InteractionDecision,
            RFQRequirements,
            SystemPerformance
        )
        
        # Create default objects for error scenario
        error_result = RFQProcessingResult(
            status="error",
            conversation_state="error",
            requirements=RFQRequirements(
                product_type="unknown",
                quantity=0,
                completeness=RequirementsCompleteness.UNCLEAR,
                missing_info=["Error occurred during processing"]
            ),
            customer_intent=CustomerIntent(
                primary_intent="unknown",
                sentiment=CustomerSentiment.NEUTRAL,
                urgency_level=1,
                price_sensitivity=1,
                readiness_to_buy=1,
                decision_factors=["Error occurred during analysis"]
            ),
            interaction_decision=InteractionDecision(
                should_ask_questions=False,
                should_generate_quote=False,
                next_action="error_handling",
                confidence_level=1,
                reasoning="Error occurred during decision making"
            ),
            clarifying_questions=[],
            next_steps=["Contact support", "Retry request"],
            message_to_customer="An error occurred while processing your request.",
            performance=SystemPerformance(
                response_time=0.0,
                accuracy_score=0.0,
                customer_satisfaction_prediction=0.0,
                improvement_suggestions=["Fix the error that occurred during processing"]
            )
        )
        
        return self.record_scenario(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            customer_persona=customer_persona,
            business_context=business_context,
            initial_inquiry=initial_inquiry,
            initial_result=error_result,
            error_info=error_info,
            metadata=metadata,
            agent_models=agent_models
        )
    
    def list_scenario_files(self) -> List[str]:
        """
        List all scenario JSON files in the reports directory.
        
        Returns:
            List of scenario file paths
        """
        pattern = "*_scenario_*.json"
        return [str(f) for f in self.reports_dir.glob(pattern)]
    
    def load_scenario(self, filepath: str) -> Dict[str, Any]:
        """
        Load a scenario from a JSON file.
        
        Args:
            filepath: Path to the scenario JSON file
            
        Returns:
            Dictionary with scenario data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_scenario_summary(self, filepath: str) -> Dict[str, Any]:
        """
        Get a summary of a scenario from its JSON file.
        
        Args:
            filepath: Path to the scenario JSON file
            
        Returns:
            Dictionary with scenario summary
        """
        scenario_data = self.load_scenario(filepath)
        
        error_info = scenario_data.get("error_info")
        error_occurred = error_info.get("error_occurred", False) if error_info else False
        
        return {
            "scenario_id": scenario_data["metadata"]["scenario_id"],
            "scenario_name": scenario_data["metadata"]["scenario_name"],
            "timestamp": scenario_data["metadata"]["timestamp"],
            "customer_persona": scenario_data["customer_profile"]["persona"],
            "quote_generated": scenario_data["analytics"]["quote_generated"],
            "total_quote_value": scenario_data["analytics"]["total_quote_value"],
            "questions_asked": scenario_data["analytics"]["questions_generated"],
            "decision_confidence": scenario_data["analytics"]["decision_confidence"],
            "error_occurred": error_occurred
        } 