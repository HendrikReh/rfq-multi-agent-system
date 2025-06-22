"""
Evaluation Intelligence Agent

This agent monitors system performance and provides improvement suggestions.
It analyzes response times, accuracy, and predicted customer satisfaction.
"""

from pydantic_ai import Agent

from .models import Quote, SystemPerformance
from .utils import get_model_name


class EvaluationIntelligenceAgent:
    """Evaluation Intelligence agent for monitoring system performance."""
    
    def __init__(self):
        model_name = get_model_name("evaluation_intelligence")
        self.agent = Agent(
            model_name,
            result_type=SystemPerformance,
            system_prompt="""
            You evaluate system performance and provide improvement suggestions.
            Analyze response times, accuracy, and predicted customer satisfaction.
            Consider factors like:
            - Processing efficiency
            - Quote accuracy
            - Customer experience quality
            - Areas for optimization
            """
        )
    
    async def evaluate(self, processing_time: float, quote: Quote) -> SystemPerformance:
        """
        Evaluate system performance and provide improvement suggestions.
        
        Args:
            processing_time: Time taken to process the RFQ
            quote: Generated quote to evaluate
            
        Returns:
            SystemPerformance: Performance metrics and improvement suggestions
        """
        context = f"Processing time: {processing_time}s\nQuote: {quote.model_dump()}"
        result = await self.agent.run(context)
        return result.output 