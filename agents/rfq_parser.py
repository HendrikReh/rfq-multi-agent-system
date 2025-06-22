"""
RFQ Parser Agent

This agent extracts structured requirements from customer RFQ requests.
It parses product type, quantity, specifications, delivery dates, and special requirements,
and assesses the completeness of the information provided.
"""

from pydantic_ai import Agent

from .models import RFQRequirements
from .utils import get_model_name


class RFQParser:
    """RFQ Parser agent for extracting and assessing structured requirements."""
    
    def __init__(self):
        model_name = get_model_name("rfq_parser")
        self.agent = Agent(
            model_name,
            result_type=RFQRequirements,
            system_prompt="""
            You are an expert RFQ parser and requirements analyst. Extract structured requirements 
            from customer requests and assess their completeness.
            
            Focus on identifying:
            - Product/service type (required)
            - Quantities needed (important for pricing)
            - Technical specifications (critical for accurate quotes)
            - Timeline/delivery requirements (affects pricing and feasibility)
            - Budget constraints (essential for appropriate pricing)
            - Special requirements or conditions
            
            CRITICAL: Assess the completeness of requirements:
            - COMPLETE: All essential information provided (product, quantity, specs, timeline, budget)
            - PARTIAL: Most key information provided, minor details missing
            - MINIMAL: Basic product type identified, but missing critical details
            - UNCLEAR: Vague or ambiguous request, difficult to understand needs
            
            Identify specific missing information that would be needed for accurate quoting:
            - Missing quantity information
            - Unclear technical specifications
            - No timeline/delivery requirements
            - No budget information provided
            - Ambiguous product/service requirements
            - Missing decision-maker information
            - Unclear implementation requirements
            """
        )
    
    async def parse(self, rfq_text: str) -> RFQRequirements:
        """
        Parse RFQ text into structured requirements with completeness assessment.
        
        Args:
            rfq_text: The customer's RFQ message text
            
        Returns:
            RFQRequirements: Structured requirements with completeness assessment
        """
        enhanced_prompt = f"""
        Analyze this customer RFQ request and extract structured requirements.
        Pay special attention to what information is missing that would be needed for accurate quoting.
        
        Customer Request: {rfq_text}
        
        Extract all available information and clearly identify what's missing for a complete quote.
        """
        
        result = await self.agent.run(enhanced_prompt)
        return result.output 