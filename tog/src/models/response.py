from typing import List
from pydantic import BaseModel, Field, field_validator
import re

class ExtractionResponse(BaseModel):
    """
    Pydantic model to validate the response from the entity extraction prompt.
    The response should be a comma-separated list of entities.
    """
    entities: List[str] = Field(..., description="List of extracted entities")

    @field_validator('entities')
    def check_non_empty(cls, value):
        if not value:
            raise ValueError("At least one entity must be extracted")
        return value

    @classmethod
    def from_extraction_output(cls, output: str) -> 'ExtractionResponse':
        """
        Parse the extraction output into an ExtractionResponse object.
        
        Args:
            output: The raw output from the extraction prompt
            
        Returns:
            ExtractionResponse object
        """
        # Clean the output and extract the entities
        # Look for content between triple backticks
        match = re.search(r'```\s*(.*?)\s*```', output, re.DOTALL)
        
        if match:
            content = match.group(1)
        else:
            content = output
            
        # Split by commas and strip whitespace
        entities = [item.strip() for item in content.split(',') if item.strip()]
        
        return cls(entities=entities)