from typing import List
from pydantic import BaseModel, Field, field_validator
import re
import json

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

class RelationPruneResponse(BaseModel):
    """
    Pydantic model to validate the response from the relation pruning prompt.
    The response should be a JSON object mapping relations to relevance scores.
    """
    relations: dict[str, float] = Field(..., description="Dictionary mapping relations to their relevance scores")

    @field_validator('relations')
    def check_scores_sum_to_one(cls, value):
        if not value:
            raise ValueError("At least one relation must be returned")
        
        total = sum(value.values())
        if not 0.99 <= total <= 1.01:  # Allow for minor floating-point errors
            raise ValueError(f"Relation scores must sum to 1.0, got {total}")
        
        return value

    @classmethod
    def from_prune_output(cls, output: str) -> 'RelationPruneResponse':
        """
        Parse the relation pruning output into a RelationPruneResponse object.
        
        Args:
            output: The raw output from the relation pruning prompt
            
        Returns:
            RelationPruneResponse object
        """
        # Clean the output and extract the JSON
        # Look for content between triple backticks or find JSON directly
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', output, re.DOTALL)
        
        if match:
            content = match.group(1)
        else:
            # Try to extract JSON directly
            content = output.strip()
            
        try:
            relations_dict = json.loads(content)
            if not isinstance(relations_dict, dict):
                raise ValueError("Expected a JSON object/dictionary")
            
            # Convert all scores to float for consistency
            relations_dict = {k: float(v) for k, v in relations_dict.items()}
            
            return cls(relations=relations_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from output: {str(e)}")
        
