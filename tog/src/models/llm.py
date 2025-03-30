from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from tog.src.models.entity import Entity, RDFEntity

class EntityExtractionResponse(BaseModel):
    """
    Pydantic model for the complete response from an entity extraction operation.
    """
    entities: List[Entity] = Field(default_factory=list, 
                                   description="List of extracted entities")
    
