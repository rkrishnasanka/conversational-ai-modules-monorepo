from dataclasses import dataclass,field
from typing import List, Dict, Any
@dataclass
class Relation:
    """
    A class representing a relation between entities."""
    id: str
    subject_id: str
    object_id: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# class Relationship:
#     """
#     Minimalist Relationship class with flexible metadata
#     """
#     id: str
#     source_id: str
#     target_id: str
#     types: List[str] = field(default_factory=list)
#     metadata: Dict[str, Any] = field(default_factory=dict)