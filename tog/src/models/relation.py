from dataclasses import dataclass

@dataclass
class Relation:
    """
    A class representing a relation between entities."""
    type: str
    metadata: dict = None

# class Relationship:
#     """
#     Minimalist Relationship class with flexible metadata
#     """
#     id: str
#     source_id: str
#     target_id: str
#     types: List[str] = field(default_factory=list)
#     metadata: Dict[str, Any] = field(default_factory=dict)