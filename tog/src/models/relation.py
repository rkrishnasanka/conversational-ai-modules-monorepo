from dataclasses import dataclass

@dataclass
class Relation:
    """
    A class representing a relation between entities."""
    type: str
    description: str
    metadata: dict = None
