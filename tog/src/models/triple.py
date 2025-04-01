from dataclasses import dataclass
from tog.src.models.entity import Entity
from tog.src.models.relation import Relation

@dataclass
class Triple:
    """
    A class representing a triple consisting of a subject, predicate, and object.
    """
    subject: Entity
    predicate: Relation
    object: Entity
    metadata: dict = None

