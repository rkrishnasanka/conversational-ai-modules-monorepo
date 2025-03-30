from dataclasses import dataclass

@dataclass
class Entity:
    """A class representing an entity."""
    id: str
    name: str
    type: str
    description: str = None
    metadata: dict = None

@dataclass
class RDFEntity(Entity):
    """A class representing an RDF entity."""
    uri: str
    graph: str = None
    label: str = None