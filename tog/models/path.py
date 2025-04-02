from dataclasses import dataclass, field
import heapq
from typing import List
from tog.models.entity import Entity
from tog.models.relation import Relation
from tog.models.triple import Triple

@dataclass
class Path:
    path: List[Triple] = field(default_factory=list)
    confidence_score: float = None
    metadata: dict = field(default_factory=dict)

    def add_triple(self, triple: Triple):
        """Add a triple to the path."""
        self.path.append(triple)
    
    def set_confidence_score(self, score: float):
        """Set the confidence score for the path."""
        self.confidence_score = score
    
    def get_last_entity(self) -> Entity:
        """Get the last entity in the path."""
        if self.path:
            return self.path[-1].object
        return None
    
    def get_last_relation(self) -> Relation:
        """Get the last relation in the path."""
        if self.path:
            return self.path[-1].predicate
        return None

@dataclass
class TopNPaths:
    n: int
    heap: List[Path] = None
    
    def __post_init__(self):
        if self.heap is None:
            self.heap = []
    
    def add_path(self, path, confidence):
        # Add the path with negative confidence to maintain a max-heap
        heapq.heappush(self.heap, (-confidence, path))
        if len(self.heap) > self.n:
            heapq.heappop(self.heap)
    
    def get_paths(self):
        # Return paths sorted by confidence in descending order
        return [path for _, path in sorted(self.heap, reverse=True)]
