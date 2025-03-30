from dataclasses import dataclass
import heapq
from tog.src.models.triple import Triple

@dataclass
class Path:
    triples: list[Triple] = None
    confidence_score: float = None
    metadata: dict = None

@dataclass
class TopNPaths:
    n: int
    heap: list = None
    
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
