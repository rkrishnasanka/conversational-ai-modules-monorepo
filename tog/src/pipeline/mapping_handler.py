from abc import ABC, abstractmethod
from typing import Optional
from tog.src.models.entity import Entity
from tog.src.models.kg import KnowledgeGraph
from fuzzywuzzy import process
from tog.src.utils.logger import setup_logger
class MappingHandler(ABC):
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.logger = setup_logger(name="mapping_handler", log_filename="mapping_handler.log")
        self.logger.info(f"MappingHandler initialized with kg: {KnowledgeGraph}")

    @abstractmethod
    def fuzzy_match(self, entity: str) -> Optional[Entity]:
        """
        Perform fuzzy matching on the entity against the knowledge graph.
        """
        # Placeholder for fuzzy matching logic
        pass

    @abstractmethod
    def semantic_match(self, entity: str) -> Optional[Entity]:
        """
        Perform semantic matching on the entity against the knowledge graph.
        """
        # Placeholder for semantic matching logic
        pass

    @abstractmethod
    def levanstein_match(self, entity: str) -> Optional[Entity]:
        """
        Perform Levenshtein matching on the entity against the knowledge graph.
        """
        # Placeholder for Levenshtein matching logic
        pass

class Neo4jMappingHandler(MappingHandler):
    
    def fuzzy_match(self):
        """
        Perform fuzzy matching on the entity against the Neo4j knowledge graph.
        """
        # Implement fuzzy matching logic using Neo4j
        pass

    def semantic_match(self):
        """
        Perform semantic matching on the entity against the Neo4j knowledge graph.
        """
        # Implement semantic matching logic using Neo4j
        pass

    def levanstein_match(self):
        """
        Perform Levenshtein matching on the entity against the Neo4j knowledge graph.
        """
        # Implement Levenshtein matching logic using Neo4j
        pass