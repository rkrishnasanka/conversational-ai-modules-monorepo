from abc import ABC, abstractmethod
from typing import List

from tog.llms.base_llm import BaseLLM
from tog.models.entity import Entity
from tog.kgs import KnowledgeGraph
from tog.utils.logger import setup_logger
from tog.utils.prompt_loader import PromptLoader


class EntityExplorer(ABC):
    """
    Abstract base class for entity exploration in a knowledge graph.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prompt_loader = PromptLoader()
        self.logger = setup_logger(name="entity_explorer", log_filename="entity_explorer.log")
    
    def explore_entities(self, entity: Entity) -> List[Entity]:
        """
        Get entities related to the given entity in the knowledge graph.
        
        Args:
            entity: The entity to explore.
            
        Returns:
            A list of related entities.
        """
        candidate_entities: List[Entity] = self._get_related_entities(entity)
        self.logger.debug(f"Candidate entities for {entity.name}: {candidate_entities}")
        
        pruned_entities: List[Entity] = self._prune_entities(candidate_entities)
        self.logger.debug(f"Pruned entities for {entity.name}: {pruned_entities}")
        
        return pruned_entities
    
    # TODO: Implement the following methods
    @abstractmethod
    def _get_related_entities(self, entity: Entity) -> List[Entity]:
        """
        Get related entities for the given entity from the knowledge graph.
        
        Args:
            entity: The entity to explore.
            
        Returns:
            A list of candidate entities.
        """
        # Placeholder for related entity retrieval logic
        pass

    def _prune_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Prune the list of entities based on certain criteria.
        
        Args:
            entities: The list of candidate entities to prune.
            
        Returns:
            A pruned list of entities.
        """
        # Placeholder for pruning logic it is same for all explorers
        return entities
    
class Neo4jEntityExplorer(EntityExplorer):
    """
    Entity explorer for Neo4j knowledge graph.
    """
    
    def _get_related_entities(self, entity: Entity) -> List[Entity]:
        """
        Get related entities for the given entity from the Neo4j knowledge graph.
        
        Args:
            entity: The entity to explore.
            
        Returns:
            A list of candidate entities.
        """
        # Implement related entity retrieval logic using Neo4j
        pass