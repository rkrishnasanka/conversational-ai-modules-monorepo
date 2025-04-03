from abc import ABC, abstractmethod
from typing import List
from tog.llms.base_llm import BaseLLM
from tog.models.entity import Entity
from tog.kgs import KnowledgeGraph
from tog.models.path import TopNPaths
from tog.models.relation import Relation
from tog.utils.logger import setup_logger
from tog.utils.prompt_loader import PromptLoader

class RelationExplorer(ABC):
    def __init__(self, query: str, llm: BaseLLM, kg: KnowledgeGraph):
        self.query = query
        self.llm = llm
        self.kg = kg
        self.prompt_loader = PromptLoader()
        self.logger = setup_logger(name="relation_explorer", log_filename="relation_explorer.log")

    def explore_relations(self, entity: Entity) -> List[Entity]:
        """
        Explore relations of the given entity in the knowledge graph.
        
        Args:
            entity: The entity to explore relations for.
            
        Returns:
            A list of related entities or relations.
        """
        candidate_relations: List[Relation] = self._get_relations(entity)
        self.logger.debug(f"Candidate relations for {entity.name}: {candidate_relations}")

        pruned_relations: List[Relation] = self._prune_relations(candidate_relations)
        self.logger.debug(f"Pruned relations for {entity.name}: {pruned_relations}")
        
        return pruned_relations
    
    @abstractmethod
    def _get_relations(self, entity: Entity) -> List[Relation]:
        """
        Get relations for the given entity from the knowledge graph.
        
        Args:
            entity: The entity to get relations for.
            
        Returns:
            A list of candidate relations.
        """
        pass

    @abstractmethod
    def _prune_relations(self, query: str, entity: Entity, relations: List[Relation]) -> List[Entity]:
        """
        Prune the list of relations based on certain criteria.
        
        Args:
            relations: The list of candidate relations to prune.
            
        Returns:
            A pruned list of relations.
        """
        # Placeholder for pruning logic
        # it is same for all explorers
        pass

class Neo4jRelationExplorer(RelationExplorer):
    def _get_relations(self, entity: Entity) -> List[Relation]:
        """
        Get relations for the given entity from the Neo4j knowledge graph.
        
        Args:
            entity: The entity to get relations for.
            
        Returns:
            A list of candidate relations.
        """
        # Placeholder for Neo4j relation retrieval logic
        # This should interact with the Neo4j database to get relations
        pass

