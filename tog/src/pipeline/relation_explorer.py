from abc import ABC, abstractmethod
from typing import List
from tog.src.llms.base_llm import BaseLLM
from tog.src.models.entity import Entity
from tog.src.models.kg import KnowledgeGraph
from tog.src.models.path import TopNPaths
from tog.src.utils.logger import setup_logger
from tog.src.utils.prompt_loader import PromptLoader

class RelationExplorer(ABC):
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph):
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
        candidate_relations: List[Entity] = self._get_relations(entity)
        self.logger.debug(f"Candidate relations for {entity.name}: {candidate_relations}")

        pruned_relations: List[Entity] = self._prune_relations(candidate_relations)
        self.logger.debug(f"Pruned relations for {entity.name}: {pruned_relations}")

        return pruned_relations
    
    @abstractmethod
    def _get_relations(self, entity: Entity) -> List[Entity]:
        """
        Get relations for the given entity from the knowledge graph.
        
        Args:
            entity: The entity to get relations for.
            
        Returns:
            A list of candidate relations.
        """
        pass

    @abstractmethod
    def _prune_relations(self, relations: List[Entity]) -> List[Entity]:
        """
        Prune the list of relations based on certain criteria.
        
        Args:
            relations: The list of candidate relations to prune.
            
        Returns:
            A pruned list of relations.
        """
        pass

class Neo4jRelationExplorer(RelationExplorer):
    def _get_relations(self, entity: Entity) -> List[Entity]:
        """
        Get relations for the given entity from the Neo4j knowledge graph.
        
        Args:
            entity: The entity to get relations for.
            
        Returns:
            A list of candidate relations.
        """
        query = f"MATCH (e:Entity {{name: '{entity.name}'}})-[r]->(related) RETURN r, related"
        results = self.kg.query(query)
        return [Entity(name=result["related"]["name"]) for result in results]

    # TODO: Write pruning logic
    def _prune_relations(self, relations: List[Entity]) -> List[Entity]:
        """
        Prune the list of relations based on certain criteria.
        
        Args:
            relations: The list of candidate relations to prune.
            
        Returns:
            A pruned list of relations.
        """
        # Placeholder for pruning logic
        pass
