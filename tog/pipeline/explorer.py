from abc import ABC, abstractmethod
from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from typing import Dict, List

from tog.models.entity import Entity
from tog.models.relation import Relation

class Explorer:
    """
    Abstract base class for exploring entities and relations in a knowledge graph.
    """
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prompt: str, prompt_params: Dict[str, str] = None):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prompt = prompt
        self.prompt_params = prompt_params or {}


class RelationExplorer(Explorer, ABC):
    """
    Abstract base class for relation explorers.
    This class defines the interface for exploring relations in a knowledge graph.
    """

    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prompt: str, prompt_params: Dict[str, str] = None):
        """
        Initialize the RelationExplorer with a language model, knowledge graph, and optional query and prompt.

        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): Optional query string for exploration.
            prompt (str): Optional prompt string for exploration.
            prompt_params (dict): Optional parameters for the prompt.
        """
        super().__init__(llm, kg, query, prompt, prompt_params)

    def explore_relations(self, entity: Entity) -> List[Relation]:
        """
        Explore relations associated with the given entity.

        Args:
            entity (Entity): The entity to explore relations for.

        Returns:
            List[Relation]: A list of relations associated with the entity.
        """
        candidate_relations: List[Relation] = self.get_candidates(entity)
        pruned_relations: List[Relation] = self.prune_candidates(candidate_relations)
        return pruned_relations
    
    @abstractmethod
    def get_candidates(self, entity: Entity) -> List[Relation]:
        """
        Get candidate relations for the given entity.

        Args:
            entity (Entity): The entity to get candidates for.

        Returns:
            List[Relation]: A list of candidate relations.
        """
        pass

    def prune_candidates(self, relations: List[Relation]) -> List[Relation]:
        """
        Prune the candidate relations based on some criteria.

        Args:
            relations (List[Relation]): The list of candidate relations to prune.

        Returns:
            List[Relation]: A list of pruned relations.
        """
        # Implement pruning logic here
        return relations

class Neo4jRelationExplorer(RelationExplorer):
    """
    A concrete implementation of RelationExplorer for Neo4j knowledge graph.
    """
    
    def get_candidates(self, entity: Entity) -> List[Relation]:
        """
        Get candidate relations for the given entity using Neo4j knowledge graph.

        Args:
            entity (Entity): The entity to get candidates for.

        Returns:
            List[Relation]: A list of candidate relations.
        """
        try:
            # Query to find all relationships where the given entity is either the subject or object
            query = """
            MATCH (n)-[r]-(m)
            WHERE n.id = $entity_id
            RETURN 
                n.id as subject_id, 
                m.id as object_id,
                type(r) as relation_type,
                id(r) as relation_id,
                properties(r) as metadata
            """
            
            # Execute the query using the knowledge graph
            results = self.kg.query(query, entity_id=entity.id)
            
            # Convert the query results to Relation objects
            relations = []
            for result in results:
                relation_id = str(result.get("relation_id", ""))

                if not relation_id:
                    # TODO: Handle case where relation_id is not found
                    ...

                # Create a Relation object
                relation = Relation(
                    id=relation_id,
                    subject_id=result["subject_id"],
                    object_id=result["object_id"],
                    type=result["type"],
                    metadata=result.get("metadata", {})
                )
                relations.append(relation)
                
            return relations
        except Exception as e:
            # Log error and return empty list
            import logging
            logging.error(f"Error querying Neo4j for relations: {str(e)}")
            return []
        
class EntityExplorer(ABC):
    """
    Abstract base class for entity exploration in a knowledge graph.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str):
        self.llm = llm
        self.kg = kg
        self.query = query
    
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
    def _get_related_entities(self, relation: Relation) -> List[Entity]:
        """
        Get related entities for the given entity from the knowledge graph.
        
        Args:
            entity: The entity to explore.
            
        Returns:
            A list of candidate entities.
        """
        # Placeholder for related entity retrieval logic
        # This should be implemented in subclasses
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

        # get the prompt for pruning entities from prompts dir using the prompt loader
        # pass the prompt to the llm and get the response
        # parse the response and return the pruned entities
        return entities
    
class Neo4jEntityExplorer(EntityExplorer):
    """
    Entity explorer for Neo4j knowledge graph.
    """
    
    def _get_related_entities(self, relation: Relation) -> List[Entity]:
        """
        Get related entities for the given entity from the Neo4j knowledge graph.
        
        Args:
            entity: The entity to explore.
            
        Returns:
            A list of candidate entities.
        """
        # Implement related entity retrieval logic using Neo4j

        # write a cypher query to get the related entities from Neo4j
        # pass the query to the kg and get the response
        # convert the entities to the Entity objects
        # return the list of entities
        pass

if __name__ == "__main__":
    # Example usage
    from tog.llms import GroqLLM
    from tog.kgs import Neo4jKnowledgeGraph
    
    llm = GroqLLM()
    kg = Neo4jKnowledgeGraph()

    query = "Who is the CEO of Google?"
    prompt = "Find the CEO of {company}."
    prompt_params = {"company": "Google"}

    relation_explorer = Neo4jRelationExplorer(llm, kg, query, prompt, prompt_params)
    entity = Entity(id="3be66527971910fae63df4a4342ba4e0", name="Google", type="Company")
    relations = relation_explorer.explore_relations(entity)