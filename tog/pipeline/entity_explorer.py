from tog.models.entity import Entity
from tog.models.relation import Relation
from typing import List, Dict, Any
from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from abc import ABC, abstractmethod
from tog.utils.logger import setup_logger

class EntityExplorer(ABC):
    """
    Abstract base class for entity exploration in a knowledge graph.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prompt: str = None, prompt_params: Dict[str, str] = None):
        """
        Initialize the EntityExplorer with a language model and knowledge graph.
        
        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): Query string to guide exploration.
            prompt (str): Optional prompt template for LLM.
            prompt_params (Dict): Optional parameters for the prompt template.
        """
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prompt = prompt
        self.prompt_params = prompt_params or {}
        self.logger = setup_logger(name=self.__class__.__name__, log_filename=f"{self.__class__.__name__.lower()}.log")
    
    def explore_entities(self, relation: Relation) -> List[Entity]:
        """
        Get entities related through the given relation in the knowledge graph.
        
        Args:
            relation: The relation to explore through.
            
        Returns:
            A list of related entities.
        """
        candidate_entities: List[Entity] = self._get_related_entities(relation)
        self.logger.debug(f"Candidate entities for relation {relation.type}: {len(candidate_entities)}")
        
        pruned_entities: List[Entity] = self._prune_entities(candidate_entities)
        self.logger.debug(f"Pruned entities for relation {relation.type}: {len(pruned_entities)}")
        
        return pruned_entities
    
    @abstractmethod
    def _get_related_entities(self, relation: Relation) -> List[Entity]:
        """
        Get related entities for the given relation from the knowledge graph.
        
        Args:
            relation: The relation to explore through.
            
        Returns:
            A list of candidate entities.
        """
        pass

    def _prune_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Prune the list of entities based on relevance to the query using LLM.
        
        Args:
            entities: The list of candidate entities to prune.
            
        Returns:
            A pruned list of entities.
        """
        if not entities:
            return []
            
        if not self.query or not self.prompt:
            self.logger.warning("No query or prompt provided for entity pruning, returning all entities")
            return entities
            
        # Format entities for the prompt
        entities_text = ""
        for i, entity in enumerate(entities):
            entity_type = entity.type if entity.type else "Unknown"
            entities_text += f"{i+1}. Entity: {entity.name}, Type: {entity_type}, ID: {entity.id}\n"
        
        # Create prompt for LLM
        formatted_prompt = self.prompt.format(
            query=self.query,
            entities=entities_text,
            **self.prompt_params
        )
        
        # Get LLM response
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant for knowledge graph exploration."},
                {"role": "user", "content": formatted_prompt}
            ]
            response = self.llm.generate(messages)
            
            # Parse the response to get scored entities
            scored_entities = []
            current_entities = list(entities)  # Make a copy to preserve original order
            
            # Simple parsing logic - extract entity numbers and scores
            for line in response.split('\n'):
                if ":" in line and ("Entity" in line or "entity" in line):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            entity_num = int(''.join(filter(str.isdigit, parts[0]))) - 1  # Adjust for 0-based indexing
                            
                            # Extract score from the line
                            score_text = parts[1].split('-')[0].strip()
                            score = float(''.join(filter(lambda c: c.isdigit() or c == '.', score_text)))
                            
                            if 0 <= entity_num < len(current_entities) and score >= 7.0:  # Only keep high-scoring entities
                                entity = current_entities[entity_num]
                                entity.metadata["score"] = score
                                entity.metadata["selected"] = True
                                scored_entities.append(entity)
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Error parsing entity from LLM response: {line}, Error: {e}")
            
            # Sort by score
            scored_entities.sort(key=lambda e: e.metadata.get("score", 0), reverse=True)
            
            self.logger.info(f"Pruned {len(entities)} entities to {len(scored_entities)} using LLM")
            return scored_entities
            
        except Exception as e:
            self.logger.error(f"Error in LLM pruning: {e}")
            return entities  # Return all entities if pruning fails
    
class Neo4jEntityExplorer(EntityExplorer):
    """
    Entity explorer for Neo4j knowledge graph.
    """
    
    def _get_related_entities(self, relation: Relation) -> List[Entity]:
        """
        Get related entities for the given relation from the Neo4j knowledge graph.
        
        Args:
            relation: The relation to explore through.
            
        Returns:
            A list of candidate entities.
        """
        try:
            # Determine which entity ID we're looking for (the one that's not the source)
            source_id = relation.subject_id
            target_id = relation.object_id
            
            # Query to get details of the target entity
            query = """
            MATCH (e)
            WHERE e.id = $entity_id
            RETURN 
                e.id as id,
                e.name as name,
                labels(e) as types,
                properties(e) as properties
            """
            
            # Execute the query using the knowledge graph
            results = self.kg.query(query, entity_id=target_id)
            
            entities = []
            for result in results:
                if not result.get("id"):
                    self.logger.warning(f"Missing entity ID in query result")
                    continue
                
                # Get the entity type (use first label as primary type)
                entity_type = result.get("types", ["Entity"])[0] if result.get("types") else "Entity"
                
                # Create metadata with relation information
                metadata = result.get("properties", {})
                metadata["source_relation"] = relation.type
                metadata["source_entity_id"] = source_id
                metadata["relation_id"] = relation.id
                metadata["relation_direction"] = relation.metadata.get("direction", "unknown")
                
                # Create Entity object
                entity = Entity(
                    id=result["id"],
                    name=result.get("name", f"Entity-{result['id']}"),
                    type=entity_type,
                    metadata=metadata
                )
                entities.append(entity)
            
            self.logger.info(f"Found {len(entities)} entities for relation {relation.type} (ID: {relation.id})")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error querying Neo4j for entities: {str(e)}")
            return []