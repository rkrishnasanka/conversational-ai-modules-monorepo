from abc import ABC, abstractmethod
from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from typing import Dict, List
import logging
from tog.models.entity import Entity
from tog.models.relation import Relation

class Explorer:
    """
    Abstract base class for exploring entities and relations in a knowledge graph.
    """
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, prompt_params: Dict[str, str] = None, system_prompt: str = None):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prune_prompt = prune_prompt
        self.prompt_params = prompt_params or {}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.logger = logging.getLogger(self.__class__.__name__)


class RelationExplorer(Explorer, ABC):
    """
    Abstract base class for relation explorers.
    This class defines the interface for exploring relations in a knowledge graph.
    """

    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, prompt_params: Dict[str, str] = None):
        """
        Initialize the RelationExplorer with a language model, knowledge graph, and optional query and prompt.

        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): Optional query string for exploration.
            prune_prompt (str): Optional prompt string for pruning candidates.
            prompt_params (dict): Optional parameters for the prompt.
        """
        super().__init__(llm, kg, query, prune_prompt, prompt_params)

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
        Prune the candidate relations using LLM to score their relevance to the query.

        Args:
            relations (List[Relation]): The list of candidate relations to prune.

        Returns:
            List[Relation]: A list of pruned relations ordered by relevance score.
        """
        if not relations:
            self.logger.debug("No relations to prune")
            return []
        
        try:
            # Format relations for the prompt
            relations_text = ""
            
            for i, relation in enumerate(relations, 1):
                # Determine direction description
                if relation.metadata.get("is_incoming", False):
                    direction = f"{relation.metadata.get('source_name', 'Unknown')} → {relation.metadata.get('target_name', 'Unknown')}"
                else:
                    direction = f"{relation.metadata.get('source_name', 'Unknown')} → {relation.metadata.get('target_name', 'Unknown')}"
                
                relations_text += f"{i}. Relation: {relation.type}, Direction: {direction}\n"
            
            # Get the number of relations to select (default to 3 or less if fewer relations)
            n = min(3, len(relations))
            
            # Format the prompt
            prompt = f"""
            Please retrieve {n} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {n} relations is 1). Return the result in valid JSON format.
            
            Q: {self.query}
            Topic Entity: {self.prompt_params.get('entity_name', 'Unknown Entity')}
            Relations: 
            {relations_text}
            """
            
            # Call LLM for scoring
            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in analyzing semantic relations between entities and questions. Your task is to evaluate which relations are most relevant to answering a given question about a topic entity."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.generate(messages, temperature=0.3)
            
            # Parse LLM response to extract JSON
            import json
            import re
            
            # Extract JSON from response 
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    scores_dict = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse LLM response as JSON: {response}")
                    return relations[:n]  # Return top n if parsing fails
            else:
                self.logger.error(f"No JSON found in LLM response: {response}")
                return relations[:n]  # Return top n if no JSON found
            
            # Assign scores to relations
            scored_relations = []
            for relation in relations:
                # Find if this relation type is in the scores
                score = 0.0
                for rel_type, rel_score in scores_dict.items():
                    if relation.type.lower() in rel_type.lower():
                        score = float(rel_score)
                        break
                
                # Add score to metadata
                relation.metadata["relevance_score"] = score
                scored_relations.append(relation)
            
            # Sort by score in descending order
            scored_relations.sort(key=lambda r: r.metadata.get("relevance_score", 0.0), reverse=True)
            
            # Return top n relations
            return scored_relations[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning relations: {e}")
            # Return at most n relations if pruning fails
            return relations[:min(3, len(relations))]
        
class Neo4jRelationExplorer(RelationExplorer):
    """
    A concrete implementation of RelationExplorer for Neo4j knowledge graph.
    """
    
    def get_candidates(self, entity: Entity) -> List[Relation]:
        """
        Get candidate relations for the given entity using Neo4j knowledge graph.
        Retrieves both incoming and outgoing relations.

        Args:
            entity (Entity): The entity to get candidates for.

        Returns:
            List[Relation]: A list of candidate relations.
        """
        try:
            self.logger.debug(f"Getting candidate relations for entity: {entity.name} (ID: {entity.id})")
            
            # Query to find all relationships where the given entity is either the source (outgoing)
            query_outgoing = """
            MATCH (source)-[r]->(target)
            WHERE source.id = $entity_id
            RETURN 
                source.id as source_id,
                source.name as source_name, 
                target.id as target_id,
                target.name as target_name,
                type(r) as relation_type,
                r.id as relation_id,
                properties(r) as metadata,
                false as is_incoming
            """
            
            # Query to find all relationships where the given entity is the target (incoming)
            query_incoming = """
            MATCH (source)-[r]->(target)
            WHERE target.id = $entity_id
            RETURN 
                source.id as source_id,
                source.name as source_name,
                target.id as target_id,
                target.name as target_name,
                type(r) as relation_type,
                r.id as relation_id,
                properties(r) as metadata,
                true as is_incoming
            """
            
            # Execute the queries using the knowledge graph
            outgoing_results = self.kg.query(query_outgoing, entity_id=entity.id)
            incoming_results = self.kg.query(query_incoming, entity_id=entity.id)
            
            # Combine results
            all_results = outgoing_results + incoming_results
            
            # Convert the query results to Relation objects
            relations = []
            for result in all_results:
                relation_id = str(result.get("relation_id", ""))
                
                # If relation_id is not found, generate one
                if not relation_id:
                    # Generate a unique ID based on the entities and relation type
                    relation_id = f"rel_{result['source_id']}_{result['relation_type']}_{result['target_id']}"
                
                # Determine source and target based on direction
                is_incoming = result.get("is_incoming", False)
                source_id = result["source_id"]
                target_id = result["target_id"]
                
                # Get metadata from result
                metadata = result.get("metadata", {})
                
                # Add additional information to metadata
                metadata["source_name"] = result.get("source_name", "")
                metadata["target_name"] = result.get("target_name", "")
                metadata["is_incoming"] = is_incoming
                
                # Create a Relation object
                relation = Relation(
                    id=relation_id,
                    source_id=source_id,
                    target_id=target_id,
                    type=result["relation_type"],
                    metadata=metadata
                )
                relations.append(relation)
            
            self.logger.debug(f"Found {len(relations)} candidate relations for entity: {entity.name}")
            return relations
            
        except Exception as e:
            self.logger.error(f"Error getting candidate relations for entity {entity.name}: {e}")
            return []

class EntityExplorer(Explorer):
    """
    Abstract base class for entity exploration in a knowledge graph.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, prompt_params: Dict[str, str] = None):
        """
        Initialize the EntityExplorer with a language model, knowledge graph, and query.
        
        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): The query string for exploration.
            prune_prompt (str): The prompt for pruning entities.
            prompt_params (Dict[str, str], optional): Parameters for the prompt.
        """
        super().__init__(llm, kg, query, prune_prompt, prompt_params)
    
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
        if not entities:
            self.logger.debug("No entities to prune")
            return []
        
        try:
            # Format entities for the prompt
            entities_text = ""
            for i, entity in enumerate(entities, 1):
                entity_desc = f"{entity.name} ({entity.type})"
                if entity.metadata and "description" in entity.metadata:
                    entity_desc += f": {entity.metadata['description']}"
                entities_text += f"{i}. {entity_desc}\n"
            
            # Get the number of entities to select (default to 3 or less if fewer entities)
            n = min(3, len(entities))
            
            # Format the prompt with the prune_prompt template
            formatted_prompt = self.prune_prompt.format(
                query=self.query,
                n=n,
                entities=entities_text,
                **self.prompt_params
            )
            
            # Call LLM for scoring
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=0.3)
            
            # Parse LLM response to extract JSON
            import json
            import re
            
            # Extract JSON from response 
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    scores_dict = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse LLM response as JSON: {response}")
                    return entities[:n]  # Return top n if parsing fails
            else:
                self.logger.error(f"No JSON found in LLM response: {response}")
                return entities[:n]  # Return top n if no JSON found
            
            # Assign scores to entities
            scored_entities = []
            for entity in entities:
                # Find if this entity is in the scores
                score = 0.0
                for entity_name, entity_score in scores_dict.items():
                    if entity.name.lower() in entity_name.lower():
                        score = float(entity_score)
                        break
                
                # Add score to metadata
                entity.metadata["relevance_score"] = score
                scored_entities.append(entity)
            
            # Sort by score in descending order
            scored_entities.sort(key=lambda e: e.metadata.get("relevance_score", 0.0), reverse=True)
            
            # Return top n entities
            return scored_entities[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning entities: {e}")
            # Return at most n entities if pruning fails
            return entities[:min(3, len(entities))]
    
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
        try:
            self.logger.debug(f"Getting related entities for entity: {entity.name} (ID: {entity.id})")
            
            # Query to find all entities connected to the given entity
            query = """
            MATCH (source)-[r]-(target)
            WHERE source.id = $entity_id AND target.id <> $entity_id
            RETURN DISTINCT
                target.id as id,
                target.name as name,
                labels(target)[0] as type,
                properties(target) as properties
            """
            
            # Execute the query using the knowledge graph
            results = self.kg.query(query, entity_id=entity.id)
            
            # Convert the query results to Entity objects
            entities = []
            for result in results:
                entity_id = result["id"]
                entity_name = result["name"]
                entity_type = result["type"]
                
                # Get properties from result
                properties = result.get("properties", {})
                
                # Create a metadata dictionary
                metadata = {k: v for k, v in properties.items() if k not in ["id", "name"]}
                
                # Create an Entity object
                entity = Entity(
                    id=entity_id,
                    name=entity_name,
                    type=entity_type,
                    metadata=metadata
                )
                entities.append(entity)
            
            self.logger.debug(f"Found {len(entities)} related entities for entity: {entity.name}")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error getting related entities for entity {entity.name}: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    from tog.llms import GroqLLM
    from tog.kgs import Neo4jKnowledgeGraph
    
    llm = GroqLLM()
    kg = Neo4jKnowledgeGraph()

    query = "Who is the CEO of Google?"
    prune_prompt = """
    Please retrieve {n} entities that contribute to answering the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {n} entities is 1). Return the result in valid JSON format.
    
    Q: {query}
    Topic Entity: {entity_name}
    Entities: 
    {entities}
    """
    prompt_params = {"entity_name": "Google"}
    
    # Example of relation exploration
    relation_explorer = Neo4jRelationExplorer(llm, kg, query, prune_prompt, prompt_params)
    entity = Entity(id="3be66527971910fae63df4a4342ba4e0", name="Patients", type="Demographic Group", 
                    metadata={"description": "Individuals participating in the survey about medical cannabis use."})
    relations = relation_explorer.explore_relations(entity)
    
    print(f"Relations for {entity.name}:")
    print("length:", len(relations))
    for relation in relations:
        print(f"- {relation.type} (ID: {relation.id})")
    
    # # Example of entity exploration
    # entity_explorer = Neo4jEntityExplorer(llm, kg, query, prune_prompt, prompt_params)
    # related_entities = entity_explorer.explore_entities(entity)
    
    # print(f"\nRelated entities for {entity.name}:")
    # print("length:", len(related_entities))
    # for related_entity in related_entities:
    #     print(f"- {related_entity.name} ({related_entity.type})")