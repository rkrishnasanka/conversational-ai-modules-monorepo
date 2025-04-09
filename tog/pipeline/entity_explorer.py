import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from tog.models.entity import Entity
from tog.models.relation import Relation
from tog.models.triple import Triple

class EntityExplorer(ABC):
    """
    Abstract base class for entity exploration in a knowledge graph.
    Responsible for discovering and ranking entities connected through relations.
    """
    
    def __init__(self, 
                 llm: BaseLLM, 
                 kg: KnowledgeGraph, 
                 query: str, 
                 max_entities_per_round: int = 3,
                 system_prompt: str = None):
        """
        Initialize the EntityExplorer with a language model, knowledge graph, and query.
        
        Args:
            llm: The language model to use for exploration and ranking
            kg: The knowledge graph to explore
            query: The query string for exploration guidance
            max_entities_per_round: Maximum number of entities to keep per exploration round
            system_prompt: System prompt for the LLM
        """
        self.llm = llm
        self.kg = kg
        self.query = query
        self.max_entities_per_round = max_entities_per_round
        self.system_prompt = system_prompt or "You are a helpful assistant specialized in entity analysis."
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def explore_entities(self, entity: Entity, relations: List[Relation]) -> List[Tuple[Entity, Relation, Entity]]:
        """
        Explore entities connected to the topic entity through the selected relations.
        
        Args:
            entity: The entity to explore from
            relations: The relations to follow for exploration
            
        Returns:
            A list of tuples (source_entity, relation, target_entity)
        """
        self.logger.info(f"Starting entity exploration for '{entity.name}'")
        
        # Step 1: Find all connected entities
        entity_relation_tuples = self._discover_connected_entities(entity, relations)
        self.logger.info(f"Found {len(entity_relation_tuples)} entity-relation tuples")
        
        # Step 2: Select the most relevant entities
        pruned_tuples = self._batch_prune_entities(entity_relation_tuples)
        self.logger.info(f"Pruned to {len(pruned_tuples)} entity-relation tuples")
        
        return pruned_tuples
    
    @abstractmethod
    def _discover_connected_entities(self, entity: Entity, relations: List[Relation]) -> List[Tuple[Entity, Relation, Entity]]:
        """
        Discover entities connected to the source entity through the given relations.
        
        Args:
            entity: The source entity
            relations: The relations to follow
            
        Returns:
            A list of tuples (source_entity, relation, target_entity)
        """
        pass
    
    def _batch_prune_entities(self, entity_relation_tuples: List[Tuple[Entity, Relation, Entity]]) -> List[Tuple[Entity, Relation, Entity]]:
        """
        Prune entities based on their relevance to the query using batch processing.
        
        Args:
            entity_relation_tuples: List of (source, relation, target) tuples
            
        Returns:
            A pruned list of tuples
        """
        if not entity_relation_tuples:
            self.logger.debug("No entity-relation tuples to prune")
            return []
        
        try:
            # Format tuples for LLM
            tuples_text = ""
            for i, (source, relation, target) in enumerate(entity_relation_tuples, 1):
                source_desc = f"{source.name} ({source.type})"
                relation_desc = relation.type
                target_desc = f"{target.name} ({target.type})"
                
                # Add descriptions if available
                if source.metadata and "description" in source.metadata:
                    source_desc += f": {source.metadata['description']}"
                if relation.metadata and "description" in relation.metadata:
                    relation_desc += f": {relation.metadata['description']}"
                if target.metadata and "description" in target.metadata:
                    target_desc += f": {target.metadata['description']}"
                
                tuples_text += f"{i}. {source_desc} -[{relation_desc}]-> {target_desc}\n"
            
            # Prepare prompt for ranking
            prompt = f"""
            Please analyze the following knowledge graph triples and rank them based on their relevance to the query.
            
            QUERY: {self.query}
            
            KNOWLEDGE GRAPH TRIPLES:
            {tuples_text}
            
            For each triple, assign a relevance score between 0.0 and 1.0, where 1.0 is highly relevant and 0.0 is not relevant at all.
            Consider how useful each triple would be in answering the query.
            
            Return your response as a JSON object where the keys are the indexes of the triples and the values are their relevance scores.
            Example format:
            {{
                "1": 0.9,
                "2": 0.7,
                "3": 0.3
            }}
            
            Return only the JSON object, no additional explanations.
            """
            
            # Call LLM for scoring
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.generate(messages, temperature=0.2)
            
            # Parse LLM response to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    scores_dict = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse LLM response as JSON: {response}")
                    return entity_relation_tuples[:self.max_entities_per_round]
            else:
                self.logger.error(f"No JSON found in LLM response: {response}")
                return entity_relation_tuples[:self.max_entities_per_round]
            
            # Assign scores to entities
            for idx_str, score in scores_dict.items():
                try:
                    idx = int(idx_str) - 1  # Convert to 0-based index
                    if 0 <= idx < len(entity_relation_tuples):
                        _, relation, target = entity_relation_tuples[idx]
                        target.metadata["relevance_score"] = float(score)
                        relation.metadata["relevance_score"] = float(score)  # Also score the relation
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error processing score for index {idx_str}: {e}")
            
            # Sort by score in descending order
            scored_tuples = sorted(
                entity_relation_tuples,
                key=lambda t: t[2].metadata.get("relevance_score", 0.0),
                reverse=True
            )
            
            # Return top N tuples
            return scored_tuples[:self.max_entities_per_round]
            
        except Exception as e:
            self.logger.error(f"Error in batch pruning entities: {e}")
            return entity_relation_tuples[:self.max_entities_per_round]


class Neo4jEntityExplorer(EntityExplorer):
    """
    Entity explorer implementation for Neo4j knowledge graph.
    """
    
    def _discover_connected_entities(self, entity: Entity, relations: List[Relation]) -> List[Tuple[Entity, Relation, Entity]]:
        """
        Discover entities connected to the source entity through the given relations using Neo4j.
        
        Args:
            entity: The source entity
            relations: The relations to follow
            
        Returns:
            A list of tuples (source_entity, relation, target_entity)
        """
        entity_relation_tuples = []
        
        try:
            # Collect relation IDs for batch query
            relation_ids = [relation.id for relation in relations]
            
            # Query to find all connected entities through the specified relations
            # This handles both directions: entity as source or as target
            query = """
            MATCH (e)-[r]->(target)
            WHERE e.id = $entity_id AND r.id IN $relation_ids
            RETURN 
                e.id as source_id,
                e.name as source_name,
                labels(e)[0] as source_type,
                properties(e) as source_properties,
                r.id as relation_id,
                type(r) as relation_type,
                properties(r) as relation_properties,
                target.id as target_id,
                target.name as target_name,
                labels(target)[0] as target_type,
                properties(target) as target_properties
            UNION
            MATCH (source)-[r]->(e)
            WHERE e.id = $entity_id AND r.id IN $relation_ids
            RETURN 
                source.id as source_id,
                source.name as source_name,
                labels(source)[0] as source_type,
                properties(source) as source_properties,
                r.id as relation_id,
                type(r) as relation_type,
                properties(r) as relation_properties,
                e.id as target_id,
                e.name as target_name,
                labels(e)[0] as target_type,
                properties(e) as target_properties
            """
            
            # Execute the query
            self.logger.debug(f"Executing Neo4j query for entity ID: {entity.id}")
            results = self.kg.query(query, entity_id=entity.id, relation_ids=relation_ids)
            
            self.logger.debug(f"Found {len(results)} connected entities")
            
            # Process results
            for result in results:
                # Create source entity
                source_entity = Entity(
                    id=result["source_id"],
                    name=result["source_name"],
                    type=result["source_type"],
                    metadata={k: v for k, v in result.get("source_properties", {}).items() 
                              if k not in ["id", "name"]}
                )
                
                # Find matching relation from the provided list
                relation = None
                relation_id = result["relation_id"]
                for r in relations:
                    if r.id == relation_id:
                        relation = r
                        break
                
                # If relation not found, create it from result
                if not relation:
                    relation = Relation(
                        id=relation_id,
                        source_id=result["source_id"],
                        target_id=result["target_id"],
                        type=result["relation_type"],
                        metadata={k: v for k, v in result.get("relation_properties", {}).items() 
                                  if k not in ["id"]}
                    )
                
                # Create target entity
                target_entity = Entity(
                    id=result["target_id"],
                    name=result["target_name"],
                    type=result["target_type"],
                    metadata={k: v for k, v in result.get("target_properties", {}).items() 
                              if k not in ["id", "name"]}
                )
                
                # Add the tuple to the list
                entity_relation_tuples.append((source_entity, relation, target_entity))
            
            return entity_relation_tuples
            
        except Exception as e:
            self.logger.error(f"Error discovering connected entities: {e}")
            return []