from abc import ABC, abstractmethod
from typing import Dict, List
import logging
import json
import re

from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from tog.models.entity import Entity
from tog.models.relation import Relation

class Explorer:
    """Abstract base class for exploring entities and relations in a knowledge graph."""
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, 
                 prompt_params: Dict[str, str] = None, system_prompt: str = None):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prune_prompt = prune_prompt
        self.prompt_params = prompt_params or {}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.logger = logging.getLogger(self.__class__.__name__)


class RelationExplorer(Explorer, ABC):
    """Abstract base class for relation explorers."""

    def explore_relations(self, entity: Entity) -> List[Relation]:
        """Explore relations associated with the given entity."""
        candidate_relations = self.get_candidates(entity)
        return self.prune_candidates(candidate_relations)
    
    @abstractmethod
    def get_candidates(self, entity: Entity) -> List[Relation]:
        """Get candidate relations for the given entity."""
        pass

    def prune_candidates(self, relations: List[Relation]) -> List[Relation]:
        """Prune candidate relations using LLM to score their relevance."""
        if not relations:
            self.logger.debug("No relations to prune")
            return []
        
        try:
            # Format relations for the prompt
            relations_text = self._format_relations_text(relations)
            n = min(3, len(relations))
            
            # Get LLM response
            prompt = self._create_relations_prompt(relations_text, n)
            scores_dict = self._get_llm_scores(prompt)
            
            # Score and sort relations
            scored_relations = self._score_relations(relations, scores_dict)
            return scored_relations[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning relations: {e}")
            return relations[:min(3, len(relations))]
    
    def _format_relations_text(self, relations: List[Relation]) -> str:
        """Format relations for the prompt."""
        relations_text = ""
        for i, relation in enumerate(relations, 1):
            is_incoming = relation.metadata.get("is_incoming", False)
            source = relation.metadata.get('source_name', 'Unknown')
            target = relation.metadata.get('target_name', 'Unknown')
            direction = f"{source} → {target}"
            relations_text += f"{i}. Relation: {relation.type}, Direction: {direction}\n"
        return relations_text
    
    def _create_relations_prompt(self, relations_text: str, n: int) -> str:
        """Create prompt for relation scoring."""
        return f"""
        Please retrieve {n} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {n} relations is 1). Return the result in valid JSON format.
        
        Q: {self.query}
        Topic Entity: {self.prompt_params.get('entity_name', 'Unknown Entity')}
        Relations: 
        {relations_text}
        """
    
    def _get_llm_scores(self, prompt: str) -> Dict:
        """Get scores from LLM."""
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing semantic relations between entities and questions."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages, temperature=0.3)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse LLM response as JSON")
                return {}
        else:
            self.logger.error(f"No JSON found in LLM response")
            return {}
    
    def _score_relations(self, relations: List[Relation], scores_dict: Dict) -> List[Relation]:
        """Score relations based on LLM response."""
        scored_relations = []
        for relation in relations:
            score = 0.0
            for rel_type, rel_score in scores_dict.items():
                if relation.type.lower() in rel_type.lower():
                    score = float(rel_score)
                    break
            
            relation.metadata["relevance_score"] = score
            scored_relations.append(relation)
        
        scored_relations.sort(key=lambda r: r.metadata.get("relevance_score", 0.0), reverse=True)
        return scored_relations
        

class Neo4jRelationExplorer(RelationExplorer):
    """A concrete implementation of RelationExplorer for Neo4j knowledge graph."""
    
    def get_candidates(self, entity: Entity) -> List[Relation]:
        """Get candidate relations for the given entity using Neo4j knowledge graph."""
        try:
            self.logger.debug(f"Getting candidate relations for entity: {entity.name} (ID: {entity.id})")
            
            outgoing_results = self._query_outgoing_relations(entity.id)
            incoming_results = self._query_incoming_relations(entity.id)
            all_results = outgoing_results + incoming_results
            
            relations = self._convert_results_to_relations(all_results)
            
            self.logger.debug(f"Found {len(relations)} candidate relations for entity: {entity.name}")
            return relations
            
        except Exception as e:
            self.logger.error(f"Error getting candidate relations for entity {entity.name}: {e}")
            return []
    
    def _query_outgoing_relations(self, entity_id: str):
        """Query outgoing relations."""
        query = """
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
        return self.kg.query(query, entity_id=entity_id)
    
    def _query_incoming_relations(self, entity_id: str):
        """Query incoming relations."""
        query = """
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
        return self.kg.query(query, entity_id=entity_id)
    
    def _convert_results_to_relations(self, results):
        """Convert query results to Relation objects."""
        relations = []
        for result in results:
            relation_id = str(result.get("relation_id", ""))
            
            # Generate ID if not found
            if not relation_id:
                relation_id = f"rel_{result['source_id']}_{result['relation_type']}_{result['target_id']}"
            
            metadata = result.get("metadata", {})
            metadata.update({
                "source_name": result.get("source_name", ""),
                "target_name": result.get("target_name", ""),
                "is_incoming": result.get("is_incoming", False)
            })
            
            relation = Relation(
                id=relation_id,
                source_id=result["source_id"],
                target_id=result["target_id"],
                type=result["relation_type"],
                metadata=metadata
            )
            relations.append(relation)
        
        return relations


class EntityExplorer(Explorer):
    """Abstract base class for entity exploration in a knowledge graph."""
    
    def explore_entities(self, entity: Entity) -> List[Entity]:
        """Get entities related to the given entity in the knowledge graph."""
        candidate_entities = self._get_related_entities(entity)
        self.logger.debug(f"Candidate entities for {entity.name}: {candidate_entities}")
        
        pruned_entities = self._prune_entities(candidate_entities)
        self.logger.debug(f"Pruned entities for {entity.name}: {pruned_entities}")
        
        return pruned_entities
    
    @abstractmethod
    def _get_related_entities(self, entity: Entity) -> List[Entity]:
        """Get related entities for the given entity from the knowledge graph."""
        pass

    def _prune_entities(self, entities: List[Entity]) -> List[Entity]:
        """Prune the list of entities based on certain criteria."""
        if not entities:
            self.logger.debug("No entities to prune")
            return []
        
        try:
            entities_text = self._format_entities_text(entities)
            n = min(3, len(entities))
            
            # Get LLM response
            formatted_prompt = self._create_entities_prompt(entities_text, n)
            scores_dict = self._get_llm_scores(formatted_prompt)
            
            # Score and sort entities
            scored_entities = self._score_entities(entities, scores_dict)
            return scored_entities[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning entities: {e}")
            return entities[:min(3, len(entities))]
    
    def _format_entities_text(self, entities: List[Entity]) -> str:
        """Format entities for the prompt."""
        entities_text = ""
        for i, entity in enumerate(entities, 1):
            entity_desc = f"{entity.name} ({entity.type})"
            if entity.metadata and "description" in entity.metadata:
                entity_desc += f": {entity.metadata['description']}"
            entities_text += f"{i}. {entity_desc}\n"
        return entities_text
    
    def _create_entities_prompt(self, entities_text: str, n: int) -> str:
        """Create prompt for entity scoring."""
        return self.prune_prompt.format(
            query=self.query,
            n=n,
            entities=entities_text,
            **self.prompt_params
        )
    
    def _get_llm_scores(self, prompt: str) -> Dict:
        """Get scores from LLM."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages, temperature=0.3)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON")
                return {}
        else:
            self.logger.error("No JSON found in LLM response")
            return {}
    
    def _score_entities(self, entities: List[Entity], scores_dict: Dict) -> List[Entity]:
        """Score entities based on LLM response."""
        scored_entities = []
        for entity in entities:
            score = 0.0
            for entity_name, entity_score in scores_dict.items():
                if entity.name.lower() in entity_name.lower():
                    score = float(entity_score)
                    break
            
            entity.metadata["relevance_score"] = score
            scored_entities.append(entity)
        
        scored_entities.sort(key=lambda e: e.metadata.get("relevance_score", 0.0), reverse=True)
        return scored_entities


class Neo4jEntityExplorer(EntityExplorer):
    """Entity explorer for Neo4j knowledge graph."""
    
    def _get_related_entities(self, entity: Entity) -> List[Entity]:
        """Get related entities for the given entity from the Neo4j knowledge graph."""
        try:
            self.logger.debug(f"Getting related entities for entity: {entity.name} (ID: {entity.id})")
            
            results = self._query_related_entities(entity.id)
            entities = self._convert_results_to_entities(results)
            
            self.logger.debug(f"Found {len(entities)} related entities for entity: {entity.name}")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error getting related entities for entity {entity.name}: {e}")
            return []
    
    def _query_related_entities(self, entity_id: str):
        """Query related entities."""
        query = """
        MATCH (source)-[r]-(target)
        WHERE source.id = $entity_id AND target.id <> $entity_id
        RETURN DISTINCT
            target.id as id,
            target.name as name,
            labels(target)[0] as type,
            properties(target) as properties
        """
        return self.kg.query(query, entity_id=entity_id)
    
    def _convert_results_to_entities(self, results):
        """Convert query results to Entity objects."""
        entities = []
        for result in results:
            metadata = {k: v for k, v in result.get("properties", {}).items() 
                      if k not in ["id", "name"]}
            
            entity = Entity(
                id=result["id"],
                name=result["name"],
                type=result["type"],
                metadata=metadata
            )
            entities.append(entity)
        
        return entities


if __name__ == "__main__":
    # Example usage
    import logging
    from tog.llms import AzureOpenAILLM
    from tog.kgs import Neo4jKnowledgeGraph
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    llm = AzureOpenAILLM(model_name="gpt-35-turbo")
    kg = Neo4jKnowledgeGraph()

    query = "What are the medical benefits of Medical Cannabis?"
    prune_prompt = """
    Please retrieve {n} entities that contribute to answering the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {n} entities is 1). Return the result in valid JSON format.
    
    Q: {query}
    Topic Entity: {entity_name}
    Entities: 
    {entities}
    """
    prompt_params = {"entity_name": "Chronic Pain"}
    
    # Example of relation exploration
    relation_explorer = Neo4jRelationExplorer(llm, kg, query, prune_prompt, prompt_params)
    entity = Entity(id="192db73673d90090cf1cb7d1be13aebc", name="Chronic Pain", type="Medical Condition", 
                    metadata={"description": "A long-lasting pain that persists beyond the usual recovery period or accompanies a chronic health condition."})
    relations = relation_explorer.explore_relations(entity)
    
    print(f"Relations for {entity.name}:")
    print("length:", len(relations))
    for relation in relations:
        print(f"- {relation.type} (ID: {relation.id})")