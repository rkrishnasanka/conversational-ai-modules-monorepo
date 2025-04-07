import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import json
import re

from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from tog.models.entity import Entity
from tog.models.relation import Relation

class Explorer:
    """
    Abstract base class for exploring entities and relations in a knowledge graph.
    """
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, 
                 prompt_params: Dict[str, str] = None, system_prompt: str = None):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prune_prompt = prune_prompt
        self.prompt_params = prompt_params or {}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.logger = logging.getLogger(self.__class__.__name__)


class EntityExplorer(Explorer):
    """
    Class for entity exploration in a knowledge graph following the ToG approach.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prune_prompt: str, 
                 prompt_params: Dict[str, str] = None, max_entities_per_round: int = 3):
        """
        Initialize the EntityExplorer with a language model, knowledge graph, and query.
        
        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): The query string for exploration.
            prune_prompt (str): The prompt for pruning entities.
            prompt_params (Dict[str, str], optional): Parameters for the prompt.
            max_entities_per_round (int): Maximum number of entities to keep per exploration round (W parameter).
        """
        super().__init__(llm, kg, query, prune_prompt, prompt_params)
        self.max_entities_per_round = max_entities_per_round
    
    def explore_entities(self, topic_entity: Entity, selected_relations: List[Relation]) -> List[Entity]:
        """
        Explore entities connected to the topic entity through the selected relations.
        
        Args:
            topic_entity: The entity to explore from.
            selected_relations: The relations to follow for exploration.
            
        Returns:
            A list of discovered and pruned entities.
        """
        self.logger.debug(f"Exploring entities connected to {topic_entity.name} through {len(selected_relations)} relations")
        
        # Step 1: Entity Discovery - Find all connected entities
        candidate_entities = self._get_connected_entities(topic_entity, selected_relations)
        self.logger.debug(f"Found {len(candidate_entities)} candidate entities for {topic_entity.name}")
        
        # Step 2: Context Retrieval - Get contexts for candidate entities
        # Note: In a full implementation, this would involve retrieving documents/chunks
        # But for this implementation, we'll skip the actual document retrieval
        
        # Step 3: Entity Pruning - Select the most relevant entities
        pruned_entities = self._prune_entities(candidate_entities, topic_entity)
        self.logger.debug(f"Pruned to {len(pruned_entities)} entities for {topic_entity.name}")
        
        return pruned_entities
    
    def _get_connected_entities(self, entity: Entity, relations: List[Relation]) -> List[Entity]:
        """
        Get entities connected to the given entity through the specified relations.
        
        Args:
            entity: The source entity.
            relations: The relations to follow.
            
        Returns:
            A list of connected entities.
        """
        connected_entities = []
        
        for relation in relations:
            # Determine if the given entity is the source or target in this relation
            is_source = relation.source_id == entity.id
            is_target = relation.target_id == entity.id
            
            if is_source:
                # The entity is the source, so we need to find the target entity
                connected_entity = self._get_entity_by_id(relation.target_id)
                if connected_entity:
                    # Add relation metadata to the entity for later use
                    connected_entity.metadata["source_relation"] = relation.type
                    connected_entity.metadata["relation_direction"] = "outgoing"
                    connected_entities.append(connected_entity)
            
            elif is_target:
                # The entity is the target, so we need to find the source entity
                connected_entity = self._get_entity_by_id(relation.source_id)
                if connected_entity:
                    # Add relation metadata to the entity for later use
                    connected_entity.metadata["source_relation"] = relation.type
                    connected_entity.metadata["relation_direction"] = "incoming"
                    connected_entities.append(connected_entity)
        
        return connected_entities
    
    @abstractmethod
    def _get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID from the knowledge graph.
        
        Args:
            entity_id: The ID of the entity to retrieve.
            
        Returns:
            The entity object if found, None otherwise.
        """
        pass

    def _prune_entities(self, entities: List[Entity], topic_entity: Entity) -> List[Entity]:
        """
        Prune the list of entities based on relevance to the query.
        
        Args:
            entities: The list of candidate entities to prune.
            topic_entity: The original topic entity to provide context.
            
        Returns:
            A pruned list of entities.
        """
        if not entities:
            self.logger.debug("No entities to prune")
            return []
        
        try:
            # Create a mapping from index to entity for easier reference
            entity_map = {i: entity for i, entity in enumerate(entities, 1)}
            
            # Format entities for the prompt
            entities_text = ""
            for i, entity in enumerate(entities, 1):
                entity_desc = f"{entity.name} ({entity.type})"
                if entity.metadata and "description" in entity.metadata:
                    entity_desc += f": {entity.metadata['description']}"
                if "source_relation" in entity.metadata:
                    rel_direction = "←" if entity.metadata.get("relation_direction") == "incoming" else "→"
                    entity_desc += f" [Relation: {entity.metadata['source_relation']} {rel_direction}]"
                entities_text += f"{i}. {entity_desc}\n"
            
            # Get the number of entities to select (default to max_entities_per_round or less if fewer entities)
            n = min(self.max_entities_per_round, len(entities))
            
            # Update prompt params with the topic entity name if not already set
            prompt_params = self.prompt_params.copy()
            if "entity_name" not in prompt_params:
                prompt_params["entity_name"] = topic_entity.name
                
            # Format the prompt with the prune_prompt template
            formatted_prompt = self.prune_prompt.format(
                query=self.query,
                n=n,
                entities=entities_text,
                **prompt_params
            )
            
            # Add explicit instruction for JSON format
            formatted_prompt += "\n\nPlease return your response as a valid JSON object with the following format:\n"
            formatted_prompt += "{\n"
            for i, entity in enumerate(entities, 1):
                if i <= n:
                    formatted_prompt += f'  "{entity.name}": score_value'
                    if i < n:
                        formatted_prompt += ","
                    formatted_prompt += "\n"
            formatted_prompt += "}\n"
            formatted_prompt += f"where score_value is a number between 0 and 1, and the sum of all scores equals 1."
            
            # Call LLM for scoring
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
            
            self.logger.debug(f"Sending pruning prompt to LLM: {formatted_prompt}")
            response = self.llm.generate(messages, temperature=0.3)
            self.logger.debug(f"LLM response: {response}")
            
            # Parse LLM response to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    self.logger.debug(f"Extracted JSON string: {json_str}")
                    scores_dict = json.loads(json_str)
                    self.logger.debug(f"Parsed scores: {scores_dict}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                    # Try a more aggressive approach to find valid JSON
                    try:
                        # Try to find anything that looks like a JSON object with entity names and scores
                        score_pairs = re.findall(r'"([^"]+)":\s*(0\.\d+|1\.0|1)', response)
                        if score_pairs:
                            scores_dict = {name: float(score) for name, score in score_pairs}
                            self.logger.debug(f"Extracted scores from regex: {scores_dict}")
                        else:
                            self.logger.error("Couldn't extract scores with regex either")
                            return entities[:n]  # Return top n if extraction fails
                    except Exception as e2:
                        self.logger.error(f"Alternative JSON extraction failed: {e2}")
                        return entities[:n]  # Return top n if parsing fails
            else:
                # Try to parse response as a numbered list with scores
                try:
                    score_pattern = r'(\d+)\.\s+([^:]+).*?score.*?(\d+\.\d+)'
                    score_matches = re.findall(score_pattern, response, re.IGNORECASE)
                    if score_matches:
                        scores_dict = {}
                        for idx_str, entity_name, score_str in score_matches:
                            idx = int(idx_str)
                            if idx in entity_map:
                                entity = entity_map[idx]
                                scores_dict[entity.name] = float(score_str)
                        self.logger.debug(f"Extracted scores from numbered list: {scores_dict}")
                    else:
                        self.logger.error(f"No JSON or numbered list found in LLM response: {response}")
                        return entities[:n]  # Return top n if no pattern found
                except Exception as e:
                    self.logger.error(f"Failed to extract scores from response: {e}")
                    return entities[:n]  # Return top n if extraction fails
            
            # Assign scores to entities - use more flexible matching
            for entity in entities:
                # Initialize score to 0
                score = 0.0
                entity_name = entity.name.lower()
                
                # Try direct match first
                if entity.name in scores_dict:
                    score = float(scores_dict[entity.name])
                else:
                    # Try more flexible matching
                    for name, value in scores_dict.items():
                        name_lower = name.lower()
                        # Check if entity name is in the score key or vice versa
                        if entity_name in name_lower or name_lower in entity_name:
                            score = float(value)
                            break
                
                # Add score to metadata
                entity.metadata["relevance_score"] = score
                self.logger.debug(f"Assigned score {score} to entity {entity.name}")
            
            # Sort by score in descending order
            scored_entities = sorted(entities, key=lambda e: e.metadata.get("relevance_score", 0.0), reverse=True)
            
            # Return top n entities
            return scored_entities[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning entities: {e}")
            # Return at most n entities if pruning fails
            return entities[:min(self.max_entities_per_round, len(entities))]


class Neo4jEntityExplorer(EntityExplorer):
    """
    Entity explorer for Neo4j knowledge graph.
    """
    
    def _get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID from the Neo4j knowledge graph.
        
        Args:
            entity_id: The ID of the entity to retrieve.
            
        Returns:
            The entity object if found, None otherwise.
        """
        try:
            # Query to find an entity by its ID
            query = """
            MATCH (e)
            WHERE e.id = $entity_id
            RETURN 
                e.id as id,
                e.name as name,
                labels(e)[0] as type,
                properties(e) as properties
            LIMIT 1
            """
            
            # Execute the query using the knowledge graph
            results = self.kg.query(query, entity_id=entity_id)
            
            if not results:
                self.logger.debug(f"No entity found with ID: {entity_id}")
                return None
            
            result = results[0]
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
            
            return entity
            
        except Exception as e:
            self.logger.error(f"Error getting entity with ID {entity_id}: {e}")
            return None
    
    # def get_connected_entities_batch(self, topic_entities: List[Entity], all_relations: List[List[Relation]]) -> List[Entity]:
    #     """
    #     Get connected entities for multiple topic entities in batch.
        
    #     Args:
    #         topic_entities: List of topic entities.
    #         all_relations: List of relations lists, one list per topic entity.
            
    #     Returns:
    #         A list of all connected entities.
    #     """
    #     all_connected_entities = []
        
    #     for i, entity in enumerate(topic_entities):
    #         if i < len(all_relations):
    #             entity_relations = all_relations[i]
    #             connected_entities = self._get_connected_entities(entity, entity_relations)
    #             all_connected_entities.extend(connected_entities)
        
    #     return all_connected_entities


if __name__ == "__main__":
    # Example usage
    import logging
    from tog.llms import AzureOpenAILLM
    from tog.kgs import Neo4jKnowledgeGraph
    from tog.models.entity import Entity
    from tog.models.relation import Relation
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize components
    llm = AzureOpenAILLM(model_name="gpt-35-turbo")
    kg = Neo4jKnowledgeGraph()

    # Define query and prompts
    query = "What are the medical benefits of Medical Cannabis?"
    
    # Entity pruning prompt - updated for clearer JSON instructions
    entity_prune_prompt = """
    Please analyze the following candidate entities to determine which {n} entities most contribute to answering the question. Rate each entity's contribution on a scale from 0 to 1, where the sum of the scores of the {n} entities equals 1.
    
    Question: {query}
    Topic Entity: {entity_name}
    Candidate Entities: 
    {entities}
    
    For the {n} most relevant entities, provide their names and scores.
    """
    
    # Define parameters
    prompt_params = {"entity_name": "Medical Cannabis"}
    
    # Initialize explorer
    entity_explorer = Neo4jEntityExplorer(
        llm=llm,
        kg=kg,
        query=query,
        prune_prompt=entity_prune_prompt,
        prompt_params=prompt_params,
        max_entities_per_round=3
    )
    
    # Create example topic entity
    topic_entity = Entity(id="192db73673d90090cf1cb7d1be13aebc", name="Chronic Pain", type="Medical Condition", 
                    metadata={"description": "A long-lasting pain that persists beyond the usual recovery period or accompanies a chronic health condition."})
    
    # Create some example relations
    selected_relations = [
        Relation(
            id="2a6ba87dd042a7b00030e8ca34808e9e",
            source_id="ac9c5fd77eab51efd41402be9e24cc2b",
            target_id="192db73673d90090cf1cb7d1be13aebc",
            type="Potential Treatment For",
            metadata={"description": "THC is researched for its potential to treat chronic pain.", "strength": 0.7}
        ),
        Relation(
            id="7da9bc343725edd3fe8d6c68fbb3d6ab",
            source_id="e25a5fa07488cf994c749288e930943c",
            target_id="192db73673d90090cf1cb7d1be13aebc",
            type="Potential Treatment For",
            metadata={"description": "CBD is researched for its potential to treat chronic pain.", "strength": 0.7}
        ),
        Relation(
            id="65dab97f5bde5196a6d0c175b95c63af",
            source_id="3be66527971910fae63df4a4342ba4e0",
            target_id="192db73673d90090cf1cb7d1be13aebc",
            type="Treats",
            metadata={"description": "Medical cannabis is used to alleviate symptoms associated with chronic pain.", "strength": 0.8}
        )
    ]
    
    # Explore connected entities
    connected_entities = entity_explorer.explore_entities(topic_entity, selected_relations)
    
    # Print results
    print(f"\nConnected entities for {topic_entity.name}:")
    for entity in connected_entities:
        print(f"- {entity.name} ({entity.type}), Score: {entity.metadata.get('relevance_score', 0.0):.2f}")
        print(f"  Relation: {entity.metadata.get('source_relation', 'Unknown')}")
        print(f"  Direction: {entity.metadata.get('relation_direction', 'Unknown')}")