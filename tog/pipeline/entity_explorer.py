import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import json
import re

from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from tog.models.entity import Entity
from tog.models.relation import Relation


# Configure logging to filter out noisy libraries
def configure_logging():
    """Configure logging to show only application logs and filter out noisy libraries."""
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Add console handler with formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Silence noisy libraries
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Set explorers to DEBUG level
    logging.getLogger('Explorer').setLevel(logging.DEBUG)
    logging.getLogger('EntityExplorer').setLevel(logging.DEBUG)
    logging.getLogger('Neo4jEntityExplorer').setLevel(logging.DEBUG)


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
        self.logger.info(f"Initialized EntityExplorer with query: '{query}' and max_entities_per_round: {max_entities_per_round}")
    
    def explore_entities(self, topic_entity: Entity, selected_relations: List[Relation]) -> List[Entity]:
        """
        Explore entities connected to the topic entity through the selected relations.
        
        Args:
            topic_entity: The entity to explore from.
            selected_relations: The relations to follow for exploration.
            
        Returns:
            A list of discovered and pruned entities.
        """
        self.logger.info(f"===== STEP 1: Starting entity exploration for '{topic_entity.name}' =====")
        self.logger.info(f"Using {len(selected_relations)} relations for exploration")
        
        # Step 1: Entity Discovery - Find all connected entities
        self.logger.info("===== STEP 2: Entity Discovery - Finding connected entities =====")
        candidate_entities = self._get_connected_entities(topic_entity, selected_relations)
        self.logger.info(f"Found {len(candidate_entities)} candidate entities for '{topic_entity.name}'")
        
        # Log found entities
        for i, entity in enumerate(candidate_entities, 1):
            relation_info = f"{entity.metadata.get('source_relation', 'Unknown')} ({entity.metadata.get('relation_direction', 'Unknown')})"
            self.logger.info(f"  Entity {i}: {entity.name} ({entity.type}) - Relation: {relation_info}")
        
        # Step 2: Context Retrieval - Get contexts for candidate entities
        self.logger.info("===== STEP 3: Context Retrieval - Getting contexts for entities =====")
        self.logger.info("Note: Skipping actual document retrieval in this implementation")
        
        # Step 3: Entity Pruning - Select the most relevant entities
        self.logger.info("===== STEP 4: Entity Pruning - Selecting most relevant entities =====")
        pruned_entities = self._prune_entities(candidate_entities, topic_entity)
        self.logger.info(f"Pruned to {len(pruned_entities)} entities for '{topic_entity.name}'")
        
        # Log pruned entities with scores
        for i, entity in enumerate(pruned_entities, 1):
            score = entity.metadata.get("relevance_score", 0.0)
            relation_info = f"{entity.metadata.get('source_relation', 'Unknown')} ({entity.metadata.get('relation_direction', 'Unknown')})"
            self.logger.info(f"  Selected Entity {i}: {entity.name} ({entity.type}) - Score: {score:.2f} - Relation: {relation_info}")
        
        self.logger.info("===== STEP 5: Entity Exploration Complete =====")
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
        self.logger.debug(f"Finding entities connected to '{entity.name}' through {len(relations)} relations")
        connected_entities = []
        
        for i, relation in enumerate(relations, 1):
            self.logger.debug(f"Processing relation {i}: {relation.type} (ID: {relation.id})")
            
            # Determine if the given entity is the source or target in this relation
            is_source = relation.source_id == entity.id
            is_target = relation.target_id == entity.id
            
            if is_source:
                # The entity is the source, so we need to find the target entity
                self.logger.debug(f"'{entity.name}' is source, retrieving target entity with ID: {relation.target_id}")
                connected_entity = self._get_entity_by_id(relation.target_id)
                if connected_entity:
                    # Add relation metadata to the entity for later use
                    connected_entity.metadata["source_relation"] = relation.type
                    connected_entity.metadata["relation_direction"] = "outgoing"
                    connected_entities.append(connected_entity)
                    self.logger.debug(f"Added connected entity: {connected_entity.name} (via outgoing relation)")
                else:
                    self.logger.warning(f"Target entity with ID {relation.target_id} not found")
            
            elif is_target:
                # The entity is the target, so we need to find the source entity
                self.logger.debug(f"'{entity.name}' is target, retrieving source entity with ID: {relation.source_id}")
                connected_entity = self._get_entity_by_id(relation.source_id)
                if connected_entity:
                    # Add relation metadata to the entity for later use
                    connected_entity.metadata["source_relation"] = relation.type
                    connected_entity.metadata["relation_direction"] = "incoming"
                    connected_entities.append(connected_entity)
                    self.logger.debug(f"Added connected entity: {connected_entity.name} (via incoming relation)")
                else:
                    self.logger.warning(f"Source entity with ID {relation.source_id} not found")
            
            else:
                self.logger.warning(f"Entity {entity.name} is neither source nor target in relation {relation.id}")
        
        self.logger.debug(f"Total connected entities found: {len(connected_entities)}")
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
            self.logger.info("No entities to prune")
            return []
        
        try:
            self.logger.debug(f"Starting entity pruning for {len(entities)} entities")
            
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
            
            self.logger.debug(f"Preparing pruning prompt for {n} entities out of {len(entities)}")
                
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
            
            self.logger.debug("Sending pruning prompt to LLM")
            response = self.llm.generate(messages, temperature=0.3)
            self.logger.debug("Received response from LLM")
            
            # Parse LLM response to extract JSON
            self.logger.debug("Parsing LLM response for entity scores")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    self.logger.debug("JSON string extracted successfully")
                    scores_dict = json.loads(json_str)
                    self.logger.debug(f"Scores parsed: {scores_dict}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    # Try a more aggressive approach to find valid JSON
                    try:
                        # Try to find anything that looks like a JSON object with entity names and scores
                        score_pairs = re.findall(r'"([^"]+)":\s*(0\.\d+|1\.0|1)', response)
                        if score_pairs:
                            scores_dict = {name: float(score) for name, score in score_pairs}
                            self.logger.debug(f"Extracted scores from regex: {scores_dict}")
                        else:
                            self.logger.warning("Couldn't extract scores with regex either")
                            return entities[:n]  # Return top n if extraction fails
                    except Exception as e2:
                        self.logger.warning(f"Alternative JSON extraction failed: {e2}")
                        return entities[:n]  # Return top n if parsing fails
            else:
                # Try to parse response as a numbered list with scores
                self.logger.debug("No JSON found, attempting to parse numbered list")
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
                        self.logger.warning(f"No JSON or numbered list found in LLM response")
                        return entities[:n]  # Return top n if no pattern found
                except Exception as e:
                    self.logger.warning(f"Failed to extract scores from response: {e}")
                    return entities[:n]  # Return top n if extraction fails
            
            # Assign scores to entities - use more flexible matching
            self.logger.debug("Assigning scores to entities")
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
                self.logger.debug(f"Assigned score {score} to entity '{entity.name}'")
            
            # Sort by score in descending order
            scored_entities = sorted(entities, key=lambda e: e.metadata.get("relevance_score", 0.0), reverse=True)
            
            # Return top n entities
            self.logger.debug(f"Returning top {n} entities by relevance score")
            return scored_entities[:n]
            
        except Exception as e:
            self.logger.error(f"Error pruning entities: {e}", exc_info=True)
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
            self.logger.debug(f"Retrieving entity with ID: {entity_id} from Neo4j")
            
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
            self.logger.debug(f"Executing Neo4j query for entity ID: {entity_id}")
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
            
            self.logger.debug(f"Successfully retrieved entity: {entity_name} ({entity_type})")
            return entity
            
        except Exception as e:
            self.logger.error(f"Error getting entity with ID {entity_id}: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    # Example usage
    import logging
    from tog.llms import AzureOpenAILLM
    from tog.kgs import Neo4jKnowledgeGraph
    from tog.models.entity import Entity
    from tog.models.relation import Relation
    
    # Configure improved logging
    configure_logging()
    
    # Create a main logger for this script
    main_logger = logging.getLogger("EntityExplorerDemo") 
    main_logger.setLevel(logging.INFO)
    
    main_logger.info("Starting Entity Explorer Demo")
    
    # Initialize components
    main_logger.info("Initializing LLM and Knowledge Graph")
    llm = AzureOpenAILLM(model_name="gpt-35-turbo")
    kg = Neo4jKnowledgeGraph()

    # Define query and prompts
    query = "What are the medical benefits of Medical Cannabis?"
    main_logger.info(f"Query: {query}")
    
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
    main_logger.info("Initializing Neo4jEntityExplorer")
    entity_explorer = Neo4jEntityExplorer(
        llm=llm,
        kg=kg,
        query=query,
        prune_prompt=entity_prune_prompt,
        prompt_params=prompt_params,
        max_entities_per_round=2
    )
    
    # Create example topic entity
    main_logger.info("Creating example topic entity")
    topic_entity = Entity(
        id="192db73673d90090cf1cb7d1be13aebc",
        name="Chronic Pain",
        type="Medical Condition", 
        metadata={"description": "A long-lasting pain that persists beyond the usual recovery period or accompanies a chronic health condition."}
    )
    
    # Create some example relations
    main_logger.info("Creating example relations")
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
    main_logger.info("Starting entity exploration...")
    connected_entities = entity_explorer.explore_entities(topic_entity, selected_relations)
    
    # Print results
    main_logger.info("\n=== RESULTS ===")
    main_logger.info(f"Connected entities for {topic_entity.name}:")
    print("8" * 20)
    print("Connected entities for {topic_entity.name}:")
    print("8" * 20)
    print("Connected entities:",connected_entities)
    print("8" * 20)
    for entity in connected_entities:
        main_logger.info(f"- {entity.name} ({entity.type}), Score: {entity.metadata.get('relevance_score', 0.0):.2f}")
        main_logger.info(f"  Relation: {entity.metadata.get('source_relation', 'Unknown')}")
        main_logger.info(f"  Direction: {entity.metadata.get('relation_direction', 'Unknown')}")
    
    main_logger.info("Entity Explorer Demo completed successfully")