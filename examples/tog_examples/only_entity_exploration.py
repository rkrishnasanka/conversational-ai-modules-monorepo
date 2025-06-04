import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from tog.llms import BaseLLM
from tog.kgs import KnowledgeGraph
from tog.models.entity import Entity
from tog.models.relation import Relation


class EntityTriple:
    """
    A class representing a triple of (source_entity, relation, target_entity).
    """
    def __init__(self, source_entity: Entity, relation: Relation, target_entity: Entity):
        self.source_entity = source_entity
        self.relation = relation
        self.target_entity = target_entity
    
    def __repr__(self):
        return f"(Entity(id='{self.source_entity.id}', name='{self.source_entity.name}', type='{self.source_entity.type}', metadata={self.source_entity.metadata}), \n Relation(id='{self.relation.id}', source_id='{self.relation.source_id}', target_id='{self.relation.target_id}', type='{self.relation.type}', metadata={self.relation.metadata}), \n Entity(id='{self.target_entity.id}', name='{self.target_entity.name}', type='{self.target_entity.type}', metadata={self.target_entity.metadata}))"


class Explorer:
    """
    Abstract base class for exploring entities and relations in a knowledge graph.
    """
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, prompt_params: Dict[str, str] = None, system_prompt: str = None):
        self.llm = llm
        self.kg = kg
        self.query = query
        self.prompt_params = prompt_params or {}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.logger = logging.getLogger(self.__class__.__name__)


class EntityExplorer(Explorer):
    """
    Class for entity exploration in a knowledge graph following the ToG approach.
    """
    
    def __init__(self, llm: BaseLLM, kg: KnowledgeGraph, query: str, 
                 prompt_params: Dict[str, str] = None):
        """
        Initialize the EntityExplorer with a language model, knowledge graph, and query.
        
        Args:
            llm (BaseLLM): The language model to use for exploration.
            kg (KnowledgeGraph): The knowledge graph to explore.
            query (str): The query string for exploration.
            prompt_params (Dict[str, str], optional): Parameters for the prompt.
        """
        super().__init__(llm, kg, query, prompt_params)
        self.logger.info(f"Initialized EntityExplorer with query: '{query}'")
    
    def explore_entities(self, topic_entity: Entity, selected_relations: List[Relation]) -> List[EntityTriple]:
        """
        Explore entities connected to the topic entity through the selected relations.
        Based on the Tail() function described in the paper.
        
        Args:
            topic_entity: The entity to explore from.
            selected_relations: The relations to follow for exploration.
            
        Returns:
            A list of entity triples (source_entity, relation, target_entity).
        """
        self.logger.info(f"===== STEP 1: Starting entity exploration for '{topic_entity.name}' =====")
        self.logger.info(f"Using {len(selected_relations)} relations for exploration")
        
        # Entity Discovery - Find all connected entities using the Tail() function
        self.logger.info("===== STEP 2: Entity Discovery - Finding connected entities using Tail() =====")
        entity_triples = self._get_connected_entities(topic_entity, selected_relations)
        self.logger.info(f"Found {len(entity_triples)} connected entity triples for '{topic_entity.name}'")
        
        # Log found entity triples
        for i, triple in enumerate(entity_triples, 1):
            self.logger.info(f"  Triple {i}: {triple.source_entity.name} -{triple.relation.type}-> {triple.target_entity.name}")
        
        self.logger.info("===== STEP 3: Entity Exploration Complete =====")
        return entity_triples
    
    def _get_connected_entities(self, entity: Entity, relations: List[Relation]) -> List[EntityTriple]:
        """
        Get entity triples connected to the given entity through the specified relations.
        This implements the Tail() function described in the paper.
        
        Args:
            entity: The source entity.
            relations: The relations to follow.
            
        Returns:
            A list of entity triples (source_entity, relation, target_entity).
        """
        self.logger.debug(f"Finding entity triples connected to '{entity.name}' through {len(relations)} relations")
        entity_triples = []
        
        for i, relation in enumerate(relations, 1):
            self.logger.debug(f"Processing relation {i}: {relation.type} (ID: {relation.id})")
            
            # Determine if the given entity is the source or target in this relation
            is_source = relation.source_id == entity.id
            is_target = relation.target_id == entity.id
            
            if is_source:
                # The entity is the source, so we need to find the target entity
                self.logger.debug(f"'{entity.name}' is source, retrieving target entity with ID: {relation.target_id}")
                target_entity = self._get_entity_by_id(relation.target_id)
                if target_entity:
                    # Create a triple with the source entity, relation, and target entity
                    triple = EntityTriple(entity, relation, target_entity)
                    entity_triples.append(triple)
                    self.logger.debug(f"Added entity triple: {entity.name} -{relation.type}-> {target_entity.name}")
                else:
                    self.logger.warning(f"Target entity with ID {relation.target_id} not found")
            
            elif is_target:
                # The entity is the target, so we need to find the source entity
                self.logger.debug(f"'{entity.name}' is target, retrieving source entity with ID: {relation.source_id}")
                source_entity = self._get_entity_by_id(relation.source_id)
                if source_entity:
                    # Create a triple with the source entity, relation, and target entity (current entity)
                    triple = EntityTriple(source_entity, relation, entity)
                    entity_triples.append(triple)
                    self.logger.debug(f"Added entity triple: {source_entity.name} -{relation.type}-> {entity.name}")
                else:
                    self.logger.warning(f"Source entity with ID {relation.source_id} not found")
            
            else:
                self.logger.warning(f"Entity {entity.name} is neither source nor target in relation {relation.id}")
        
        self.logger.debug(f"Total entity triples found: {len(entity_triples)}")
        return entity_triples
    
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
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
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
    
    # Initialize explorer
    main_logger.info("Initializing Neo4jEntityExplorer")
    entity_explorer = Neo4jEntityExplorer(
        llm=llm,
        kg=kg,
        query=query,
        prompt_params={"entity_name": "Medical Cannabis"}
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
    entity_triples = entity_explorer.explore_entities(topic_entity, selected_relations)
    
    print("*"*10)
    print(entity_triples)
    print("*"*10)
    print("Entity triples:")
    for triple in entity_triples:
        print(f"Source: {triple.source_entity.name} ({triple.source_entity.type})")
        print(f"Relation: {triple.relation.type}")
        print(f"Target: {triple.target_entity.name} ({triple.target_entity.type})")
    # Print results
    main_logger.info("\n=== RESULTS ===")
    main_logger.info(f"Entity triples for {topic_entity.name}:")
    for i, triple in enumerate(entity_triples, 1):
        main_logger.info(f"\nTriple {i}:")
        main_logger.info(f"Source: {triple.source_entity.name} ({triple.source_entity.type})")
        main_logger.info(f"Relation: {triple.relation.type}")
        main_logger.info(f"Target: {triple.target_entity.name} ({triple.target_entity.type})")
    
    main_logger.info("Entity Explorer Demo completed successfully")