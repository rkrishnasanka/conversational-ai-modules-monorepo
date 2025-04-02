from dotenv import load_dotenv
from typing import List
from tog.pipeline import MappingHandler
from tog.utils.logger import setup_logger
from tog.models.entity import Entity
from tog.models.kg import KnowledgeGraph

# Load environment variables
load_dotenv()

class EntityMapper:
    def __init__(self, kg: KnowledgeGraph, mapping_handler: MappingHandler):
        self.kg = kg
        self.mapping_handler = mapping_handler
        self.logger = setup_logger(__name__)

    def map_entities(self, extracted_entities: List[str]) -> List[Entity]:
        """
        Map extracted entities to knowledge graph entities using Neo4j.
        """
        # create an empty list to store the mapped entities
        mapped_entities = []
        
        try:
            with self.kg.driver.session() as session:
                for entity_name in extracted_entities:
                    # write a query to get the entities with the name of the extracted entities
                    query = """
                    MATCH (e)
                    WHERE toLower(e.name) = toLower($entity_name)
                    RETURN e.name as name, labels(e) as types, elementId(e) as id
                    LIMIT 1
                    """
                    
                    result = session.run(query, entity_name=entity_name)
                    record = result.single()
                    
                    if record:
                        # for each entity that is present in the knowledge graph, create an Entity Object with it and append it to the list
                        self.logger.debug(f"Entity found in KG: {record}")
                        entity = Entity(
                            id=str(record["id"]),
                            name=record["name"],
                            type=record["types"][0] if record["types"] else None  # Use first type if available
                        )
                        # Add mapping type to metadata
                        entity.metadata["mapping_type"] = "exact_match"
                        mapped_entities.append(entity)
                    else:
                        # for each entity that is not present in the knowledge graph, call self._use_mapping_handler() with the extracted_entity and KnowledgeGraph
                        self.logger.debug(f"Entity not found in KG, using mapping handler: {entity_name}")
                        mapped_entity = self._use_mapping_handler(entity_name)
                        if mapped_entity:
                            mapped_entities.append(mapped_entity)
                            
        except Exception as e:
            self.logger.error(f"Error mapping entities: {str(e)}")
            
        # return the list of mapped entities
        return mapped_entities

    def _use_mapping_handler(self, extracted_entity: str) -> Entity:
        """
        Use the mapping handler to map an entity to the knowledge graph.
        """
        try:
            # call the map method with the extracted entity
            mapped_entity = self.mapping_handler.fuzzy_match(extracted_entity)
            self.logger.debug(f"Mapped entity using mapping handler: {mapped_entity}")
            # # Add mapping type to metadata
            # if mapped_entity:
            #     mapped_entity.metadata["mapping_type"] = "fuzzy_match" # It is already added in the mapping handler.. thinking, which is better this one or the one in the mapping handler
            return mapped_entity
        except Exception as e:
            self.logger.error(f"Error using mapping handler for entity '{extracted_entity}': {str(e)}")
            return None
        