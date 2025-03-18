import os
import logging
from typing import List, Dict, Any, Optional, Union
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from tog.src.models.entity import Entity

logger = logging.getLogger(__name__)

class EntityMapper:
    """
    Maps extracted entities to entities in a knowledge graph (.nt file)
    using SPARQL queries as described in the Think-On-Graph paper.
    """
    
    def __init__(self, kg_source: Union[str, Graph]):
        """
        Initialize the EntityMapper with either a path to a knowledge graph or a Graph object.
        
        Args:
            kg_source: Path to the knowledge graph file (.nt format) or a Graph object
        """
        self.kg_graph = Graph()
        
        if isinstance(kg_source, Graph):
            self.kg_graph = kg_source
            logger.info(f"Using provided Graph with {len(self.kg_graph)} triples")
        else:
            self._load_knowledge_graph(kg_source)
        
    def _load_knowledge_graph(self, kg_path: str) -> None:
        """
        Load the knowledge graph from a file.
        
        Args:
            kg_path: Path to the knowledge graph file
        """
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
        
        self.kg_graph.parse(kg_path, format="nt")
        logger.info(f"Loaded knowledge graph with {len(self.kg_graph)} triples")
    
    def map_entities(self, extracted_entities: List[Entity]) -> List[Dict[str, Any]]:
        """
        Map extracted entities to entities in the knowledge graph.
        
        Args:
            extracted_entities: List of entities extracted by EntityExtractor
            
        Returns:
            List of mapped entities with KG URIs and additional information
        """
        mapped_entities = []
        
        for entity in extracted_entities:
            kg_entity = self._find_matching_entity(entity)
            if kg_entity:
                mapped_entities.append(kg_entity)
        
        return mapped_entities
    
    def _find_matching_entity(self, entity: Entity) -> Optional[Dict[str, Any]]:
        """
        Find a matching entity in the knowledge graph for an extracted entity.
        
        Args:
            entity: Extracted entity
            
        Returns:
            Dictionary containing the mapped entity information or None if no match
        """
        # Basic exact name matching query
        query_str = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?entity ?label ?type
        WHERE {
            ?entity rdfs:label ?label .
            ?entity rdf:type ?type .
            FILTER(lcase(str(?label)) = lcase(%s))
        }
        LIMIT 5
        """
        
        # Prepare the query with the entity name
        query = prepareQuery(query_str % f'"{entity.name}"')
        
        # Execute the query
        results = self.kg_graph.query(query)
        
        matches = []
        for row in results:
            matches.append({
                'uri': str(row.entity),
                'label': str(row.label),
                'type': str(row.type),
                'confidence': 1.0,  # Exact match confidence
                'original_entity': entity.to_dict()
            })
        
        if matches:
            # Return the best match (in this simple version, we take the first match)
            return matches[0]
        
        # If no exact match is found, try fuzzy matching (placeholder for future implementation)
        return None
    
    def get_entity_relations(self, entity_uri: str) -> List[Dict[str, Any]]:
        """
        Get all relations for a specific entity in the knowledge graph.
        
        Args:
            entity_uri: URI of the entity
            
        Returns:
            List of relations (subject, predicate, object)
        """
        query_str = """
        SELECT ?predicate ?object ?objectLabel
        WHERE {
            <%s> ?predicate ?object .
            OPTIONAL { ?object rdfs:label ?objectLabel }
        }
        """
        
        query = prepareQuery(query_str % entity_uri)
        results = self.kg_graph.query(query)
        
        relations = []
        for row in results:
            relation = {
                'subject': entity_uri,
                'predicate': str(row.predicate),
                'object': str(row.object)
            }
            
            if hasattr(row, 'objectLabel') and row.objectLabel:
                relation['object_label'] = str(row.objectLabel)
                
            relations.append(relation)
            
        return relations

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    kg_path = "path/to/your/knowledge_graph.nt"
    entity_mapper = EntityMapper(kg_path)
    
    # Example extracted entities (replace with actual extraction)
    extracted_entities = [
        Entity(name="ExampleEntity", type="ExampleType", metadata={"info": "example"})
    ]
    
    mapped_entities = entity_mapper.map_entities(extracted_entities)
    for entity in mapped_entities:
        logger.info(entity)