import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

logger = logging.getLogger(__name__)

class KnowledgeGraph(ABC):
    """
    Abstract base class for knowledge graph implementations.
    Provides a common interface for different knowledge graph backends.
    """
    
    @abstractmethod
    def __init__(self, source: Any, **kwargs):
        """Initialize a knowledge graph from a source"""
        pass
    
    @abstractmethod
    def query(self, query_str: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a query against the knowledge graph"""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity details by ID"""
        pass
    
    @abstractmethod
    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find an entity by name"""
        pass
    
    @abstractmethod
    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relations for a specific entity"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the size of the knowledge graph"""
        pass
    
    @abstractmethod
    def add_triple(self, subject: str, predicate: str, object_: str) -> None:
        """Add a triple to the knowledge graph"""
        pass


class RDFKnowledgeGraph(KnowledgeGraph):
    """
    RDF-based implementation of a knowledge graph using rdflib.
    """
    
    def __init__(self, source: Union[str, Graph], **kwargs):
        """
        Initialize the RDF knowledge graph with either a path to a knowledge graph file or a Graph object.
        
        Args:
            source: Path to the knowledge graph file (.nt format) or a Graph object
            **kwargs: Additional arguments for graph initialization
        """
        self.graph = Graph()
        
        if isinstance(source, Graph):
            self.graph = source
            logger.info(f"Using provided Graph with {len(self.graph)} triples")
        elif isinstance(source, str):
            self._load_from_file(source, **kwargs)
        else:
            raise ValueError("Source must be either a file path (str) or a rdflib.Graph object")
    
    def _load_from_file(self, kg_path: str, format: str = "nt") -> None:
        """
        Load the knowledge graph from a file.
        
        Args:
            kg_path: Path to the knowledge graph file
            format: Format of the knowledge graph file (default: nt)
        """
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
        
        self.graph.parse(kg_path, format=format)
        logger.info(f"Loaded knowledge graph with {len(self.graph)} triples")
    
    def query(self, query_str: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query against the knowledge graph.
        
        Args:
            query_str: SPARQL query string
            **kwargs: Additional arguments for query execution
            
        Returns:
            List of result dictionaries
        """
        prepared_query = prepareQuery(query_str)
        results = self.graph.query(prepared_query, **kwargs)
        
        # Convert query results to dictionaries
        result_dicts = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                if hasattr(row, var) and getattr(row, var) is not None:
                    row_dict[var] = str(getattr(row, var))
            result_dicts.append(row_dict)
            
        return result_dicts
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity details by ID (URI).
        
        Args:
            entity_id: URI of the entity
            
        Returns:
            Dictionary containing entity information or None if not found
        """
        query_str = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?predicate ?object ?label ?type
        WHERE {
            <%s> ?predicate ?object .
            OPTIONAL { <%s> rdfs:label ?label } .
            OPTIONAL { <%s> rdf:type ?type }
        }
        """ % (entity_id, entity_id, entity_id)
        
        results = self.query(query_str)
        
        if not results:
            return None
            
        entity = {
            'uri': entity_id,
            'properties': {}
        }
        
        for result in results:
            if 'label' in result and 'label' not in entity:
                entity['label'] = result['label']
            if 'type' in result and 'type' not in entity:
                entity['type'] = result['type']
            if 'predicate' in result and 'object' in result:
                entity['properties'][result['predicate']] = result['object']
                
        return entity
    
    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find an entity by name (label).
        
        Args:
            name: Name/label of the entity
            
        Returns:
            Dictionary containing entity information or None if not found
        """
        query_str = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?entity ?label ?type
        WHERE {
            ?entity rdfs:label ?label .
            ?entity rdf:type ?type .
            FILTER(lcase(str(?label)) = lcase("%s"))
        }
        LIMIT 1
        """ % name
        
        results = self.query(query_str)
        
        if not results:
            return None
            
        entity_uri = results[0].get('entity')
        if not entity_uri:
            return None
            
        return self.get_entity(entity_uri)
    
    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get all relations for a specific entity in the knowledge graph.
        
        Args:
            entity_id: URI of the entity
            
        Returns:
            List of relations (subject, predicate, object)
        """
        query_str = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?predicate ?object ?objectLabel
        WHERE {
            <%s> ?predicate ?object .
            OPTIONAL { ?object rdfs:label ?objectLabel }
        }
        """ % entity_id
        
        results = self.query(query_str)
        
        relations = []
        for row in results:
            relation = {
                'subject': entity_id,
                'predicate': row.get('predicate'),
                'object': row.get('object')
            }
            
            if 'objectLabel' in row:
                relation['object_label'] = row['objectLabel']
                
            relations.append(relation)
            
        return relations
    
    def size(self) -> int:
        """
        Get the number of triples in the knowledge graph.
        
        Returns:
            Number of triples
        """
        return len(self.graph)
    
    def add_triple(self, subject: str, predicate: str, object_: str) -> None:
        """
        Add a triple to the knowledge graph.
        
        Args:
            subject: Subject of the triple
            predicate: Predicate of the triple
            object_: Object of the triple
        """
        subject_node = rdflib.URIRef(subject)
        predicate_node = rdflib.URIRef(predicate)
        
        # Determine if object is a URI or literal
        if object_.startswith('http://') or object_.startswith('https://'):
            object_node = rdflib.URIRef(object_)
        else:
            object_node = rdflib.Literal(object_)
            
        self.graph.add((subject_node, predicate_node, object_node))