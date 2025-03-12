import logging
import re
from typing import Dict, List, Tuple, Optional, Set, Any
import rdflib
from rdflib import Graph, URIRef, Literal
from fuzzywuzzy import fuzz
import pandas as pd
from tog.src.models.entity import Entity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityMapper:
    """
    Maps extracted entities to entities in an RDF knowledge graph using fuzzy matching
    and semantic similarity techniques. Uses SPARQL queries for efficient graph interaction.
    """
    
    def __init__(
        self,
        kg_path: str = None,
        kg_graph: Graph = None,
        match_threshold: int = 75,
        top_k: int = 5,
        type_match_bonus: int = 10,
        use_cache: bool = True,
    ):
        """
        Initialize the EntityMapper with a knowledge graph.
        
        Args:
            kg_path: Path to the RDF knowledge graph file
            kg_graph: Directly provide an RDFLib Graph object
            match_threshold: Minimum score (0-100) for a match to be considered valid
            top_k: Maximum number of mappings to return per entity
            type_match_bonus: Extra score points added when entity types match
            use_cache: Whether to cache entity mappings for faster repeated lookups
        """
        self.match_threshold = match_threshold
        self.top_k = top_k
        self.type_match_bonus = type_match_bonus
        self.use_cache = use_cache
        
        # Initialize the knowledge graph
        if kg_graph is not None:
            self.kg = kg_graph
        elif kg_path is not None:
            self.kg = self._load_kg(kg_path)
        else:
            raise ValueError("Either kg_path or kg_graph must be provided")
            
        # Cache of entity labels and their URIs
        self.entity_cache = {}
        self.entity_labels = []
        self.entity_types = {}
        self.uri_to_label = {}
        
        # Initialize the cache
        if self.use_cache:
            self._build_entity_cache()
    
    def _load_kg(self, kg_path: str) -> Graph:
        """
        Load the RDF knowledge graph from file.
        
        Args:
            kg_path: Path to the knowledge graph file
            
        Returns:
            An RDFLib Graph object
        """
        logger.info(f"Loading knowledge graph from {kg_path}")
        graph = Graph()
        try:
            # Try to determine the format based on file extension
            file_format = kg_path.split('.')[-1].lower()
            if file_format == 'ttl':
                format_name = 'turtle'
            elif file_format == 'rdf':
                format_name = 'xml'
            elif file_format == 'nt':
                format_name = 'nt'
            elif file_format == 'n3':
                format_name = 'n3'
            elif file_format == 'jsonld':
                format_name = 'json-ld'
            else:
                format_name = 'turtle'  # Default to Turtle format
                
            graph.parse(kg_path, format=format_name)
            logger.info(f"Loaded {len(graph)} triples from knowledge graph")
            return graph
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            raise
    
    def _build_entity_cache(self):
        """
        Build a cache of entity labels, types, and URIs from the knowledge graph
        using SPARQL queries for efficiency.
        """
        logger.info("Building entity cache from knowledge graph...")
        
        # SPARQL query to get entities with their labels
        # This queries multiple common label predicates
        label_query = """
        SELECT ?entity ?label WHERE {
          {
            ?entity rdfs:label ?label .
          } UNION {
            ?entity skos:prefLabel ?label .
          } UNION {
            ?entity schema:name ?label .
          } UNION {
            ?entity dc:title ?label .
          } UNION {
            ?entity foaf:name ?label .
          }
          FILTER(ISBLANK(?entity) = false)
          FILTER(ISLITERAL(?label))
        }
        """
        
        # Define common prefixes for SPARQL
        prefixes = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://schema.org/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        """
        
        full_label_query = prefixes + label_query
        
        try:
            # Execute query to get entity labels
            label_results = self.kg.query(full_label_query)
            
            # Process label results
            for row in label_results:
                entity_uri = row['entity']
                label = str(row['label'])
                
                if label:
                    self.entity_cache[entity_uri] = label
                    self.uri_to_label[entity_uri] = label
                    self.entity_labels.append(label)
            
            # SPARQL query to get entity types
            type_query = """
            SELECT ?entity ?type WHERE {
              ?entity rdf:type|rdfs:type ?type .
              FILTER(ISBLANK(?entity) = false)
              FILTER(ISBLANK(?type) = false)
            }
            """
            
            full_type_query = prefixes + type_query
            
            # Execute query to get entity types
            type_results = self.kg.query(full_type_query)
            
            # Process type results
            for row in type_results:
                entity_uri = row['entity']
                type_uri = row['type']
                
                if entity_uri in self.entity_cache:
                    type_name = self._extract_type_from_uri(type_uri)
                    if type_name:
                        if entity_uri in self.entity_types:
                            self.entity_types[entity_uri].add(type_name)
                        else:
                            self.entity_types[entity_uri] = {type_name}
            
            logger.info(f"Built cache with {len(self.entity_cache)} entities")
            
        except Exception as e:
            logger.error(f"Error building entity cache with SPARQL: {str(e)}")
            logger.info("Falling back to direct graph traversal for cache building")
            self._build_entity_cache_fallback()
    
    def _build_entity_cache_fallback(self):
        """
        Fallback method to build entity cache by direct graph traversal
        if SPARQL queries fail.
        """
        # Common label predicates in different ontologies
        label_predicates = [
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
            URIRef("http://schema.org/name"),
            URIRef("http://purl.org/dc/elements/1.1/title"),
            URIRef("http://xmlns.com/foaf/0.1/name"),
        ]
        
        # Common type predicates
        type_predicates = [
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#type"),
        ]
        
        # Collect all entities with labels
        entities = {}
        for predicate in label_predicates:
            for s, p, o in self.kg.triples((None, predicate, None)):
                if isinstance(o, Literal):
                    label = str(o)
                    if label:
                        entities[s] = label
                        self.uri_to_label[s] = label
                        self.entity_labels.append(label)
        
        # Get entity types
        for entity_uri in entities:
            for type_pred in type_predicates:
                for _, _, type_uri in self.kg.triples((entity_uri, type_pred, None)):
                    # Extract the type name from the URI
                    type_name = self._extract_type_from_uri(type_uri)
                    if type_name:
                        if entity_uri in self.entity_types:
                            self.entity_types[entity_uri].add(type_name)
                        else:
                            self.entity_types[entity_uri] = {type_name}
        
        logger.info(f"Built cache with {len(entities)} entities (fallback method)")
        self.entity_cache = entities
    
    def _extract_type_from_uri(self, uri: URIRef) -> str:
        """Extract a human-readable type name from a type URI."""
        # Try to get the fragment after # or the last part after /
        uri_str = str(uri)
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        else:
            return uri_str.split('/')[-1]
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        Normalize entity type for better comparison.
        
        Args:
            entity_type: The entity type to normalize
            
        Returns:
            Normalized entity type
        """
        if not entity_type:
            return ""
            
        # Convert to lowercase
        normalized = entity_type.lower()
        
        # Remove common suffixes
        for suffix in ["class", "type", "category", "entity"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        return normalized
        
    def _calculate_match_score(
        self, extracted_entity: Entity, kg_uri: URIRef, kg_label: str, kg_types: Set[str]
    ) -> float:
        """
        Calculate a match score between an extracted entity and a KG entity.
        
        Args:
            extracted_entity: The extracted entity
            kg_uri: The URI of the KG entity
            kg_label: The label of the KG entity
            kg_types: The types of the KG entity
            
        Returns:
            A match score between 0 and 100+type_match_bonus
        """
        # Calculate string similarity
        similarity_score = fuzz.token_sort_ratio(extracted_entity.name.lower(), kg_label.lower())
        
        # Add bonus for type match if we have type information
        if extracted_entity.type and kg_types:
            normalized_extracted_type = self._normalize_entity_type(extracted_entity.type)
            
            for kg_type in kg_types:
                normalized_kg_type = self._normalize_entity_type(kg_type)
                if normalized_extracted_type in normalized_kg_type or normalized_kg_type in normalized_extracted_type:
                    similarity_score += self.type_match_bonus
                    break
        
        return similarity_score
    
    def map_entity(self, entity: Entity) -> List[Dict[str, Any]]:
        """
        Map a single extracted entity to entities in the knowledge graph using SPARQL for
        direct entity lookup if cache is not being used.
        
        Args:
            entity: The extracted entity to map
            
        Returns:
            A list of dictionaries containing mapping information, 
            sorted by match score (highest first)
        """
        if not entity.name:
            return []
            
        matches = []
        
        # Use cached entities if available
        if self.use_cache and self.entity_cache:
            for uri, label in self.entity_cache.items():
                kg_types = self.entity_types.get(uri, set())
                score = self._calculate_match_score(entity, uri, label, kg_types)
                
                if score >= self.match_threshold:
                    matches.append({
                        "uri": str(uri),
                        "label": label,
                        "types": list(kg_types),
                        "score": score,
                        "extracted_entity": entity.__dict__
                    })
        else:
            # If cache is not used, use SPARQL to query for entities
            try:
                # Query for labels
                query = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX schema: <http://schema.org/>
                
                SELECT ?entity ?label WHERE {
                  {
                    ?entity rdfs:label ?label .
                  } UNION {
                    ?entity schema:name ?label .
                  }
                  FILTER(ISLITERAL(?label))
                }
                """
                
                results = self.kg.query(query)
                
                for row in results:
                    entity_uri = row['entity']
                    label = str(row['label'])
                    
                    # Get types for this entity
                    type_query = """
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    
                    SELECT ?type WHERE {
                      <%s> rdf:type ?type .
                    }
                    """ % str(entity_uri)
                    
                    type_results = self.kg.query(type_query)
                    kg_types = set()
                    
                    for type_row in type_results:
                        type_uri = type_row['type']
                        type_name = self._extract_type_from_uri(type_uri)
                        if type_name:
                            kg_types.add(type_name)
                    
                    score = self._calculate_match_score(entity, entity_uri, label, kg_types)
                    
                    if score >= self.match_threshold:
                        matches.append({
                            "uri": str(entity_uri),
                            "label": label,
                            "types": list(kg_types),
                            "score": score,
                            "extracted_entity": entity.__dict__
                        })
                        
            except Exception as e:
                logger.error(f"Error in SPARQL query for entity mapping: {str(e)}")
                # Fall back to the simpler method if SPARQL query fails
                return self._map_entity_fallback(entity)
        
        # Sort matches by score (descending) and limit to top_k
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:self.top_k]
        return matches
    
    def _map_entity_fallback(self, entity: Entity) -> List[Dict[str, Any]]:
        """Fallback method for entity mapping if SPARQL queries fail."""
        matches = []
        
        # Directly query the graph for labels
        for predicate in ["http://www.w3.org/2000/01/rdf-schema#label", "http://schema.org/name"]:
            for s, p, o in self.kg.triples((None, URIRef(predicate), None)):
                if isinstance(o, Literal):
                    label = str(o)
                    kg_types = set()
                    
                    # Get types for this entity
                    for _, _, t in self.kg.triples((s, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None)):
                        kg_types.add(self._extract_type_from_uri(t))
                    
                    score = self._calculate_match_score(entity, s, label, kg_types)
                    
                    if score >= self.match_threshold:
                        matches.append({
                            "uri": str(s),
                            "label": label,
                            "types": list(kg_types),
                            "score": score,
                            "extracted_entity": entity.__dict__
                        })
        
        # Sort matches by score (descending) and limit to top_k
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:self.top_k]
        return matches
    
    def map_entities(self, entities: List[Entity]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map a list of extracted entities to entities in the knowledge graph.
        
        Args:
            entities: List of extracted entities to map
            
        Returns:
            Dictionary mapping entity names to lists of matched kg entities
        """
        entity_mappings = {}
        for entity in entities:
            mapped = self.map_entity(entity)
            if mapped:
                entity_mappings[entity.name] = mapped
        
        return entity_mappings
    
    def get_entity_context(self, uri: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get context information for an entity from the knowledge graph using SPARQL.
        
        Args:
            uri: The URI of the entity
            depth: How many hops to traverse for context
            
        Returns:
            Dictionary with entity context information
        """
        uri_ref = URIRef(uri)
        context = {
            "uri": uri,
            "label": self.uri_to_label.get(uri_ref, str(uri_ref).split('/')[-1]),
            "properties": [],
            "related_entities": []
        }
        
        try:
            # SPARQL query for outgoing properties
            outgoing_query = """
            SELECT ?pred ?obj ?objLabel WHERE {
              <%s> ?pred ?obj .
              OPTIONAL {
                ?obj rdfs:label|schema:name ?objLabel .
                FILTER(ISLITERAL(?objLabel))
              }
            }
            """ % uri
            
            # Add prefixes
            prefixed_outgoing_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>
            """ + outgoing_query
            
            # Execute query for outgoing properties
            outgoing_results = self.kg.query(prefixed_outgoing_query)
            
            # Process outgoing properties
            for row in outgoing_results:
                pred = row['pred']
                obj = row['obj']
                
                # Extract predicate name from URI
                pred_name = str(pred).split('/')[-1].split('#')[-1]
                
                if isinstance(obj, URIRef):
                    # Use the provided label if available, otherwise extract from URI
                    obj_label = str(row['objLabel']) if row['objLabel'] else str(obj).split('/')[-1]
                    
                    context["properties"].append({
                        "predicate": pred_name,
                        "predicate_uri": str(pred),
                        "object": obj_label,
                        "object_uri": str(obj),
                        "is_uri": True
                    })
                    
                    # Add to related entities if within depth
                    if depth > 0:
                        context["related_entities"].append({
                            "uri": str(obj),
                            "label": obj_label,
                            "relation": pred_name
                        })
                else:
                    # Literal object
                    context["properties"].append({
                        "predicate": pred_name,
                        "predicate_uri": str(pred),
                        "object": str(obj),
                        "is_uri": False
                    })
            
            # SPARQL query for incoming relations
            incoming_query = """
            SELECT ?subj ?pred ?subjLabel WHERE {
              ?subj ?pred <%s> .
              OPTIONAL {
                ?subj rdfs:label|schema:name ?subjLabel .
                FILTER(ISLITERAL(?subjLabel))
              }
            }
            """ % uri
            
            # Add prefixes
            prefixed_incoming_query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>
            """ + incoming_query
            
            # Execute query for incoming relations
            incoming_results = self.kg.query(prefixed_incoming_query)
            
            # Process incoming relations
            for row in incoming_results:
                subj = row['subj']
                pred = row['pred']
                
                # Extract predicate name from URI
                pred_name = str(pred).split('/')[-1].split('#')[-1]
                
                # Use the provided label if available, otherwise extract from URI
                subj_label = str(row['subjLabel']) if row['subjLabel'] else str(subj).split('/')[-1]
                
                context["properties"].append({
                    "predicate": f"reverse_{pred_name}",
                    "predicate_uri": str(pred),
                    "object": subj_label,
                    "object_uri": str(subj),
                    "is_uri": True,
                    "is_reverse": True
                })
                
                # Add to related entities if within depth
                if depth > 0:
                    context["related_entities"].append({
                        "uri": str(subj),
                        "label": subj_label,
                        "relation": f"reverse_{pred_name}"
                    })
                    
        except Exception as e:
            logger.error(f"Error executing SPARQL for entity context: {str(e)}")
            logger.info("Falling back to direct graph traversal for context")
            return self._get_entity_context_fallback(uri_ref, depth)
            
        return context
    
    def _get_entity_context_fallback(self, uri_ref: URIRef, depth: int = 1) -> Dict[str, Any]:
        """Fallback method to get entity context by direct traversal if SPARQL fails."""
        context = {
            "uri": str(uri_ref),
            "label": self.uri_to_label.get(uri_ref, str(uri_ref).split('/')[-1]),
            "properties": [],
            "related_entities": []
        }
        
        # Get direct properties
        for p, o in self.kg.predicate_objects(uri_ref):
            pred_name = str(p).split('/')[-1].split('#')[-1]
            
            if isinstance(o, URIRef):
                obj_label = self.uri_to_label.get(o, str(o).split('/')[-1])
                context["properties"].append({
                    "predicate": pred_name,
                    "predicate_uri": str(p),
                    "object": obj_label,
                    "object_uri": str(o),
                    "is_uri": True
                })
                
                # Add to related entities if within depth
                if depth > 0:
                    context["related_entities"].append({
                        "uri": str(o),
                        "label": obj_label,
                        "relation": pred_name
                    })
            else:
                context["properties"].append({
                    "predicate": pred_name,
                    "predicate_uri": str(p),
                    "object": str(o),
                    "is_uri": False
                })
        
        # Get incoming relations
        for s, p in self.kg.subject_predicates(uri_ref):
            pred_name = str(p).split('/')[-1].split('#')[-1]
            subj_label = self.uri_to_label.get(s, str(s).split('/')[-1])
            
            context["properties"].append({
                "predicate": f"reverse_{pred_name}",
                "predicate_uri": str(p),
                "object": subj_label,
                "object_uri": str(s),
                "is_uri": True,
                "is_reverse": True
            })
            
            # Add to related entities if within depth
            if depth > 0:
                context["related_entities"].append({
                    "uri": str(s),
                    "label": subj_label,
                    "relation": f"reverse_{pred_name}"
                })
        
        return context
    
    def to_dataframe(self, entity_mappings: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Convert entity mappings to a pandas DataFrame for easier analysis.
        
        Args:
            entity_mappings: Result from map_entities method
            
        Returns:
            DataFrame with entity mapping information
        """
        rows = []
        for entity_name, mappings in entity_mappings.items():
            for mapping in mappings:
                row = {
                    "extracted_entity": entity_name,
                    "extracted_type": mapping["extracted_entity"]["type"],
                    "kg_uri": mapping["uri"],
                    "kg_label": mapping["label"],
                    "kg_types": ", ".join(mapping["types"]),
                    "score": mapping["score"]
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df if not df.empty else None

    def run_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a custom SPARQL query against the knowledge graph.
        
        Args:
            query: A SPARQL query string
            
        Returns:
            List of dictionaries containing the query results
        """
        try:
            results = self.kg.query(query)
            result_list = []
            
            # Convert results to a list of dictionaries
            for row in results:
                row_dict = {}
                for var in results.vars:
                    value = row[var]
                    if value is not None:
                        row_dict[var] = str(value)
                    else:
                        row_dict[var] = None
                result_list.append(row_dict)
                
            return result_list
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    from tog.src.pipeline.entity_extractor import GroqEnityExtractor
    
    # Initialize the entity mapper with a sample KG
    kg_path = "path/to/your/knowledge_graph.ttl"
    try:
        mapper = EntityMapper(kg_path=kg_path, match_threshold=70)
        
        # Extract entities from text
        extractor = GroqEnityExtractor("mixtral-8x7b-32768")
        text = "Geoffrey Hinton is a computer scientist known for his work on neural networks."
        extracted_entities = extractor.extract_entities(text)
        
        # Map extracted entities to KG entities
        mapped_entities = mapper.map_entities(extracted_entities)
        
        # Print the results
        print(f"Found {len(extracted_entities)} entities in text:")
        for entity in extracted_entities:
            print(f"- {entity.name} ({entity.type})")
        
        print("\nMapped entities:")
        for entity_name, mappings in mapped_entities.items():
            print(f"\nEntity: {entity_name}")
            for i, mapping in enumerate(mappings, 1):
                print(f"  Match {i}: {mapping['label']} (Score: {mapping['score']})")
                print(f"  URI: {mapping['uri']}")
                print(f"  Types: {', '.join(mapping['types'])}")
                
                # Get additional context for the top match
                if i == 1:
                    context = mapper.get_entity_context(mapping['uri'])
                    print(f"  Properties: {len(context['properties'])}")
                    for prop in context['properties'][:3]:  # Show first 3 properties
                        print(f"    {prop['predicate']}: {prop['object']}")
                    if len(context['properties']) > 3:
                        print(f"    ... and {len(context['properties'])-3} more properties")
        
        # Execute a custom SPARQL query
        print("\nRunning a custom SPARQL query:")
        sample_query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?label WHERE {
          ?entity rdfs:label ?label .
          FILTER(CONTAINS(LCASE(STR(?label)), "hinton"))
        }
        LIMIT 5
        """
        
        query_results = mapper.run_sparql_query(sample_query)
        for result in query_results:
            print(f"  {result.get('entity')}: {result.get('label')}")
            
    except Exception as e:
        print(f"Error in example: {str(e)}")
