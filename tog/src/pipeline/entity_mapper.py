import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from tog.src.llms.base_llm import BaseLLM
from tog.src.llms.azure_openai_llm import AzureOpenAILLM
from tog.src.pipeline.mapping_handler import MappingHandler
from tog.src.utils.logger import setup_logger
from tog.src.models.entity import Entity
from tog.src.models.kg import KnowledgeGraph

# Load environment variables
load_dotenv()
# class EntityMapper:
#     def __init__(self, llm: Optional[BaseLLM] = None, prompts_file: str = "tog/prompts/entity_mapping_prompts.yaml"):
#         """
#         Initialize EntityMapper with Neo4j driver and LLM.
        
#         Args:
#             llm: Optional LLM instance to use for entity disambiguation and verification
#             prompts_file: Path to YAML file containing prompt templates
#         """
#         # Initialize LLM if not provided
#         if llm is None:
#             self.llm = AzureOpenAILLM(
#                 model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#                 api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#                 endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#                 api_version=os.getenv("AZURE_OPENAI_API_VERSION")
#             )
#         else:
#             self.llm = llm
        
#         # Initialize Neo4j driver
#         self.neo4j_driver = GraphDatabase.driver(
#             os.getenv("NEO4J_URI"),
#             auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
#         )
        
#         # Load prompts from YAML file
#         self.prompts = self._load_prompts(prompts_file)
    
#     def _load_prompts(self, prompts_file: str) -> Dict[str, Any]:
#         """Load prompts from YAML file"""
#         try:
#             with open(prompts_file, 'r') as file:
#                 return yaml.safe_load(file)
#         except FileNotFoundError:
#             # If file not found, use default prompts
#             print(f"Warning: Prompts file '{prompts_file}' not found. Using default prompts.")
#             return {
#                 "disambiguation": {
#                     "system": "You are an entity disambiguation assistant. Select the most appropriate entity based on context.",
#                     "user_template": """
#                     Question: {query}
#                     Entity mentioned in question: {entity}
                    
#                     Possible matches in knowledge graph:
#                     {context}
                    
#                     Based on the question context, which entity is most likely the one referred to in the question?
#                     Return only the exact name of the best matching entity.
#                     """
#                 },
#                 "verification": {
#                     "system": "You are an entity verification assistant for medical cannabis knowledge graphs.",
#                     "user_template": """
#                     Question: {query}
                    
#                     Extracted entities and their mappings to knowledge graph:
#                     {entity_text}
                    
#                     Do these entity mappings make sense in the context of this medical cannabis question?
#                     Answer only with "Yes" if they make sense, or explain briefly why they don't make sense.
#                     """
#                 }
#             }
    
#     def map_to_kg_entities(self, candidate_entities: List[str]) -> List[Dict[str, Any]]:
#         """
#         Map candidate entities to knowledge graph entities using Neo4j
#         """
#         mapped_entities = []
        
#         with self.neo4j_driver.session() as session:
#             for entity in candidate_entities:
#                 # Using fuzzy matching to find similar entities in the knowledge graph
#                 query = """
#                 MATCH (e)
#                 WHERE toLower(e.name) CONTAINS toLower($entity_name) OR
#                       apoc.text.levenshteinSimilarity(toLower(e.name), toLower($entity_name)) > 0.7 OR
#                       EXISTS {
#                           MATCH (e)-[:HAS_SYNONYM]->(s:Synonym)
#                           WHERE toLower(s.name) CONTAINS toLower($entity_name)
#                       }
#                 RETURN e.name as name, labels(e) as types, id(e) as id,
#                        apoc.text.levenshteinSimilarity(toLower(e.name), toLower($entity_name)) as score
#                 ORDER BY score DESC
#                 LIMIT 5
#                 """
                
#                 try:
#                     results = session.run(query, entity_name=entity)
#                     entity_matches = [record for record in results]
                    
#                     if entity_matches:
#                         mapped_entities.append({
#                             "query_entity": entity,
#                             "kg_entities": entity_matches
#                         })
#                 except Exception as e:
#                     print(f"Error querying Neo4j for entity {entity}: {e}")
        
#         return mapped_entities
    
#     def disambiguate_entities(self, query: str, candidate_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Disambiguate entities when multiple matches exist
#         """
#         disambiguated_entities = []     
        
#         for mapping in candidate_mappings:
#             if len(mapping["kg_entities"]) > 1:
#                 # Get context for disambiguation
#                 entities_with_context = []
                
#                 with self.neo4j_driver.session() as session:
#                     for entity in mapping["kg_entities"]:
#                         # Get relationships to provide context
#                         context_query = """
#                         MATCH (e)-[r]-(related)
#                         WHERE id(e) = $entity_id
#                         RETURN type(r) as relation_type, related.name as related_entity
#                         LIMIT 10
#                         """
                        
#                         try:
#                             context_results = session.run(context_query, entity_id=entity["id"])
#                             context = [f"{record['relation_type']}: {record['related_entity']}" 
#                                       for record in context_results]
                            
#                             entities_with_context.append({
#                                 "entity": entity,
#                                 "context": context
#                             })
#                         except Exception as e:
#                             print(f"Error getting context for entity {entity['name']}: {e}")
                
#                 # Use LLM to disambiguate based on query context
#                 context_text = "\n".join([
#                     f"Entity: {e['entity']['name']} (Types: {', '.join(e['entity']['types'])})\n" +
#                     f"Related to: {'; '.join(e['context']) if e['context'] else 'No related entities found'}"
#                     for e in entities_with_context
#                 ])
                
#                 # Use prompt from YAML
#                 messages = [
#                     {"role": "system", "content": self.prompts["disambiguation"]["system"]},
#                     {"role": "user", "content": self.prompts["disambiguation"]["user_template"].format(
#                         query=query,
#                         entity=mapping['query_entity'],
#                         context=context_text
#                     )}
#                 ]
                
#                 try:
#                     selected_entity_name = self.llm.generate(
#                         messages=messages,
#                         temperature=0.3,
#                         max_tokens=50
#                     )
                    
#                     # Find the selected entity in the original list
#                     selected = None
#                     for entity_context in entities_with_context:
#                         if entity_context["entity"]["name"].lower() in selected_entity_name.lower():
#                             selected = entity_context["entity"]
#                             break
                    
#                     if selected:
#                         disambiguated_entities.append({
#                             "query_entity": mapping["query_entity"],
#                             "kg_entity": selected
#                         })
#                     else:
#                         # If no clear match, take the highest scoring one
#                         disambiguated_entities.append({
#                             "query_entity": mapping["query_entity"],
#                             "kg_entity": mapping["kg_entities"][0]
#                         })
                
#                 except Exception as e:
#                     print(f"Error during disambiguation: {e}")
#                     # Fallback to highest scoring entity
#                     disambiguated_entities.append({
#                         "query_entity": mapping["query_entity"],
#                         "kg_entity": mapping["kg_entities"][0]
#                     })
            
#             else:
#                 # Only one match, no disambiguation needed
#                 disambiguated_entities.append({
#                     "query_entity": mapping["query_entity"],
#                     "kg_entity": mapping["kg_entities"][0] if mapping["kg_entities"] else None
#                 })
        
#         return disambiguated_entities
    
#     def verify_entity_mappings(self, query: str, mapped_entities: List[Dict[str, Any]]) -> tuple:
#         """
#         Verify if the entity mappings make sense in the context of the question
#         """
#         if not mapped_entities:
#             return False, "No entities were mapped"
        
#         # Format mapped entities for verification
#         entity_text = "\n".join([
#             f"Query: '{e['query_entity']}' → KG: '{e['kg_entity']['name']}' (Types: {', '.join(e['kg_entity']['types'])})"
#             for e in mapped_entities if e['kg_entity'] is not None
#         ])
        
#         # Use prompt from YAML
#         messages = [
#             {"role": "system", "content": self.prompts["verification"]["system"]},
#             {"role": "user", "content": self.prompts["verification"]["user_template"].format(
#                 query=query,
#                 entity_text=entity_text
#             )}
#         ]
        
#         try:
#             verification = self.llm.generate(
#                 messages=messages,
#                 temperature=0.3,
#                 max_tokens=100
#             )
            
#             is_valid = verification.lower().startswith("yes")
            
#             return is_valid, verification
        
#         except Exception as e:
#             print(f"Error during verification: {e}")
#             return False, f"Error during verification: {e}"
    
#     def map_entities(self, entities: List[str], user_query: str) -> Dict[str, Any]:
#         """
#         Complete entity mapping process with user-provided entities
#         """
#         try:
#             if not entities:
#                 return {"success": False, "message": "No entities provided", "data": None}
            
#             # Step 1: Map to knowledge graph entities
#             kg_mappings = self.map_to_kg_entities(entities)
#             if not kg_mappings:
#                 return {"success": False, "message": "No matching entities found in knowledge graph", "data": {"entities": entities}}
            
#             # Step 2: Disambiguate entities if needed
#             disambiguated = self.disambiguate_entities(user_query, kg_mappings)
            
#             # Step 3: Verify mappings
#             is_valid, verification_message = self.verify_entity_mappings(user_query, disambiguated)
            
#             # Return the results
#             return {
#                 "success": True,
#                 "message": verification_message,
#                 "data": {
#                     "original_query": user_query,
#                     "provided_entities": entities,
#                     "mapped_entities": disambiguated,
#                     "is_valid": is_valid
#                 }
#             }
        
#         except Exception as e:
#             return {"success": False, "message": f"Error in entity mapping process: {e}", "data": None}
    
#     def close(self):
#         """
#         Close the Neo4j driver connection
#         """
#         self.neo4j_driver.close()

# # Example usage
# if __name__ == "__main__":
#     # Example entities and query
#     entities = ["CBD", "anxiety"]
#     query = "What effects does CBD have on anxiety?"
    
#     mapper = EntityMapper()
#     result: Entity = mapper.map_entities(entities, query)
    
#     print("\n=== Entity Mapping Results ===")
#     print(f"Query: {result['data']['original_query']}")
#     print(f"Provided entities: {result['data']['provided_entities']}")
    
#     print("\nMapped entities:")
#     for entity in result['data']['mapped_entities']:
#         if entity['kg_entity']:
#             print(f"- {entity['query_entity']} → {entity['kg_entity']['name']} (Types: {', '.join(entity['kg_entity']['types'])})")
#         else:
#             print(f"- {entity['query_entity']} → No match found")
    
#     print(f"\nVerification: {result['message']}")
    
#     # Close connections
#     mapper.close()

class EntityMapper:
    def __init__(self, kg: KnowledgeGraph, mapping_handler: MappingHandler):
        self.kg = kg
        self.mapping_handler = mapping_handler

    def map_entities(self, extracted_entities: List[str]) -> List[Entity]:
        """
        Map extracted entities to knowledge graph entities using Neo4j.
        """
        # create an empty list to store the mapped entities
        # write a query to get the entities with the name of the extracted entities
        # for each entitiy that is present in the knowledge graph, create an Entity Object with it and append it to the list
        # for each entity that is not present in the knowledge graph, call self._use_mapping_handler() with the extracted_entity and KnowledgeGraph
        # append the result to the list of mapped entities
        # return the list of mapped entities
        ...

    def _use_mapping_handler(self, extracted_entity: str) -> Entity:
        """
        Use the mapping handler to map an entity to the knowledge graph.
        """
        # create a new instance of the mapping handler
        # call the map method with the extracted entity and the knowledge graph
        # return the mapped entity
        ...