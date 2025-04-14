import json
from pprint import pprint
from dotenv import load_dotenv
from tog.pipeline.entity_mapper import EntityMapper
from tog.kgs import Neo4jKnowledgeGraph
from tog.pipeline.mapping_handler import Neo4jMappingHandler

#!/usr/bin/env python
"""
Example script demonstrating how to use the EntityMapper class
from the tog.pipeline.entity_mapper module.
"""

# Load environment variables
load_dotenv()

def main():
    # Initialize the Knowledge Graph connection
    kg = Neo4jKnowledgeGraph()
    
    # Initialize the mapping handler
    mapping_handler = Neo4jMappingHandler(kg=kg)
    
    # Create the EntityMapper instance
    entity_mapper = EntityMapper(kg, mapping_handler)
    
    # Example entities to map
    extracted_entities = ["CBD", "anxiety", "insomnia", "depression"]
    
    print(f"Mapping entities: {extracted_entities}")
    print("-" * 50)
    
    # Map the extracted entities to knowledge graph entities
    mapped_entities = entity_mapper.map_entities(extracted_entities)
    
    # Display the results
    print(f"Found {len(mapped_entities)} mapped entities:")
    for entity in mapped_entities:
        print(f"ID: {entity.id}, Name: {entity.name}, Type: {entity.type}")
        print(f"Metadata: {entity.metadata}")
        print("-" * 30)
    
    # Convert to dictionary for JSON serialization
    entities_dict = [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "metadata": entity.metadata
        }
        for entity in mapped_entities
    ]
    
    # Print as formatted JSON
    print("\nJSON output:")
    print(json.dumps(entities_dict, indent=4))

if __name__ == "__main__":
    main()