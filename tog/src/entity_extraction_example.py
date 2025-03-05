from entity_extraction import EntityExtractor

def main():
    """Example demonstrating simplified entity extraction functionality"""
    # Initialize the entity extractor
    extractor = EntityExtractor()
    
    # Example queries
    queries = [
        "What are the health benefits of Mediterranean diet?",
        "Tell me about Tesla's latest electric vehicle technology and Elon Musk's vision for sustainable transportation",
        "How is artificial intelligence being used in healthcare and medical research?",
        "What were the main causes of World War II and its impact on European politics?",
    ]
    
    # Example custom prompts
    custom_prompt = """
    You are an expert entity extractor. From the given text, identify all key topics, concepts, and entities.
    Return ONLY a valid JSON object with a single key called "entities" containing a list of strings.
    Each string should be a distinct entity mentioned in the text.
    Be specific and precise with entity identification.
    Example format:
    {
      "entities": ["topic1", "topic2", "person1"]
    }
    """
    
    # Extract entities from each query
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        
        # Extract entities
        entities = extractor.extract_entities(query)
        print(f"Extracted entities: {entities}")
        
        # Extract with custom prompt
        custom_entities = extractor.extract_entities(query, prompt=custom_prompt)
        print(f"Custom prompt entities: {custom_entities}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
