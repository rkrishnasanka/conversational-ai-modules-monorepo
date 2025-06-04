#!/usr/bin/env python
# Example usage of the entity extractor module

import os
from pprint import pprint
from dotenv import load_dotenv
import logging



# Import the entity extractors
from tog.pipeline.entity_extractor import (
    GroqEntityExtractor,
    AzureOpenAIEntityExtractor
)

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("entity_extractor_example")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Example texts for entity extraction
    texts = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. Tim Cook is the current CEO of Apple.",
        "The Golden Gate Bridge in San Francisco, California was completed in 1937. It has a length of 8,981 feet.",
        "SpaceX was founded by Elon Musk in 2002. The company launched its Falcon 1 rocket in 2008 and has since developed the Falcon 9, Falcon Heavy, and Starship rockets."
    ]
    
    # Example 1: Using Groq
    try:
        logger.info("--- Using Groq LLM Extractor ---")
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_extractor = GroqEntityExtractor(
            model_name="llama-3.1-70b-versatile",
            api_key=groq_api_key
        )
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}")
            entities = groq_extractor.extract_entities(text)
            print(f"Text {i+1} entities:")
            pprint(entities)
            print(f"Found {len(entities)} entities\n")
    except Exception as e:
        logger.error(f"Error with Groq extractor: {e}")
    
    # Example 2: Using Azure OpenAI
    try:
        logger.info("--- Using Azure OpenAI LLM Extractor ---")
        azure_extractor = AzureOpenAIEntityExtractor(
            model_name="gpt-4o",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        )
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}")
            entities = azure_extractor.extract_entities(text)
            print(f"Text {i+1} entities:")
            pprint(entities)
            print(f"Found {len(entities)} entities\n")
    except Exception as e:
        logger.error(f"Error with Azure OpenAI extractor: {e}")
    
    # Comparing different extractors on the same text
    text = "NASA's Mars rover Perseverance landed on Mars on February 18, 2021, after traveling 293 million miles."
    print("\n--- Comparing extractors on the same text ---")
    print(f"Text: {text}\n")
    
    try:
        groq_entities = groq_extractor.extract_entities(text)
        print("Groq extracted entities:")
        pprint(groq_entities)
        
        azure_entities = azure_extractor.extract_entities(text)
        print("\nAzure OpenAI extracted entities:")
        pprint(azure_entities)
    except Exception as e:
        logger.error(f"Error during comparison: {e}")

if __name__ == "__main__":
    main()
