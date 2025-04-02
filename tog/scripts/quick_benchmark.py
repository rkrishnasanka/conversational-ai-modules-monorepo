#!/usr/bin/env python
"""
Script to run a quick benchmark with minimal queries and default settings.
Useful for a quick test of entity extraction performance.
"""
import os
import sys
from pathlib import Path
import time
import logging

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quick_benchmark")

# Quick test queries - simpler and fewer than the full benchmark
QUICK_TEST_QUERIES = [
    "What are the main benefits of exercise for mental health?",
    "Tell me about SpaceX's Starship program",
    "How does climate change impact biodiversity?"
]

def run_quick_benchmark():
    """Run a quick benchmark on entity extraction"""
    try:
        # Import entity extractor
        from tog.entity_extraction import EntityExtractor
        
        # Create extractor
        extractor = EntityExtractor()
        
        print("\n===== QUICK ENTITY EXTRACTION BENCHMARK =====\n")
        
        total_time = 0
        total_entities = 0
        
        # Process each query
        for i, query in enumerate(QUICK_TEST_QUERIES):
            print(f"\nQuery {i+1}: '{query}'\n{'-' * 50}")
            
            # Extract entities and time it
            start_time = time.time()
            entities = extractor.extract_entities(query)
            elapsed = time.time() - start_time
            
            # Print results
            print(f"Extracted {len(entities)} entities in {elapsed:.2f} seconds:")
            for entity in entities:
                print(f"  - {entity}")
            
            total_time += elapsed
            total_entities += len(entities)
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Queries processed: {len(QUICK_TEST_QUERIES)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / len(QUICK_TEST_QUERIES):.2f} seconds")
        print(f"Total entities extracted: {total_entities}")
        print(f"Average entities per query: {total_entities / len(QUICK_TEST_QUERIES):.1f}")
        print("=" * 50)
        
        # Test with a custom prompt
        print("\n===== TESTING WITH CUSTOM PROMPT =====")
        custom_prompt = """
        You are an expert topic analyzer. Extract key topics from the text.
        Return only a JSON object with the format:
        {"entities": ["topic1", "topic2", "topic3"]}
        Focus on the most important topics and be specific.
        """
        
        test_query = "How do electric vehicles compare to traditional combustion engines in terms of environmental impact and performance?"
        print(f"\nQuery: '{test_query}'")
        
        start_time = time.time()
        custom_entities = extractor.extract_entities(test_query, prompt=custom_prompt)
        elapsed = time.time() - start_time
        
        print(f"Custom prompt extracted {len(custom_entities)} entities in {elapsed:.2f} seconds:")
        for entity in custom_entities:
            print(f"  - {entity}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running quick benchmark: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_benchmark()
    sys.exit(0 if success else 1)
