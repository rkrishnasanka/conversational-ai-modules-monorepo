#!/usr/bin/env python
"""
Script to compare different prompts for entity extraction on the same query.
Helps identify which prompt strategies work best for particular use cases.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import time
from tabulate import tabulate
import logging

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prompt_comparison")

# Sample prompt strategies for comparison
PROMPT_STRATEGIES = {
    "simple": """
    Extract key entities from the text.
    Return a JSON object with the format: {"entities": ["entity1", "entity2"]}
    """,
    
    "detailed": """
    You are an advanced entity extraction system. Extract specific entities from the text.
    Focus on key topics, technologies, concepts, fields of study, and important terms.
    Return a JSON object with this format: {"entities": ["entity1", "entity2", "entity3", ...]}
    """,
    
    "educational": """
    You are an educational topic extractor. Analyze the given text and identify:
    1. Main subjects and academic disciplines
    2. Key concepts that would be important for students to understand
    3. Specific topics that might appear in a course syllabus
    Return these as a JSON list: {"entities": ["subject1", "concept1", "topic1", ...]}
    """,
    
    "technical": """
    You are a technical entity extractor. From the provided text, identify:
    1. Technologies, programming languages, frameworks, or technical methodologies
    2. Technical concepts and principles
    3. Technical domains and specializations
    Return these as a JSON list: {"entities": ["technology1", "concept1", "domain1", ...]}
    """
}

def compare_prompts(query, prompt_dict=None, output_file=None):
    """Compare different prompt strategies on the same query"""
    try:
        # Import entity extractor
        from tog.src.entity_extraction import EntityExtractor
        
        # Use provided prompts or default to the predefined strategies
        prompts = prompt_dict or PROMPT_STRATEGIES
        
        # Create extractor
        extractor = EntityExtractor()
        
        # Initialize results dictionary
        results = {
            "query": query,
            "prompts": {}
        }
        
        # Create a table for displaying results
        table_data = []
        
        print(f"\n===== COMPARING PROMPTS ON QUERY =====\n")
        print(f"Query: '{query}'\n")
        
        # Process with each prompt strategy
        for name, prompt in prompts.items():
            print(f"Testing prompt: {name}")
            
            # Extract entities and time it
            start_time = time.time()
            entities = extractor.extract_entities(query, prompt=prompt)
            elapsed = time.time() - start_time
            
            # Save results
            results["prompts"][name] = {
                "prompt_text": prompt,
                "entities": entities,
                "count": len(entities),
                "time_seconds": elapsed
            }
            
            # Add to table
            table_data.append([
                name,
                len(entities),
                f"{elapsed:.2f}",
                ", ".join(entities[:5]) + ("..." if len(entities) > 5 else "")
            ])
        
        # Print results in a table
        print("\n" + tabulate(
            table_data, 
            headers=["Prompt Strategy", "Entity Count", "Time (s)", "First 5 Entities"],
            tablefmt="grid"
        ))
        
        # Save results to file if requested
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to {output_file}")
        
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        if "tabulate" in str(e):
            print("Please install the 'tabulate' package: pip install tabulate")
        return None
    except Exception as e:
        logger.error(f"Error comparing prompts: {e}")
        return None

def main():
    """Parse command line arguments and run prompt comparison"""
    parser = argparse.ArgumentParser(description="Compare different prompt strategies for entity extraction")
    parser.add_argument("query", nargs="?", help="Query text to extract entities from")
    parser.add_argument("--file", help="Output file for detailed results (JSON)")
    parser.add_argument("--prompt-file", help="JSON file containing custom prompts to test")
    
    args = parser.parse_args()
    
    # Use default query if none provided
    if not args.query:
        args.query = "Explain how machine learning algorithms are being applied to solve climate change challenges in agriculture and renewable energy production."
        print(f"No query provided, using default query:\n\"{args.query}\"")
    
    # Load custom prompts if specified
    custom_prompts = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                custom_prompts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            return 1
    
    # Default output file if not specified
    if not args.file:
        output_dir = os.path.join(project_root, "tog", "benchmark", "results")
        os.makedirs(output_dir, exist_ok=True)
        args.file = os.path.join(output_dir, f"prompt_comparison_{int(time.time())}.json")
    
    # Run comparison
    results = compare_prompts(args.query, prompt_dict=custom_prompts, output_file=args.file)
    
    return 0 if results else 1

if __name__ == "__main__":
    sys.exit(main())
