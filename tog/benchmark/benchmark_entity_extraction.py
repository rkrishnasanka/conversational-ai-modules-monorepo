import sys
import os
import time
import json
import statistics
import csv
from datetime import datetime
from typing import List, Dict, Any
import logging

# Add parent directory to path to import the EntityExtractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.entity_extractor import EntityExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test queries of varying complexity and from different domains
TEST_QUERIES = [
    # Simple queries
    "What are the health benefits of Mediterranean diet?",
    "Tell me about electric cars and their environmental impact.",
    "How does machine learning work?",
    
    # Medium complexity
    "Can you explain the relationship between inflation, interest rates, and unemployment in macroeconomics?",
    "What are the main differences between quantum computing and classical computing architectures?",
    "How has climate change affected biodiversity in coral reef ecosystems?",
    
    # Complex queries
    "Discuss the ethical implications of using CRISPR gene editing technology in human embryos and potential long-term consequences for genetic diversity.",
    "Explain how the integration of artificial intelligence in healthcare diagnostics is changing patient outcomes, medical training, and healthcare economics.",
    "What were the major geopolitical factors that contributed to World War II, and how did they shape international relations during the subsequent Cold War era?"
]

# Different prompt strategies to test
PROMPT_STRATEGIES = {
    "default": None,  # Use the default prompt in the extractor
    
    "concise": """
    You are an entity extraction system. Identify only the main topics, concepts, and entities in the text.
    Return a JSON object with the format:
    {"entities": ["entity1", "entity2"]}
    """
    ,
    
    "detailed": """
    You are an advanced entity extraction system. Extract specific entities from the provided text.
    Focus on key topics, technologies, concepts, fields of study, and important terms.
    Identify both main topics and related subtopics.
    Exclude general concepts and focus on specific, meaningful entities.
    Return only a JSON object with this format:
    {"entities": ["entity1", "entity2", "entity3", ...]}
    """
    ,
    
    "hierarchical": """
    You are a hierarchical entity extraction system. For the given text:
    1. Identify the main primary topics (2-3 most important subjects)
    2. Identify secondary topics and concepts
    3. Return all entities as a flattened list in a JSON object:
    {"entities": ["primary_topic1", "primary_topic2", "secondary_topic1", ...]}
    """
}

class EntityExtractionBenchmark:
    """Benchmark the entity extraction functionality"""
    
    def __init__(self, output_dir: str = "./results"):
        """Initialize the benchmark with an output directory for results"""
        self.output_dir = output_dir
        self.extractor = EntityExtractor()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_result_file = None  # Track the last result file created
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            "queries": {},
            "summary": {}
        }
        
        # Use class-level references to query and prompt data
        self.TEST_QUERIES = TEST_QUERIES.copy()
        self.PROMPT_STRATEGIES = PROMPT_STRATEGIES.copy()
    
    def run_benchmark(self):
        """Run the full benchmark suite"""
        logger.info(f"Starting entity extraction benchmark at {self.timestamp}")
        
        all_times = []
        all_entity_counts = []
        
        # Test each query with each prompt strategy
        for query_index, query in enumerate(self.TEST_QUERIES):
            query_id = f"query_{query_index+1}"
            self.results["queries"][query_id] = {
                "text": query,
                "strategies": {}
            }
            
            logger.info(f"Testing query {query_index+1}/{len(self.TEST_QUERIES)}: {query[:50]}...")
            
            for strategy_name, prompt in self.PROMPT_STRATEGIES.items():
                logger.info(f"  Using strategy: {strategy_name}")
                
                # Run extraction with timing
                start_time = time.time()
                entities = self.extractor.extract_entities(query, prompt=prompt)
                end_time = time.time()
                
                # Calculate metrics
                elapsed_time = end_time - start_time
                entity_count = len(entities)
                
                all_times.append(elapsed_time)
                all_entity_counts.append(entity_count)
                
                # Store results
                self.results["queries"][query_id]["strategies"][strategy_name] = {
                    "entities": entities,
                    "count": entity_count,
                    "time_seconds": elapsed_time
                }
                
                logger.info(f"    Found {entity_count} entities in {elapsed_time:.2f} seconds")
        
        # Calculate summary statistics
        self.results["summary"] = {
            "total_queries": len(self.TEST_QUERIES),
            "total_strategies": len(self.PROMPT_STRATEGIES),
            "time_stats": {
                "min": min(all_times),
                "max": max(all_times),
                "mean": statistics.mean(all_times),
                "median": statistics.median(all_times),
                "total": sum(all_times)
            },
            "entity_count_stats": {
                "min": min(all_entity_counts),
                "max": max(all_entity_counts),
                "mean": statistics.mean(all_entity_counts),
                "median": statistics.median(all_entity_counts),
                "total": sum(all_entity_counts)
            }
        }
        
        # Save results
        self.save_results()
        self.generate_csv_report()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results as JSON"""
        result_path = os.path.join(self.output_dir, f"benchmark_results_{self.timestamp}.json")
        with open(result_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved detailed results to {result_path}")
        self.last_result_file = result_path  # Track the file we just created
    
    def generate_csv_report(self):
        """Generate CSV report for easy analysis"""
        csv_path = os.path.join(self.output_dir, f"benchmark_report_{self.timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['query_id', 'query_text', 'strategy', 'entity_count', 'time_seconds', 'entities']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for query_id, query_data in self.results["queries"].items():
                query_text = query_data["text"]
                
                for strategy_name, strategy_data in query_data["strategies"].items():
                    writer.writerow({
                        'query_id': query_id,
                        'query_text': query_text,
                        'strategy': strategy_name,
                        'entity_count': strategy_data["count"],
                        'time_seconds': f"{strategy_data['time_seconds']:.2f}",
                        'entities': ', '.join(strategy_data["entities"])
                    })
        
        logger.info(f"Generated CSV report at {csv_path}")


if __name__ == "__main__":
    benchmark = EntityExtractionBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n" + "=" * 50)
    print("ENTITY EXTRACTION BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total queries: {results['summary']['total_queries']}")
    print(f"Prompt strategies tested: {results['summary']['total_strategies']}")
    print(f"Total time: {results['summary']['time_stats']['total']:.2f} seconds")
    print(f"Average time per query: {results['summary']['time_stats']['mean']:.2f} seconds")
    print(f"Average entities per query: {results['summary']['entity_count_stats']['mean']:.1f}")
    print("=" * 50)
