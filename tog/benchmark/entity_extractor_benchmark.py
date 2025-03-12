import time
import json
import statistics
import csv
import os
import sys
import random
from typing import Dict, List, Any
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import backoff

# Add parent directory to path to import tog modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tog.src.pipeline.entity_extractor import GroqEnityExtractor
from tog.src.models.entity import Entity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityExtractorBenchmark:
    def __init__(self, model_name: str, test_data_path: str = None, output_dir: str = "./benchmark_results",
                 max_retries: int = 5, initial_wait_time: float = 2.0, max_wait_time: float = 60.0):
        self.extractor = GroqEnityExtractor(model_name)
        self.test_data = self._load_test_data(test_data_path) if test_data_path else self._get_default_test_data()
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rate limiting and retry parameters
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.max_wait_time = max_wait_time
        self.request_count = 0
        self.error_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_test_data(self, path: str) -> List[str]:
        """Load test data from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return self._get_default_test_data()
    
    def _get_default_test_data(self) -> List[str]:
        """Return default test data if no file is provided."""
        return [
            # Simple queries
            "Geoffrey Hinton is known as the father of Deep Learning.",
            "Tesla Inc. was founded by Elon Musk in Palo Alto, California.",
            "The COVID-19 vaccine was developed by Pfizer and Moderna in 2020.",
            
            # Medium complexity queries
            "Apple released the first iPhone in 2007 under the leadership of Steve Jobs, revolutionizing the smartphone industry.",
            "OpenAI's GPT-4 was released in March 2023 and demonstrated advanced capabilities in understanding and generating human language.",
            "The European Union's GDPR regulations have significantly impacted how companies handle personal data since implementation in 2018.",
            
            # Complex queries
            "The human genome project was completed in 2003 after 13 years of international collaboration, sequencing approximately 3 billion base pairs in human DNA.",
            "Climate change has led to rising sea levels, increased frequency of extreme weather events, and disruption of ecosystems worldwide according to the IPCC report.",
            "Quantum computing leverages principles of quantum mechanics like superposition and entanglement to perform computations that would be infeasible on classical computers."
        ]
    
    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Catch any exception that might be related to API rate limiting
        max_tries=5,  # Maximum number of retries
        factor=2,     # Exponential backoff factor
        jitter=backoff.full_jitter,  # Add jitter to prevent synchronized retries
    )
    def _extract_entities_with_retry(self, text: str):
        """Extract entities with exponential backoff retry logic"""
        try:
            # Adaptive rate limiting - add small delay between requests to prevent rate limiting
            if self.request_count > 0:
                # Gradually increase delay as more requests are made
                delay = min(0.1 * (self.request_count / 10), 1.0)
                time.sleep(delay)
            
            self.request_count += 1
            return self.extractor.extract_entities(text)
        except Exception as e:
            self.error_count += 1
            error_msg = str(e).lower()
            
            # Check if this is a rate limiting error
            if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit exceeded. Backing off before retry. Error: {e}")
                # Increase delay for all future requests
                time.sleep(random.uniform(2.0, 5.0))  # Random delay to desynchronize requests
            else:
                logger.error(f"Error extracting entities: {e}")
            
            # Propagate the exception to trigger backoff
            raise
    
    def run_benchmark(self, iterations: int = 1, continue_on_error: bool = True) -> Dict[str, Any]:
        """Run the benchmark for the specified number of iterations."""
        logger.info(f"Starting entity extraction benchmark with model {self.extractor.llm.model_name}")
        
        results = {
            "model": self.extractor.llm.model_name,
            "total_texts": len(self.test_data),
            "iterations": iterations,
            "execution_times": [],
            "entity_counts": [],
            "samples": [],
            "timestamp": self.timestamp,
            "errors": []
        }
        
        for i in range(iterations):
            logger.info(f"Running iteration {i+1}/{iterations}")
            iteration_results = []
            
            for idx, text in enumerate(self.test_data):
                logger.info(f"  Processing text {idx+1}/{len(self.test_data)}: {text[:50]}...")
                
                try:
                    start_time = time.time()
                    entities = self._extract_entities_with_retry(text)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    results["execution_times"].append(execution_time)
                    results["entity_counts"].append(len(entities))
                    
                    sample_result = {
                        "text": text,
                        "execution_time": execution_time,
                        "entity_count": len(entities),
                        "entities": [e.__dict__ for e in entities]
                    }
                    iteration_results.append(sample_result)
                    
                    logger.info(f"    Found {len(entities)} entities in {execution_time:.2f} seconds")
                    
                except Exception as e:
                    error_info = {
                        "text": text,
                        "iteration": i + 1,
                        "text_index": idx + 1,
                        "error": str(e)
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Failed to process text: {error_info}")
                    
                    if not continue_on_error:
                        logger.error("Stopping benchmark due to error.")
                        break
                
                # Add a small delay between texts to avoid rate limiting
                time.sleep(0.5)
            
            if not continue_on_error and results["errors"]:
                break
                
            results["samples"].extend(iteration_results)
            
            # Add a longer delay between iterations
            if i < iterations - 1:
                delay = random.uniform(3.0, 7.0)
                logger.info(f"Waiting {delay:.1f} seconds before next iteration...")
                time.sleep(delay)
        
        # Calculate summary statistics
        if results["execution_times"]:
            results["summary"] = {
                "time_stats": {
                    "min": min(results["execution_times"]),
                    "max": max(results["execution_times"]),
                    "mean": statistics.mean(results["execution_times"]),
                    "median": statistics.median(results["execution_times"]),
                    "total": sum(results["execution_times"])
                },
                "entity_count_stats": {
                    "min": min(results["entity_counts"]),
                    "max": max(results["entity_counts"]),
                    "mean": statistics.mean(results["entity_counts"]),
                    "median": statistics.median(results["entity_counts"]),
                    "total": sum(results["entity_counts"])
                }
            }
        else:
            results["summary"] = {
                "time_stats": {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "total": 0
                },
                "entity_count_stats": {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "total": 0
                }
            }
        
        # Add request statistics
        results["request_stats"] = {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save benchmark results to a JSON file."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"benchmark_results_{self.timestamp}.json")
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def generate_csv_report(self, results: Dict[str, Any], csv_path: str = None) -> str:
        """Generate a CSV report for easier analysis."""
        if csv_path is None:
            csv_path = os.path.join(self.output_dir, f"benchmark_report_{self.timestamp}.csv")
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['text_id', 'text', 'execution_time', 'entity_count', 'entities']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for idx, sample in enumerate(results["samples"]):
                    writer.writerow({
                        'text_id': idx + 1,
                        'text': sample['text'],
                        'execution_time': f"{sample['execution_time']:.2f}",
                        'entity_count': sample['entity_count'],
                        'entities': ', '.join([e['name'] for e in sample['entities']])
                    })
            
            logger.info(f"Generated CSV report at {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            return None
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = None) -> List[str]:
        """Generate visualizations from benchmark results."""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"visualizations_{self.timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        # Skip visualization if there are no results
        if not results["execution_times"]:
            logger.warning("No data to visualize. Skipping visualization.")
            return generated_files
        
        try:
            # Convert samples to DataFrame for easier plotting
            samples_data = []
            for sample in results["samples"]:
                sample_dict = {
                    "text": sample["text"][:50] + "...",  # Truncate text for display
                    "execution_time": sample["execution_time"],
                    "entity_count": sample["entity_count"]
                }
                samples_data.append(sample_dict)
            
            df = pd.DataFrame(samples_data)
            
            # Set style
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 10))
            
            # 1. Execution time distribution
            plt.subplot(2, 2, 1)
            sns.histplot(df["execution_time"], kde=True)
            plt.title("Distribution of Execution Times")
            plt.xlabel("Execution Time (seconds)")
            plt.ylabel("Frequency")
            
            # 2. Entity count distribution
            plt.subplot(2, 2, 2)
            sns.histplot(df["entity_count"], kde=True, discrete=True)
            plt.title("Distribution of Entity Counts")
            plt.xlabel("Number of Entities")
            plt.ylabel("Frequency")
            
            # 3. Execution time per text
            plt.subplot(2, 2, 3)
            sns.barplot(x=df.index, y="execution_time", data=df)
            plt.title("Execution Time per Text")
            plt.xlabel("Text ID")
            plt.ylabel("Execution Time (seconds)")
            plt.xticks(rotation=45)
            
            # 4. Entity count vs. execution time scatter plot
            plt.subplot(2, 2, 4)
            sns.scatterplot(x="execution_time", y="entity_count", data=df)
            plt.title("Entity Count vs. Execution Time")
            plt.xlabel("Execution Time (seconds)")
            plt.ylabel("Number of Entities")
            
            plt.tight_layout()
            
            # Save the combined plot
            combined_plot_path = os.path.join(output_dir, "combined_metrics.png")
            plt.savefig(combined_plot_path)
            generated_files.append(combined_plot_path)
            
            # Create summary statistics plot
            plt.figure(figsize=(10, 6))
            summary = results["summary"]
            
            # Time statistics
            time_stats = summary["time_stats"]
            time_values = [time_stats["min"], time_stats["mean"], time_stats["median"], time_stats["max"]]
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=["Min", "Mean", "Median", "Max"], y=time_values)
            plt.title("Execution Time Statistics")
            plt.ylabel("Time (seconds)")
            
            # Entity count statistics
            entity_stats = summary["entity_count_stats"]
            entity_values = [entity_stats["min"], entity_stats["mean"], entity_stats["median"], entity_stats["max"]]
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=["Min", "Mean", "Median", "Max"], y=entity_values)
            plt.title("Entity Count Statistics")
            plt.ylabel("Number of Entities")
            
            plt.tight_layout()
            
            # Save summary statistics plot
            summary_plot_path = os.path.join(output_dir, "summary_statistics.png")
            plt.savefig(summary_plot_path)
            generated_files.append(summary_plot_path)
            
            # Create error analysis plot if errors occurred
            if results.get("errors"):
                plt.figure(figsize=(8, 6))
                error_data = {
                    "Total Requests": results["request_stats"]["total_requests"],
                    "Successful": results["request_stats"]["total_requests"] - results["request_stats"]["error_count"],
                    "Failed": results["request_stats"]["error_count"]
                }
                
                sns.barplot(x=list(error_data.keys()), y=list(error_data.values()))
                plt.title("API Request Status")
                plt.ylabel("Count")
                
                # Add error rate annotation
                error_rate = results["request_stats"]["error_rate"]
                plt.annotate(f"Error Rate: {error_rate:.1%}", 
                             xy=(0.5, 0.9), 
                             xycoords='figure fraction',
                             ha='center')
                
                error_plot_path = os.path.join(output_dir, "error_analysis.png")
                plt.savefig(error_plot_path)
                generated_files.append(error_plot_path)
            
            plt.close('all')
            logger.info(f"Created visualizations in {output_dir}")
            
            return generated_files
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return generated_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark the GroqEntityExtractor')
    parser.add_argument('--model', type=str, default="mixtral-8x7b-32768", 
                        help='Model name to use for extraction')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations to run the benchmark')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data JSON file')
    parser.add_argument('--output_dir', type=str, default="./benchmark_results",
                        help='Output directory for benchmark results and visualizations')
    parser.add_argument('--skip_visualizations', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum number of retries for API failures')
    parser.add_argument('--initial_wait', type=float, default=2.0,
                        help='Initial wait time (seconds) between retries')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='Continue benchmarking even if some texts fail')
    
    args = parser.parse_args()
    
    benchmark = EntityExtractorBenchmark(
        model_name=args.model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        initial_wait_time=args.initial_wait
    )
    
    print(f"Starting benchmark with model {args.model}")
    results = benchmark.run_benchmark(
        iterations=args.iterations,
        continue_on_error=args.continue_on_error
    )
    
    # Save results
    json_path = benchmark.save_results(results)
    csv_path = benchmark.generate_csv_report(results)
    
    # Create visualizations
    if not args.skip_visualizations:
        try:
            viz_files = benchmark.create_visualizations(results)
            print(f"Generated {len(viz_files)} visualization files")
        except ImportError as e:
            print(f"Could not generate visualizations: {e}")
            print("Make sure matplotlib, seaborn and pandas are installed.")
    
    # Print summary
    print("\n" + "=" * 50)
    print("ENTITY EXTRACTION BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Model: {results['model']}")
    print(f"Total texts: {results['total_texts']}")
    print(f"Iterations: {results['iterations']}")
    
    if results["execution_times"]:
        print(f"Total execution time: {results['summary']['time_stats']['total']:.2f} seconds")
        print(f"Average execution time: {results['summary']['time_stats']['mean']:.2f} seconds")
        print(f"Average entities found: {results['summary']['entity_count_stats']['mean']:.1f}")
    else:
        print("No successful extractions completed.")
    
    print(f"Total API requests: {results['request_stats']['total_requests']}")
    print(f"Failed requests: {results['request_stats']['error_count']} ({results['request_stats']['error_rate']:.1%})")
    
    print(f"Results saved to: {json_path}")
    print(f"CSV report: {csv_path}")
    print("=" * 50)
