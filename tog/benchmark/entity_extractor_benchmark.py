import time
import json
import statistics
import csv
import os
import sys
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import backoff

# Add parent directory to path to import tog modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tog.pipeline.entity_extractor import GroqEntityExtractor
from tog.models.entity import Entity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityExtractorBenchmark:
    def __init__(self, model_name: str, test_data_path: str = None, output_dir: str = "./benchmark_results",
                 max_retries: int = 5, initial_wait_time: float = 2.0, max_wait_time: float = 60.0,
                 token_cost_per_1k: float = 0.0):
        self.extractor = GroqEntityExtractor(model_name)
        self.test_data = self._load_test_data(test_data_path) if test_data_path else self._get_default_test_data()
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rate limiting and retry parameters
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.max_wait_time = max_wait_time
        self.request_count = 0
        self.error_count = 0
        
        # Cost tracking
        self.token_cost_per_1k = token_cost_per_1k
        self.total_tokens_used = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_test_data(self, path: str) -> List[Dict]:
        """
        Load test data from a JSON file.
        Format should be a list of dictionaries with 'text' and optionally 'ground_truth_entities' fields.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data formats
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    # Convert list of strings to list of dicts
                    return [{"text": text} for text in data]
                elif all(isinstance(item, dict) for item in data):
                    # Check that all items have a 'text' field
                    if all('text' in item for item in data):
                        return data
            
            logger.error(f"Invalid test data format. Expected list of strings or list of dicts with 'text' field")
            return self._get_default_test_data()
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return self._get_default_test_data()
    
    def _get_default_test_data(self) -> List[Dict]:
        """Return default test data with ground truth for accuracy testing."""
        default_tests = [
            # Simple queries with ground truth
            {
                "text": "Geoffrey Hinton is known as the father of Deep Learning.",
                "ground_truth_entities": [
                    {"name": "Geoffrey Hinton", "type": "PERSON"}, 
                    {"name": "Deep Learning", "type": "CONCEPT"}
                ]
            },
            {
                "text": "Tesla Inc. was founded by Elon Musk in Palo Alto, California.",
                "ground_truth_entities": [
                    {"name": "Tesla Inc.", "type": "ORGANIZATION"}, 
                    {"name": "Elon Musk", "type": "PERSON"},
                    {"name": "Palo Alto", "type": "LOCATION"}, 
                    {"name": "California", "type": "LOCATION"}
                ]
            },
            {
                "text": "The COVID-19 vaccine was developed by Pfizer and Moderna in 2020.",
                "ground_truth_entities": [
                    {"name": "COVID-19", "type": "CONCEPT"}, 
                    {"name": "Pfizer", "type": "ORGANIZATION"},
                    {"name": "Moderna", "type": "ORGANIZATION"}, 
                    {"name": "2020", "type": "DATE"}
                ]
            },
            # Medium complexity queries
            {
                "text": "Apple released the first iPhone in 2007 under the leadership of Steve Jobs, revolutionizing the smartphone industry.",
                "ground_truth_entities": [
                    {"name": "Apple", "type": "ORGANIZATION"}, 
                    {"name": "iPhone", "type": "PRODUCT"},
                    {"name": "2007", "type": "DATE"}, 
                    {"name": "Steve Jobs", "type": "PERSON"},
                    {"name": "smartphone industry", "type": "CONCEPT"}
                ]
            },
            {
                "text": "OpenAI's GPT-4 was released in March 2023 and demonstrated advanced capabilities in understanding and generating human language.",
                "ground_truth_entities": [
                    {"name": "OpenAI", "type": "ORGANIZATION"}, 
                    {"name": "GPT-4", "type": "PRODUCT"},
                    {"name": "March 2023", "type": "DATE"}, 
                    {"name": "human language", "type": "CONCEPT"}
                ]
            }
        ]
        
        return default_tests
    
    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Catch any exception that might be related to API rate limiting
        max_tries=5,  # Maximum number of retries
        factor=2,     # Exponential backoff factor
        jitter=backoff.full_jitter,  # Add jitter to prevent synchronized retries
    )
    def _extract_entities_with_retry(self, text: str) -> Tuple[List[Entity], Dict[str, int]]:
        """Extract entities with exponential backoff retry logic and token tracking"""
        try:
            # Adaptive rate limiting - add small delay between requests to prevent rate limiting
            if self.request_count > 0:
                # Gradually increase delay as more requests are made
                delay = min(0.1 * (self.request_count / 10), 1.0)
                time.sleep(delay)
            
            self.request_count += 1
            start_time = time.time()
            entities = self.extractor.extract_entities(text)
            end_time = time.time()
            
            # Estimate token usage based on text length
            input_tokens = len(text.split())
            # Assuming output is roughly proportional to number of entities found
            output_tokens = sum(len(entity.name.split()) for entity in entities) * 3
            tokens_used = {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}
            
            # Update total tokens used
            self.total_tokens_used += tokens_used["total"]
            
            return entities, tokens_used
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
    
    def _calculate_accuracy_metrics(self, extracted_entities: List[Entity], 
                                   ground_truth_entities: List[Dict]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for extracted entities against ground truth.
        
        We consider an entity correctly extracted if both the name and type match.
        """
        if not ground_truth_entities:
            return {
                "precision": None,
                "recall": None,
                "f1_score": None,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
        
        # Convert Entity objects to standardized format for comparison
        extracted = [(e.name.lower(), e.type.lower()) for e in extracted_entities]
        ground_truth = [(e["name"].lower(), e["type"].lower()) for e in ground_truth_entities]
        
        # Calculate true positives, false positives, and false negatives
        true_positives = len(set(extracted).intersection(set(ground_truth)))
        false_positives = len(extracted) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        
        # Calculate F1 score
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate the cost of processing based on token count and cost per 1K tokens."""
        if self.token_cost_per_1k <= 0:
            return 0.0
        return (tokens_used / 1000) * self.token_cost_per_1k
    
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
            "errors": [],
            "accuracy_metrics": {
                "precision": [],
                "recall": [],
                "f1_score": []
            },
            "token_usage": {
                "input": [],
                "output": [],
                "total": []
            },
            "costs": []
        }
        
        # Start time for throughput calculation
        start_time_overall = time.time()
        texts_processed = 0
        
        for i in range(iterations):
            logger.info(f"Running iteration {i+1}/{iterations}")
            iteration_results = []
            
            for idx, sample in enumerate(self.test_data):
                text = sample["text"]
                ground_truth = sample.get("ground_truth_entities", [])
                
                logger.info(f"  Processing text {idx+1}/{len(self.test_data)}: {text[:50]}...")
                
                try:
                    start_time = time.time()
                    entities, tokens = self._extract_entities_with_retry(text)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    execution_time_ms = execution_time * 1000  # Convert to milliseconds
                    
                    results["execution_times"].append(execution_time)
                    results["entity_counts"].append(len(entities))
                    
                    # Track token usage
                    results["token_usage"]["input"].append(tokens["input"])
                    results["token_usage"]["output"].append(tokens["output"])
                    results["token_usage"]["total"].append(tokens["total"])
                    
                    # Calculate cost
                    cost = self._calculate_cost(tokens["total"])
                    results["costs"].append(cost)
                    
                    # Calculate accuracy metrics if ground truth is available
                    accuracy_metrics = self._calculate_accuracy_metrics(entities, ground_truth)
                    
                    if accuracy_metrics["precision"] is not None:
                        results["accuracy_metrics"]["precision"].append(accuracy_metrics["precision"])
                        results["accuracy_metrics"]["recall"].append(accuracy_metrics["recall"])
                        results["accuracy_metrics"]["f1_score"].append(accuracy_metrics["f1_score"])
                    
                    sample_result = {
                        "text": text,
                        "execution_time": execution_time,
                        "execution_time_ms": execution_time_ms,
                        "entity_count": len(entities),
                        "entities": [e.__dict__ for e in entities],
                        "ground_truth_entities": ground_truth,
                        "accuracy_metrics": accuracy_metrics,
                        "tokens_used": tokens,
                        "cost": cost
                    }
                    iteration_results.append(sample_result)
                    
                    logger.info(f"    Found {len(entities)} entities in {execution_time:.2f} seconds ({execution_time_ms:.0f} ms)")
                    if accuracy_metrics["precision"] is not None:
                        logger.info(f"    Accuracy: P={accuracy_metrics['precision']:.2f}, R={accuracy_metrics['recall']:.2f}, F1={accuracy_metrics['f1_score']:.2f}")
                    
                    texts_processed += 1
                    
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
        
        # Calculate throughput
        end_time_overall = time.time()
        total_time = end_time_overall - start_time_overall
        throughput = texts_processed / total_time if total_time > 0 else 0
        
        # Calculate summary statistics
        if results["execution_times"]:
            results["summary"] = {
                "time_stats": {
                    "min": min(results["execution_times"]),
                    "max": max(results["execution_times"]),
                    "mean": statistics.mean(results["execution_times"]),
                    "median": statistics.median(results["execution_times"]),
                    "total": sum(results["execution_times"]),
                    "throughput": throughput,  # texts per second
                    "min_ms": min(results["execution_times"]) * 1000,  # in milliseconds
                    "mean_ms": statistics.mean(results["execution_times"]) * 1000  # in milliseconds
                },
                "entity_count_stats": {
                    "min": min(results["entity_counts"]),
                    "max": max(results["entity_counts"]),
                    "mean": statistics.mean(results["entity_counts"]),
                    "median": statistics.median(results["entity_counts"]),
                    "total": sum(results["entity_counts"])
                }
            }
            
            # Add accuracy metrics summary if available
            if results["accuracy_metrics"]["precision"]:
                results["summary"]["accuracy_stats"] = {
                    "precision": {
                        "min": min(results["accuracy_metrics"]["precision"]),
                        "max": max(results["accuracy_metrics"]["precision"]),
                        "mean": statistics.mean(results["accuracy_metrics"]["precision"]),
                        "median": statistics.median(results["accuracy_metrics"]["precision"])
                    },
                    "recall": {
                        "min": min(results["accuracy_metrics"]["recall"]),
                        "max": max(results["accuracy_metrics"]["recall"]),
                        "mean": statistics.mean(results["accuracy_metrics"]["recall"]),
                        "median": statistics.median(results["accuracy_metrics"]["recall"])
                    },
                    "f1_score": {
                        "min": min(results["accuracy_metrics"]["f1_score"]),
                        "max": max(results["accuracy_metrics"]["f1_score"]),
                        "mean": statistics.mean(results["accuracy_metrics"]["f1_score"]),
                        "median": statistics.median(results["accuracy_metrics"]["f1_score"])
                    }
                }
            
            # Add token usage and cost summary
            results["summary"]["token_usage"] = {
                "input": {
                    "total": sum(results["token_usage"]["input"]),
                    "mean": statistics.mean(results["token_usage"]["input"])
                },
                "output": {
                    "total": sum(results["token_usage"]["output"]),
                    "mean": statistics.mean(results["token_usage"]["output"])
                },
                "total": {
                    "total": sum(results["token_usage"]["total"]),
                    "mean": statistics.mean(results["token_usage"]["total"])
                }
            }
            
            results["summary"]["cost"] = {
                "total": sum(results["costs"]),
                "mean": statistics.mean(results["costs"]),
                "rate": f"${self.token_cost_per_1k:.6f} per 1K tokens"
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
                fieldnames = [
                    'text_id', 'text', 'execution_time', 'execution_time_ms', 'entity_count', 'entities',
                    'precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 'false_negatives',
                    'tokens_input', 'tokens_output', 'tokens_total', 'cost'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for idx, sample in enumerate(results["samples"]):
                    accuracy = sample.get("accuracy_metrics", {})
                    precision = accuracy.get("precision")
                    recall = accuracy.get("recall")
                    f1_score = accuracy.get("f1_score")
                    true_positives = accuracy.get("true_positives", 0)
                    false_positives = accuracy.get("false_positives", 0)
                    false_negatives = accuracy.get("false_negatives", 0)
                    
                    tokens = sample.get("tokens_used", {})
                    
                    writer.writerow({
                        'text_id': idx + 1,
                        'text': sample['text'],
                        'execution_time': f"{sample['execution_time']:.2f}",
                        'execution_time_ms': f"{sample.get('execution_time_ms', sample['execution_time'] * 1000):.0f}",
                        'entity_count': sample['entity_count'],
                        'entities': ', '.join([e['name'] for e in sample['entities']]),
                        'precision': f"{precision:.2f}" if precision is not None else "N/A",
                        'recall': f"{recall:.2f}" if recall is not None else "N/A",
                        'f1_score': f"{f1_score:.2f}" if f1_score is not None else "N/A",
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'false_negatives': false_negatives,
                        'tokens_input': tokens.get("input", 0),
                        'tokens_output': tokens.get("output", 0),
                        'tokens_total': tokens.get("total", 0),
                        'cost': f"${sample.get('cost', 0):.6f}"
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
                    "execution_time_ms": sample.get("execution_time_ms", sample["execution_time"] * 1000),
                    "entity_count": sample["entity_count"],
                    "tokens_total": sample.get("tokens_used", {}).get("total", 0),
                    "cost": sample.get("cost", 0)
                }
                
                # Add accuracy metrics if available
                if sample.get("accuracy_metrics", {}).get("precision") is not None:
                    sample_dict["precision"] = sample["accuracy_metrics"]["precision"]
                    sample_dict["recall"] = sample["accuracy_metrics"]["recall"]
                    sample_dict["f1_score"] = sample["accuracy_metrics"]["f1_score"]
                    sample_dict["true_positives"] = sample["accuracy_metrics"]["true_positives"]
                    sample_dict["false_positives"] = sample["accuracy_metrics"]["false_positives"]
                    sample_dict["false_negatives"] = sample["accuracy_metrics"]["false_negatives"]
                
                samples_data.append(sample_dict)
            
            df = pd.DataFrame(samples_data)
            
            # Set style
            sns.set(style="whitegrid")

            # 1. LATENCY METRICS VISUALIZATION
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            sns.histplot(df["execution_time_ms"], kde=True)
            plt.title("Response Time Distribution")
            plt.xlabel("Response Time (ms)")
            plt.ylabel("Frequency")
            
            plt.subplot(2, 2, 2)
            sns.boxplot(y=df["execution_time_ms"])
            plt.title("Response Time Boxplot")
            plt.ylabel("Response Time (ms)")
            
            plt.subplot(2, 2, 3)
            plt.bar(["Throughput"], [results["summary"]["time_stats"]["throughput"]])
            plt.title("Throughput (texts/second)")
            plt.ylabel("Texts per Second")
            
            plt.subplot(2, 2, 4)
            sns.scatterplot(x="entity_count", y="execution_time_ms", data=df)
            plt.title("Response Time vs. Entity Count")
            plt.xlabel("Number of Entities")
            plt.ylabel("Response Time (ms)")
            
            plt.tight_layout()
            latency_plot_path = os.path.join(output_dir, "latency_metrics.png")
            plt.savefig(latency_plot_path)
            generated_files.append(latency_plot_path)
            plt.close()
            
            # 2. ACCURACY METRICS VISUALIZATION
            if "precision" in df.columns:
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 2, 1)
                accuracy_metrics = ['precision', 'recall', 'f1_score']
                means = [df[metric].mean() for metric in accuracy_metrics]
                plt.bar(accuracy_metrics, means)
                plt.title("Average Accuracy Metrics")
                plt.ylim(0, 1)
                
                plt.subplot(2, 2, 2)
                confusion_data = [
                    df["true_positives"].sum(),
                    df["false_positives"].sum(),
                    df["false_negatives"].sum()
                ]
                plt.bar(["True Positives", "False Positives", "False Negatives"], confusion_data)
                plt.title("Confusion Matrix Components")
                
                plt.subplot(2, 2, 3)
                for idx, metric in enumerate(accuracy_metrics):
                    plt.bar(
                        [idx + i*0.25 for i in range(len(df))], 
                        df[metric], 
                        width=0.2,
                        alpha=0.7
                    )
                plt.xticks([i + 0.25 for i in range(len(accuracy_metrics))], accuracy_metrics)
                plt.title("Accuracy Metrics per Sample")
                plt.ylim(0, 1)
                
                plt.subplot(2, 2, 4)
                # Create a correlation heatmap for accuracy and other metrics
                corr_metrics = ['precision', 'recall', 'f1_score', 'execution_time_ms', 'entity_count']
                if len(df) >= 3:  # Need at least 3 samples for meaningful correlation
                    corr = df[corr_metrics].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title("Metric Correlations")
                else:
                    plt.text(0.5, 0.5, "Not enough samples for correlation analysis",
                            ha='center', va='center')
                
                plt.tight_layout()
                accuracy_plot_path = os.path.join(output_dir, "accuracy_metrics.png")
                plt.savefig(accuracy_plot_path)
                generated_files.append(accuracy_plot_path)
                plt.close()
            
            # 3. COST AND TOKEN USAGE VISUALIZATION
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.histplot(df["tokens_total"], kde=True)
            plt.title("Token Usage Distribution")
            plt.xlabel("Tokens per Request")
            plt.ylabel("Frequency")
            
            plt.subplot(2, 2, 2)
            sns.histplot(df["cost"], kde=True)
            plt.title("Cost Distribution")
            plt.xlabel("Cost per Request ($)")
            plt.ylabel("Frequency")
            
            plt.subplot(2, 2, 3)
            token_summary = [
                statistics.mean(results["token_usage"]["input"]),
                statistics.mean(results["token_usage"]["output"]),
                statistics.mean(results["token_usage"]["total"])
            ]
            plt.bar(["Input", "Output", "Total"], token_summary)
            plt.title("Average Token Usage by Type")
            plt.ylabel("Average Token Count")
            
            plt.subplot(2, 2, 4)
            sns.scatterplot(x="tokens_total", y="cost", data=df)
            plt.title("Cost vs. Token Usage")
            plt.xlabel("Total Tokens")
            plt.ylabel("Cost ($)")
            
            plt.tight_layout()
            cost_plot_path = os.path.join(output_dir, "cost_metrics.png")
            plt.savefig(cost_plot_path)
            generated_files.append(cost_plot_path)
            plt.close()
            
            # 4. COMPREHENSIVE DASHBOARD
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            
            # Row 1: Latency metrics
            axes[0, 0].bar(["Average Response Time (ms)"], [results["summary"]["time_stats"]["mean_ms"]])
            axes[0, 0].set_title("Average Response Time")
            
            axes[0, 1].bar(["Throughput"], [results["summary"]["time_stats"]["throughput"]])
            axes[0, 1].set_title("Throughput (texts/second)")
            
            # Row 2: Accuracy metrics (if available)
            if "accuracy_stats" in results["summary"]:
                acc_data = {
                    'Precision': results["summary"]["accuracy_stats"]["precision"]["mean"],
                    'Recall': results["summary"]["accuracy_stats"]["recall"]["mean"],
                    'F1 Score': results["summary"]["accuracy_stats"]["f1_score"]["mean"]
                }
                axes[1, 0].bar(acc_data.keys(), acc_data.values())
                axes[1, 0].set_title("Accuracy Metrics")
                axes[1, 0].set_ylim(0, 1)
                
                # Accuracy trend over samples
                if len(df) > 1 and "precision" in df.columns:
                    for metric, color, marker in zip(['precision', 'recall', 'f1_score'], 
                                                   ['blue', 'green', 'red'],
                                                   ['o', 's', '^']):
                        axes[1, 1].plot(df.index, df[metric], marker=marker, 
                                     linestyle='-', label=metric.capitalize(), color=color)
                    axes[1, 1].set_title("Accuracy Metrics by Sample")
                    axes[1, 1].set_xlabel("Sample ID")
                    axes[1, 1].set_ylim(0, 1)
                    axes[1, 1].legend()
                else:
                    axes[1, 1].text(0.5, 0.5, "Insufficient data for trend",
                                 ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                for ax in axes[1, :]:
                    ax.text(0.5, 0.5, "No accuracy metrics available", 
                         ha='center', va='center', transform=ax.transAxes)
            
            # Row 3: Cost metrics
            token_data = [
                results["summary"]["token_usage"]["input"]["mean"],
                results["summary"]["token_usage"]["output"]["mean"],
                results["summary"]["token_usage"]["total"]["mean"]
            ]
            axes[2, 0].bar(["Input", "Output", "Total"], token_data)
            axes[2, 0].set_title("Average Token Usage")
            
            if results["costs"]:
                axes[2, 1].bar(["Avg Cost/Request"], [results["summary"]["cost"]["mean"]])
                axes[2, 1].set_title(f"Average Cost per Request (${results['summary']['cost']['mean']:.6f})")
            else:
                axes[2, 1].text(0.5, 0.5, "No cost data available", 
                             ha='center', va='center', transform=axes[2, 1].transAxes)
            
            plt.tight_layout()
            dashboard_path = os.path.join(output_dir, "metrics_dashboard.png")
            plt.savefig(dashboard_path)
            generated_files.append(dashboard_path)
            plt.close()
            
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
    parser.add_argument('--token_cost', type=float, default=0.0,
                        help='Cost per 1K tokens for the model (in USD)')
    
    args = parser.parse_args()
    
    benchmark = EntityExtractorBenchmark(
        model_name=args.model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        initial_wait_time=args.initial_wait,
        token_cost_per_1k=args.token_cost
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
        print(f"Average response time: {results['summary']['time_stats']['mean_ms']:.0f} ms")
        print(f"Throughput: {results['summary']['time_stats']['throughput']:.2f} texts/second")
        print(f"Average entities found: {results['summary']['entity_count_stats']['mean']:.1f}")
        
        # Print accuracy metrics if available
        if "accuracy_stats" in results["summary"]:
            print("\nAccuracy Metrics:")
            print(f"Precision: {results['summary']['accuracy_stats']['precision']['mean']:.2f}")
            print(f"Recall: {results['summary']['accuracy_stats']['recall']['mean']:.2f}")
            print(f"F1 Score: {results['summary']['accuracy_stats']['f1_score']['mean']:.2f}")
        
        # Print token usage and cost if applicable
        if args.token_cost > 0:
            print("\nToken Usage and Cost:")
            print(f"Average tokens per request: {results['summary']['token_usage']['total']['mean']:.0f}")
            print(f"Total cost: ${results['summary']['cost']['total']:.4f}")
            print(f"Average cost per text: ${results['summary']['cost']['mean']:.4f}")
    else:
        print("No successful extractions completed.")
    
    print(f"Total API requests: {results['request_stats']['total_requests']}")
    print(f"Failed requests: {results['request_stats']['error_count']} ({results['request_stats']['error_rate']:.1%})")
    
    print(f"Results saved to: {json_path}")
    print(f"CSV report: {csv_path}")
    print("=" * 50)
