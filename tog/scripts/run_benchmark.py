#!/usr/bin/env python
"""
Script to run entity extraction benchmarks and generate visualization plots.
"""
import os
import sys
import argparse
import time
from pathlib import Path
import logging
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_runner")

def run_benchmark(num_queries=None, output_dir=None, strategies=None, include_plots=True):
    """Run the entity extraction benchmarks with optional parameters"""
    try:
        # Dynamic import to ensure all paths are set up correctly
        from tog.benchmark.benchmark_entity_extraction import EntityExtractionBenchmark, TEST_QUERIES, PROMPT_STRATEGIES

        # Create benchmark output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(project_root, "tog", "benchmark", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply query limit if specified
        limited_queries = TEST_QUERIES
        if num_queries and 0 < num_queries < len(TEST_QUERIES):
            limited_queries = TEST_QUERIES[:num_queries]
            logger.info(f"Running benchmark with {num_queries} queries (limited from {len(TEST_QUERIES)})")
        
        # Create benchmark instance
        benchmark = EntityExtractionBenchmark(output_dir=output_dir)
        
        # Override test queries if we're using a limited set
        if num_queries:
            benchmark.TEST_QUERIES = limited_queries
        
        # Override strategies if specified
        if strategies:
            selected_strategies = {}
            for strategy in strategies:
                if strategy in PROMPT_STRATEGIES:
                    selected_strategies[strategy] = PROMPT_STRATEGIES[strategy]
                else:
                    logger.warning(f"Unknown strategy '{strategy}' - available strategies: {list(PROMPT_STRATEGIES.keys())}")
            
            if selected_strategies:
                benchmark.PROMPT_STRATEGIES = selected_strategies
                logger.info(f"Using selected strategies: {list(selected_strategies.keys())}")
        
        # Run the benchmark
        start_time = time.time()
        results = benchmark.run_benchmark()
        elapsed = time.time() - start_time
        
        # Log benchmark completion
        logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        # Generate plots if requested
        if include_plots:
            logger.info("Generating benchmark plots...")
            from tog.benchmark.plot_benchmark_results import main as plot_main
            # Find the latest results file
            result_file = benchmark.last_result_file
            plot_output_dir = os.path.join(project_root, "tog", "benchmark", "plots")
            
            # Use sys.argv hack to pass arguments to plot_main
            sys_argv_backup = sys.argv
            sys.argv = [sys.argv[0], 
                        "--file", result_file,
                        "--output", plot_output_dir]
            plot_main()
            sys.argv = sys_argv_backup
            
            logger.info(f"Plots saved to {plot_output_dir}")
        
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import benchmark modules: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return None

def main():
    """Parse command line arguments and run benchmarks"""
    parser = argparse.ArgumentParser(description="Run entity extraction benchmarks")
    parser.add_argument("--queries", type=int, help="Number of queries to benchmark (default: all)")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--strategies", nargs="+", help="List of strategies to benchmark")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    logger.info("Starting entity extraction benchmark")
    logger.info(f"Using Python {sys.version}")
    
    # Run the benchmark
    results = run_benchmark(
        num_queries=args.queries,
        output_dir=args.output,
        strategies=args.strategies,
        include_plots=not args.no_plots
    )
    
    if results:
        # Print a nice summary
        print("\n" + "=" * 60)
        print(" " * 15 + "ENTITY EXTRACTION BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total queries: {results['summary']['total_queries']}")
        print(f"Strategies tested: {results['summary']['total_strategies']}")
        print(f"Total extraction time: {results['summary']['time_stats']['total']:.2f} seconds")
        print(f"Average extraction time: {results['summary']['time_stats']['mean']:.2f} seconds per query")
        print(f"Average entities found: {results['summary']['entity_count_stats']['mean']:.1f} per query")
        print("=" * 60)
        print(f"\nDetailed results stored in: {os.path.join(project_root, 'tog', 'benchmark', 'results')}")
        if not args.no_plots:
            print(f"Visualization plots stored in: {os.path.join(project_root, 'tog', 'benchmark', 'plots')}")
        print("\nRun with --help for more options")
    else:
        print("\nBenchmark failed. Check the logs for details.")

if __name__ == "__main__":
    main()
