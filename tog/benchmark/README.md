# Entity Extraction Benchmarking Suite

This directory contains tools for benchmarking the performance of the entity extraction module. The benchmarks evaluate extraction quality, response time, and efficiency across different prompt strategies and query types.

## Benchmarking Files

- `benchmark_entity_extraction.py`: Main benchmark implementation
- `plot_benchmark_results.py`: Generates visualizations from benchmark results
- `results/`: Directory where benchmark results are stored
- `plots/`: Directory where benchmark visualization plots are stored

## Running Benchmarks

### Using Python Scripts (Recommended)

The TOG project includes convenience scripts for running benchmarks:

```bash
# Run the full benchmark suite
python -m tog.scripts.run_benchmark

# Run a quick benchmark (faster)
python -m tog.scripts.quick_benchmark

# Compare different prompt strategies on a specific query
python -m tog.scripts.compare_prompts "How does machine learning impact healthcare?"
```

### Script Options

The main benchmark runner supports several options:

```bash
# Run with only 3 queries (faster)
python -m tog.scripts.run_benchmark --queries 3

# Test only specific prompt strategies
python -m tog.scripts.run_benchmark --strategies default detailed

# Save results to a custom directory
python -m tog.scripts.run_benchmark --output /path/to/results

# Skip plot generation
python -m tog.scripts.run_benchmark --no-plots
```

### Manual Execution

You can also run the benchmark components directly:

```bash
# Run the full benchmark
python -m tog.benchmark.benchmark_entity_extraction

# Generate plots from the most recent results
python -m tog.benchmark.plot_benchmark_results
```

## Benchmark Metrics

The benchmark collects and analyzes the following metrics:

1. **Response Time**:
   - Time taken to extract entities (in seconds)
   - Min/max/mean/median times across queries

2. **Entity Count**:
   - Number of entities extracted per query
   - Min/max/mean/median counts across queries

3. **Strategy Comparison**:
   - Performance differences between prompt strategies
   - Entity count and response time for each strategy

4. **Query Complexity Analysis**:
   - How extraction performance scales with query complexity
   - Correlation between complexity and entity count/response time

## Prompt Strategies

The benchmark tests several prompt strategies:

- **default**: The built-in prompt in the EntityExtractor class
- **concise**: A minimal prompt focused on key topics only
- **detailed**: A comprehensive prompt emphasizing specificity
- **hierarchical**: A prompt that seeks to identify primary and secondary topics

## Extending the Benchmark

To add custom test queries, modify the `TEST_QUERIES` list in `benchmark_entity_extraction.py`.

To test additional prompt strategies, add new entries to the `PROMPT_STRATEGIES` dictionary.

You can also create custom prompt files for the compare_prompts tool:

```json
{
  "custom_prompt_1": "Your custom prompt text here...",
  "custom_prompt_2": "Another custom prompt..."
}
```

Then run:

```bash
python -m tog.scripts.compare_prompts --prompt-file my_prompts.json "Your test query"
```
