#!/bin/bash

# Run the entity extraction benchmark and generate plots
echo "Running entity extraction benchmark..."
python -m tog.benchmark.benchmark_entity_extraction

echo "Generating benchmark plots..."
python -m tog.benchmark.plot_benchmark_results

echo "Benchmark complete. Results and plots available in tog/benchmark/results/ and tog/benchmark/plots/"
