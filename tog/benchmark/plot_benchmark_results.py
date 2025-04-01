import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import argparse
import glob

def load_benchmark_data(json_file: str) -> pd.DataFrame:
    """Load benchmark data from JSON file and convert to DataFrame"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data into rows for DataFrame
    rows = []
    for query_id, query_info in data["queries"].items():
        query_text = query_info["text"]
        
        # Calculate query complexity based on word count
        complexity = len(query_text.split())
        
        for strategy, results in query_info["strategies"].items():
            rows.append({
                'query_id': query_id,
                'query_text': query_text,
                'query_complexity': complexity,
                'strategy': strategy,
                'entity_count': results["count"],
                'time_seconds': results["time_seconds"],
                'entities': results["entities"]
            })
    
    return pd.DataFrame(rows)

def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generate various plots to analyze benchmark results"""
    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(font_scale=1.2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Entity count per strategy
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='strategy', y='entity_count', data=df)
    ax.set_title('Entity Count by Prompt Strategy')
    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Number of Entities')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_count_by_strategy.png'))
    
    # 2. Response time per strategy
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='strategy', y='time_seconds', data=df)
    ax.set_title('Response Time by Prompt Strategy')
    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_by_strategy.png'))
    
    # 3. Response time vs. query complexity
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(x='query_complexity', y='time_seconds', hue='strategy', style='strategy', data=df)
    ax.set_title('Response Time vs. Query Complexity')
    ax.set_xlabel('Query Complexity (word count)')
    ax.set_ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_complexity.png'))
    
    # 4. Entity count vs. query complexity
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(x='query_complexity', y='entity_count', hue='strategy', style='strategy', data=df)
    ax.set_title('Entity Count vs. Query Complexity')
    ax.set_xlabel('Query Complexity (word count)')
    ax.set_ylabel('Number of Entities')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_count_vs_complexity.png'))
    
    # 5. Strategy comparison summary
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar plot
    summary = df.groupby('strategy').agg({
        'entity_count': ['mean', 'std'],
        'time_seconds': ['mean', 'std']
    })
    
    summary.columns = ['mean_entities', 'std_entities', 'mean_time', 'std_time']
    summary = summary.reset_index()
    
    # Plot using subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean entity count
    sns.barplot(x='strategy', y='mean_entities', data=summary, ax=ax1)
    ax1.set_title('Average Number of Entities by Strategy')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Mean Entity Count')
    
    # Mean response time
    sns.barplot(x='strategy', y='mean_time', data=summary, ax=ax2)
    ax2.set_title('Average Response Time by Strategy')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Mean Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'))

def main():
    parser = argparse.ArgumentParser(description='Plot entity extraction benchmark results')
    parser.add_argument('--file', help='Path to benchmark results JSON file')
    parser.add_argument('--output', default='./plots', help='Output directory for plots')
    args = parser.parse_args()
    
    # If no file specified, use the latest results file
    if args.file is None:
        results_path = os.path.join(os.path.dirname(__file__), 'results')
        json_files = glob.glob(os.path.join(results_path, 'benchmark_results_*.json'))
        if not json_files:
            print("No benchmark result files found. Run benchmark_entity_extraction.py first.")
            return
        args.file = max(json_files, key=os.path.getctime)  # Get the most recent file
        print(f"Using most recent benchmark file: {args.file}")
    
    # Load data and generate plots
    df = load_benchmark_data(args.file)
    output_dir = os.path.join(args.output, os.path.basename(args.file).replace('.json', ''))
    generate_plots(df, output_dir)
    print(f"Plots generated in {output_dir}")

if __name__ == "__main__":
    main()
