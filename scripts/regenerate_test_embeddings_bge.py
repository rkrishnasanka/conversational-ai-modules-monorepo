"""
Regenerate test embeddings using BGE (384-dim) instead of Azure OpenAI (1536-dim).
This script reads the existing TSV files and regenerates embeddings for all descriptions.
"""

import pandas as pd
from tqdm import tqdm
from utils.llm import get_default_embedding_function

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize BGE embedding model (384 dimensions)
print("Initializing BGE embedding model...")
embedding_model = get_default_embedding_function(use_local=True)
embedding_function = embedding_model.embed_query


def regenerate_embeddings(input_file: str, output_file: str, description_column: str):
    """Regenerate embeddings for a TSV file."""
    print(f"\nProcessing {input_file}...")

    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t")
    print(f"Loaded {len(df)} records")

    # Generate new embeddings with progress bar
    print(f"Generating BGE embeddings for '{description_column}' column...")
    df["embedding"] = df[description_column].progress_apply(embedding_function)

    # Convert embeddings to JSON strings
    df["embedding"] = df["embedding"].apply(lambda x: str(x))

    # Save the updated DataFrame
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved to {output_file}")

    # Verify embedding dimensions
    sample_embedding = eval(df["embedding"].iloc[0])
    print(f"Embedding dimension: {len(sample_embedding)} (should be 384)")


# Regenerate all three TSV files
files_to_process = [
    (
        "nlqs/tests/data/column_descriptions_with_embeddings.tsv",
        "nlqs/tests/data/column_descriptions_with_embeddings.tsv",
        "description",
    ),
    (
        "nlqs/tests/data/data_descriptions_with_embeddings.tsv",
        "nlqs/tests/data/data_descriptions_with_embeddings.tsv",
        "description",
    ),
    (
        "nlqs/tests/data/table_descriptions_with_embeddings.tsv",
        "nlqs/tests/data/table_descriptions_with_embeddings.tsv",
        "description",
    ),
]

print("=" * 80)
print("Regenerating test embeddings with BGE (384 dimensions)")
print("=" * 80)

for input_file, output_file, desc_col in files_to_process:
    regenerate_embeddings(input_file, output_file, desc_col)

print("\n" + "=" * 80)
print("✓ All embeddings regenerated successfully!")
print("=" * 80)
