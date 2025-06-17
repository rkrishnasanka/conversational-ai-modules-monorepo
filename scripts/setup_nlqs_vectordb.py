from pathlib import Path

import pandas as pd

from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver

chroma_config = ChromaDBConfig(
    # host=os.getenv("VECTORDB_HOST", "localhost"),
    # port=int(os.getenv("VECTORDB_PORT", 8000)),
    is_local=True,
    # username=os.getenv("VECTORDB_USER", "admin"),
    # password=os.getenv("VECTORDB_PASSWORD", "potter"),
)


print("Purge the NLQS VectorDB Collections")

# Purge the NLQS VectorDB Collections
VectorDBDriver.purge_nlqs_vectordb(chroma_config=chroma_config)

print("Setup the NLQS VectorDB Collections")

# Setup the NLQS VectorDB Collections
VectorDBDriver.initialize_nlqs_vectordb(chroma_config=chroma_config)


# Figure out how to populate the NLQS VectorDB Collections
# TODO: Populate the NLQS VectorDB column info collection
# Load the column info csv file
column_info_df = pd.read_csv("./test_data/column_metadata_with_embeddings.tsv", sep="\t")

print("Convert the embedding column from json string to List[float]")

# Convert the embedding column from json string to List[float]
column_info_df["embedding"] = column_info_df["embedding"].apply(lambda x: eval(x))

print("Populate the NLQS VectorDB column info collection")

VectorDBDriver.populate_nlqs_column_info(chroma_config, column_info_df)

# TODO: Populate the NLQS VectorDB dataset collection
dataset_info_df = pd.read_csv("./test_data/data_descriptions_with_embeddings.tsv", sep="\t")

print("Convert the embedding column from json string to List[float]")

# Convert the embedding column from json string to List[float]
dataset_info_df["embedding"] = dataset_info_df["embedding"].apply(lambda x: eval(x))

print("Populate the NLQS VectorDB dataset collection")

VectorDBDriver.populate_nlqs_dataset_info(chroma_config, dataset_info_df=dataset_info_df)
