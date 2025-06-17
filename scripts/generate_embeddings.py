import json
import os
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

# Load the CSV file into a pandas DataFrame
csv_file_path = "./test_data/data_descriptions.csv"
df = pd.read_csv(csv_file_path)


# Function to generate embeddings using Azure OpenAI
def generate_embeddings(text):
    response = client.embeddings.create(
        input=[text], 
        model="text-embedding-ada-002",
        deployment_id="text-embedding-ada-002"
    )
    return response.data[0].embedding


# Apply the function to the DataFrame with a progress bar
df["embeddings"] = df["description"].progress_apply(generate_embeddings)

# Convert the embeddings to JSON strings
df["embeddings"] = df["embeddings"].apply(json.dumps)

# Save the DataFrame with embeddings to a new CSV file
output_csv_file_path = "./test_data/data_descriptions_with_embeddings.tsv"
df.to_csv(output_csv_file_path, sep="\t", index=False)
