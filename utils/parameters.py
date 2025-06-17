import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "insert_your_openai_key_here")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "insert_your_azure_openai_key_here")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "insert_your_azure_openai_endpoint_here")
AZURE_OPENAI_EMBEDDING_ENDPOINT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_ENDPOINT", "insert_your_azure_openai_embedding_endpoint_here"
)
AZURE_OPENAI_DEPLOYMENT_NAME= os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "insert_your_azure_openai_deployment_name_here")