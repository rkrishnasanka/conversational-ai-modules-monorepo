import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace with actual key")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "replace with actual key")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "replace with actual deployment name")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "replace with actual endpoint")

DEFAULT_TABLE_NAME = "default_table"
DEFAULT_DB_NAME = "default_db"
DEFAULT_URI_COULMN_NAME = "URL"
