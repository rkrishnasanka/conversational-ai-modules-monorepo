import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace with actual key")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "replace with actual key")

# SQLite database file
SQLITE_DB_FILE = "aegion.db"

SQL_TABLE_NAME = "new_dataset"

LOGGER_FILE = "chatbot.log"

# OUTPUT_COLUMNS = ["Description", "URL", "CustomerRating", "Product"]
# OUTPUT_COLUMNS = ["CustomerRating"]
OUTPUT_COLUMNS = []

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "aegion")
