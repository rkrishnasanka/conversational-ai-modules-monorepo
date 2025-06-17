import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace with actual key")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "replace with actual key")

# SQLite database file
SQLITE_DB_FILE = "./aegion.db"

SQL_TABLE_NAME = "new_dataset"
URL_COLUMN = "URL"

LOGGER_FILE = "chatbot.log"

# OUTPUT_COLUMNS = ["Description", "URL", "CustomerRating", "Product"]
# OUTPUT_COLUMNS = ["CustomerRating"]
OUTPUT_COLUMNS = []

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "aegion")

SUPABASE_HOST = os.getenv("SUPABASE_HOST", "replace with actual key")
SUPABASE_PORT = os.getenv("SUPABASE_PORT", "replace with actual key")
SUPABASE_USER = os.getenv("SUPABASE_USER", "replace with actual key")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD", "replace with actual key")
SUPABASE_DATABASE_NAME = os.getenv("SUPABASE_DATABASE_NAME", "replace with actual key")
