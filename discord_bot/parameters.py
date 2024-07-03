import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace with actual key")
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', "replace with actual key")

# SQLite database file
SQLITE_DB_FILE = "./aegion.db"

SQL_TABLE_NAME = "new_dataset"

PRODUCT_DESCRIPTIONS_CSV = "./product_descriptions.csv"

LOGGER_FILE = "chatbot.log"

# OUTPUT_COLUMNS = ["Description", "URL", "CustomerRating", "Product"]
OUTPUT_COLUMNS = ['CustomerRating']
# OUTPUT_COLUMNS = []