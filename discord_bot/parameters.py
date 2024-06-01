import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', "fail")
# SQLite database file
SQLITE_DB_FILE = "aegion.db"

PRODUCT_DESCRIPTIONS_CSV = "product_descriptions.csv"