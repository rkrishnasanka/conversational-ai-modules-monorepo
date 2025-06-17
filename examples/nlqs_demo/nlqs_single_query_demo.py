import os
from pathlib import Path
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()

# Validate required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info("All required environment variables found")

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.vectordb_driver import ChromaDBConfig
from nlqs.nlqs import NLQS, NLQSResult

# Setup database configuration
logger.info("Setting up database configuration...")
db_path = Path("aegion.db")
logger.debug(f"Database path: {db_path.absolute()}")
sqlite_config = SQLiteConnectionConfig(
    db_file=db_path,
    dataset_table_name="new_dataset",
    uri_column="URL",
    output_columns=["Product", "Category", "CBD", "THC", "Description", "MedicalBenefitsReported"]
)
DEFAULT_COLUMN_INFO_COLLECTION_NAME = "nlqs_column_info"
DEFAULT_DATASET_COLLECTION_NAME = "nlqs_descriptive_data"
DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME = "nlqs_table_descriptions"

# Setup ChromaDB configuration
logger.info("Setting up ChromaDB configuration...")
chroma_config = ChromaDBConfig(
    persist_path=Path("chroma"),
    table_description_collection_name = DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME,
    column_info_collection_name = DEFAULT_COLUMN_INFO_COLLECTION_NAME,
    dataset_collection_name = DEFAULT_DATASET_COLLECTION_NAME,
)

# Environment variables should already be loaded from .env file
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
logger.debug(f"Using Azure OpenAI API version: {os.environ['AZURE_OPENAI_API_VERSION']}")

def print_result(result: Dict[str, Any]) -> None:
    """Print a single result record in a formatted way."""
    print("\nProduct Details:")
    print("-" * 50)
    for key, value in result.items():
        if value:  # Only print non-empty values
            print(f"{key.replace('_', ' ').title()}: {value}")

class NLQSDemo:
    def __init__(self):
        """Initialize the NLQS demo with configuration."""
        logger.info("Initializing NLQS demo...")
        try:
            self.nlqs = NLQS(connection_config=sqlite_config, chroma_config=chroma_config)
            logger.info("NLQS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLQS: {str(e)}")
            raise
        self.chat_history = []

    def process_query(self, query: str) -> None:
        """Process a natural language query and print results."""
        logger.info(f"Processing query: {query}")
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        try:
            logger.debug("Executing NLQS query workflow...")
            result = self.nlqs.execute_nlqs_query_workflow(query, self.chat_history)
            
            if result.is_input_irrelevant:
                logger.info("Query flagged as irrelevant")
                print("\n⚠️ Query was flagged as irrelevant (possible SQL injection or general conversation)")
                return
            
            if not result.records:
                logger.info("No matching records found")
                print("\n❌ No matching records found")
                return
            
            logger.info(f"Found {len(result.records)} matching records")
            print(f"\n✅ Found {len(result.records)} matching records:")
            for record in result.records:
                print_result(record)
            
            if result.uris:
                logger.debug(f"Found {len(result.uris)} associated URLs")
                print("\nAssociated URLs:")
                print("-" * 50)
                for uri in result.uris:
                    print(f"🔗 {uri}")
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"\n❌ Error processing query: {str(e)}")
            
def main():
    """Run the NLQS demo with a single test query."""
    logger.info("Starting NLQS demo...")
    
    # Initialize NLQS demo
    try:
        demo = NLQSDemo()
    except Exception as e:
        logger.error(f"Failed to initialize NLQS demo: {str(e)}", exc_info=True)
        print(f"Failed to initialize NLQS: {str(e)}")
        return

    # Test with a single query first
    test_query =  "Show me products with high CBD content"
    
    logger.info("Starting test with a single query")
    print("Starting NLQS Demo...")
    print("=" * 50)
    
    demo.process_query(test_query)

if __name__ == "__main__":
    main()