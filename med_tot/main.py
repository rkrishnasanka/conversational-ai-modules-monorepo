import logging
import json
import os
from dotenv import load_dotenv
from tree_of_thoughts_executor import TreeOfThoughtsExecutor

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv()

# Example usage
if __name__ == "__main__":
    # Retrieve the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Sample CSV data (replace this with your actual sample data)
    sample_csv_data = """Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description
New York,Living Room,CBD Oil,Tinctures,PKG001,B001,500,10,0,0,0,0,4.5,Pain Relief,Monthly,http://example.com/cbd-oil,High-quality CBD oil for pain relief
Los Angeles,Bedroom,THC Gummies,Edibles,PKG002,B002,0,100,0,0,0,0,4.8,Relaxation,Weekly,http://example.com/thc-gummies,Delicious THC gummies for relaxation
"""

    # Example user query
    user_query = "What are the highest rated CBD products for pain relief?"

    try:
        # Initialize the TreeOfThoughtsExecutor with sample data and API key
        executor = TreeOfThoughtsExecutor(sample_csv_data=sample_csv_data, api_key=api_key)
        # Execute the problem-solving process
        output = executor.execute(user_query=user_query, num_thoughts=3, max_steps=3, best_states_count=2)
        # Print the result in a formatted JSON structure
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")
