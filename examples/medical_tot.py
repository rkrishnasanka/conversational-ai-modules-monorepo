import json
import logging
import os

from dotenv import load_dotenv

from tot.state_evaluator import StateEvaluator
from tot.thought_generator import ThoughtGenerator
from tot.tree_of_thoughts_executor import ToTExecutorInputs, TreeOfThoughtsExecutor

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv()


def get_json_output_prompt() -> str:
    """
    Provide the JSON output prompt for the Tree of Thoughts process.

    Returns:
        str: JSON output prompt.
    """
    return """
    User Query: {user_query}
    Final State: {final_state}
    Thought Path: {thought_path}
    Sample Data: {sample_data}

    Generate a JSON object with the following structure:
    {{
        "summary": "A detailed summary of the user input and analysis, focusing on qualitative aspects",
        "quantitative_data": {{
            "column name": "relevant numerical data from the analysis",
            ...
        }},
        "qualitative_data": {{
            "column name": "detailed qualitative information related to the query",
            ...
        }},
        "user_requested_columns": ["List of columns the user explicitly or implicitly requested"],
        "intent": "information_request",
    }}

    Guidelines for generating the response:
    1. Focus primarily on extracting and presenting qualitative data that's most relevant to the user's query.
    2. Include detailed descriptions, categories, or other text-based information in the qualitative_data section.
    3. For the quantitative_data, only include numerical data that's directly relevant to the query.
    4. In the summary, provide a comprehensive analysis that ties together the qualitative and quantitative aspects.
    5. Ensure all data included is directly related to the user's query.
    6. If the query mentions specific criteria (e.g., a particular rating), make sure to filter the data accordingly.
    7. For user_requested_columns, only include columns that are explicitly or implicitly requested in the user's query. If no columns are requested, return an empty list [].

    The columns present in our database are: "Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description"
    """


def get_classification_prompt() -> str:
    """
    Provide the classification prompt for intent classification.

    Returns:
        str: Classification prompt.
    """
    return """
    Classify the user's intent based on the following input:

    User Input: {user_input}

    Possible intents:
    1. Phatic communication (greetings, farewells, etc.)
    2. Profanity or vulgar input
    3. SQL injection attempt
    4. Information request
    5. Other (not related to available data)

    Respond with only the number corresponding to the intent.
    """


def main():
    # Retrieve the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        return

    # Sample CSV data (replace this with your actual sample data)
    sample_csv_data = """Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description
New York,Living Room,CBD Oil,Tinctures,PKG001,B001,500,10,0,0,0,0,4.5,Pain Relief,Monthly,http://example.com/cbd-oil,High-quality CBD oil for pain relief
Los Angeles,Bedroom,THC Gummies,Edibles,PKG002,B002,0,100,0,0,0,0,4.8,Relaxation,Weekly,http://example.com/thc-gummies,Delicious THC gummies for relaxation
"""

    # Example user query
    user_query = "What are the highest rated CBD products for pain relief?"

    try:
        # Initialize the TreeOfThoughtsExecutor with sample data, API key, and prompts
        executor = TreeOfThoughtsExecutor(
            ToTExecutorInputs(
                api_key=api_key,
                json_output_prompt=get_json_output_prompt(),
                classification_prompt=get_classification_prompt(),
                thought_generation_prompt=ThoughtGenerator(
                    api_key=api_key
                ).thought_generation_prompt,  # Uncomment if available
                evaluation_prompt=StateEvaluator(api_key=api_key).evaluation_prompt,  # Uncomment if available
                sample_csv_data=sample_csv_data,
                # num_thoughts=3,
                # num_iterations=3
            )
        )

        # Execute the problem-solving process
        output = executor.execute(user_query=user_query, chat_history=[])
        print(output)
        # Print the result in a formatted JSON structure
        print(json.dumps(output, indent=2))
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
