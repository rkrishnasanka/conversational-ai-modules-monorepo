import logging
import os
from dotenv import load_dotenv
from tree_of_thoughts_executor import TreeOfThoughtsExecutor

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function to execute the TreeOfThoughtsExecutor with a user query.
    """
    # Load environment variables from a .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual API key

    if not api_key:
        logging.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    user_query = "explain me the complete lifecycle of mlops"

    try:
        # Initialize the TreeOfThoughtsExecutor with the provided API key
        executor = TreeOfThoughtsExecutor(api_key=api_key)
        
        # Execute the problem-solving process
        output = executor.execute(user_query=user_query, num_thoughts=3, max_steps=3, best_states_count=2)
        
        # Print the formatted output string
        print(output)
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
