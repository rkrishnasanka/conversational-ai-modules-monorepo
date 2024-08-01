import logging
import json
from typing import Optional, Dict
from framework import Framework

class TreeOfThoughtsExecutor:
    """
    Executor class for solving problems using the Tree of Thoughts framework.
    
    Attributes:
        api_key (str): The API key for accessing the OpenAI service.
        sample_csv_data (Optional[str]): Sample CSV data for testing purposes.
        tot_framework (Framework): Instance of the Framework class for problem-solving.
    """
    
    def __init__(self, api_key: str, sample_csv_data: Optional[str] = None):
        """
        Initialize the executor with the provided API key and optional sample CSV data.

        Args:
            api_key (str): The API key for accessing the OpenAI service.
            sample_csv_data (Optional[str]): Sample CSV data for testing purposes.
        """
        if not api_key:
            raise ValueError("api_key must not be empty")

        self.sample_csv_data = sample_csv_data
        self.api_key = api_key
        self.tot_framework = Framework(api_key=self.api_key)
        logging.info("TreeOfThoughtsExecutor initialized successfully.")

    def execute(self, user_query: str, num_thoughts: int = 3, max_steps: int = 3, best_states_count: int = 2) -> str:
        """
        Execute the problem-solving process for the given user query.

        Args:
            user_query (str): The user's problem query to be solved.
            num_thoughts (int): The number of thoughts to generate at each step. Default is 3.
            max_steps (int): The maximum number of steps to perform in the BFS. Default is 3.
            best_states_count (int): The number of best states to keep at each step. Default is 2.

        Returns:
            str: The final output as a JSON string.
        """
        if not user_query:
            raise ValueError("user_query must not be empty")

        logging.info(f"Starting execution process for query: {user_query}")
        try:
            result = self.tot_framework.solve(problem=user_query, k=num_thoughts, T=max_steps, b=best_states_count)
            best_state, best_path = result
            parsed_solution = self.tot_framework.parse_solution(best_state, best_path)
            output = self.tot_framework.generate_output(parsed_solution)
            logging.info("Completed the problem-solving process.")
            return output
        except Exception as e:
            logging.error(f"Error during execution: {e}")
            return json.dumps({"error": str(e)})
