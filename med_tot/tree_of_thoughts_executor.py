import logging
from typing import Dict, Any
from sample_data_manager import SampleDataManager
from intent_classifier import IntentClassifier
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from tree_of_thoughts import TreeOfThoughts

class TreeOfThoughtsExecutor:
    """
    Executes the Tree of Thoughts framework to solve a problem using provided sample data and user input.
    """
    def __init__(self, sample_csv_data: str, api_key: str):
        """
        Initialize the TreeOfThoughtsExecutor with sample CSV data and OpenAI API key.

        Args:
            sample_csv_data (str): CSV formatted sample data as a string.
            api_key (str): The OpenAI API key for accessing the GPT model.

        Raises:
            ValueError: If sample_csv_data or api_key is empty.
        """
        if not sample_csv_data or not api_key:
            raise ValueError("sample_csv_data and api_key must not be empty")

        self.sample_csv_data = sample_csv_data
        self.api_key = api_key

        # Initialize components for the Tree of Thoughts framework
        self.sample_data_manager = SampleDataManager(csv_data=self.sample_csv_data)
        self.intent_classifier = IntentClassifier(api_key=self.api_key)
        self.thought_generator = ThoughtGenerator(api_key=self.api_key)
        self.state_evaluator = StateEvaluator(api_key=self.api_key)

        # Create an instance of the TreeOfThoughts class with the initialized components
        self.tree_of_thoughts = TreeOfThoughts(
            api_key=self.api_key,
            sample_data_manager=self.sample_data_manager,
            intent_classifier=self.intent_classifier,
            thought_generator=self.thought_generator,
            state_evaluator=self.state_evaluator
        )

        logging.info("TreeOfThoughtsExecutor initialized successfully.")

    def execute(self, user_query: str, num_thoughts: int = 3, max_steps: int = 3, best_states_count: int = 2) -> Dict[str, Any]:
        """
        Execute the problem-solving process using the Tree of Thoughts framework.

        Args:
            user_query (str): The input from the user to be solved.
            num_thoughts (int, optional): Number of thoughts to generate at each step. Default is 3.
            max_steps (int, optional): Maximum number of steps in the process. Default is 3.
            best_states_count (int, optional): Number of best states to consider for evaluation. Default is 2.

        Returns:
            Dict[str, Any]: Final output including all states, best states, and their evaluations.

        Raises:
            ValueError: If user_query is empty.
        """
        if not user_query:
            raise ValueError("user_query must not be empty")

        logging.info(f"Starting execution process for query: {user_query}")

        try:
            # Execute the Tree of Thoughts process
            result = self.tree_of_thoughts.solve(
                user_input=user_query,
                chat_history=[],
                num_thoughts=num_thoughts,
                max_steps=max_steps,
                best_states_count=best_states_count
            )
            logging.info("Completed the problem-solving process.")
            return result
        except Exception as e:
            logging.error(f"Error during execution: {e}")
            raise
