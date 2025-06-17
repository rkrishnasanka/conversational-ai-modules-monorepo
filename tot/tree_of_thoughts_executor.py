from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tot.tree_of_thoughts import TreeOfThoughts


@dataclass
class ToTExecutorInputs:
    """
    Data class to hold input parameters for TreeOfThoughtsExecutor.
    """

    api_key: str
    azure_endpoint: str
    deployment_name: str
    api_version: str = "2023-05-15"
    json_output_prompt: str = ""
    classification_prompt: str = ""
    evaluation_prompt: str = ""
    sample_csv_data: str = ""
    num_thoughts: Optional[int] = None
    num_iterations: Optional[int] = None
    thought_generation_prompt: Optional[str] = None


class TreeOfThoughtsExecutor:
    """
    Executor class for the Tree of Thoughts process.

    This class initializes and executes the Tree of Thoughts algorithm
    using the provided input parameters.
    """

    def __init__(self, data_inputs: ToTExecutorInputs):
        """
        Initialize the TreeOfThoughtsExecutor.

        Args:
            data_inputs (ToTExecutorInputs): An instance containing all necessary parameters.

        Raises:
            ValueError: If required inputs are missing.
        """
        if not data_inputs.json_output_prompt:
            raise ValueError("JSON output prompt must be provided")
        if not data_inputs.classification_prompt:
            raise ValueError("Classification prompt must be provided")
        if not data_inputs.api_key:
            raise ValueError("API key must be provided")
        if not data_inputs.azure_endpoint:
            raise ValueError("Azure endpoint must be provided")
        if not data_inputs.deployment_name:
            raise ValueError("Deployment name must be provided")

        self.data_inputs: ToTExecutorInputs = data_inputs

        self.num_thoughts = data_inputs.num_thoughts or 3
        self.num_iterations = data_inputs.num_iterations or 3

        # Initialize TreeOfThoughts with Azure OpenAI parameters and prompts
        self.tree_of_thoughts = TreeOfThoughts(
            api_key=data_inputs.api_key,
            azure_endpoint=data_inputs.azure_endpoint,
            deployment_name=data_inputs.deployment_name,
            api_version=data_inputs.api_version,
            sample_data=data_inputs.sample_csv_data,
            classification_prompt=data_inputs.classification_prompt,
            thought_generation_prompt=data_inputs.thought_generation_prompt,
            state_evaluation_prompt=data_inputs.evaluation_prompt,
            json_output_prompt=data_inputs.json_output_prompt,
        )

    def execute(self, user_query: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Execute the Tree of Thoughts process.

        Args:
            user_query (str): The user query string.
            chat_history (List[Tuple[str, str]]): Previous conversation history.

        Returns:
            Dict[str, Any]: The result of the Tree of Thoughts process.

        Raises:
            ValueError: If user_query is empty.
        """
        if not user_query:
            raise ValueError("User query must not be empty")

        return self.tree_of_thoughts.solve(
            user_input=user_query,
            chat_history=chat_history,
            num_thoughts=self.num_thoughts,
            max_steps=self.num_iterations,
            best_states_count=2,  # Adjust this parameter as needed
        )
