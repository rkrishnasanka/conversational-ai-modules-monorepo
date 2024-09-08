from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from tot.tree_of_thoughts import TreeOfThoughts


@dataclass
class ToTExecutorInputs:
    """
    Data class to hold input parameters for TreeOfThoughtsExecutor.
    """

    api_key: str
    json_output_prompt: str
    classification_prompt: str
    evaluation_prompt: str
    sample_csv_data: str
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
            data_inputs (DataInputs): An instance of DataInputs containing all necessary parameters.

        Raises:
            ValueError: If required inputs are missing.
        """
        if not data_inputs.json_output_prompt:
            raise ValueError("JSON output prompt must be provided")
        if not data_inputs.classification_prompt:
            raise ValueError("Classification prompt must be provided")

        self.data_inputs: ToTExecutorInputs = data_inputs

        self.num_thoughts = data_inputs.num_thoughts or 3
        self.num_iterations = data_inputs.num_iterations or 3

        self.tree_of_thoughts = TreeOfThoughts(
            api_key=data_inputs.api_key,
            sample_data=data_inputs.sample_csv_data,
            classification_prompt=data_inputs.classification_prompt,
            thought_generation_prompt=data_inputs.thought_generation_prompt,
            state_evaluation_prompt=data_inputs.evaluation_prompt,
            json_output_prompt=self.data_inputs.json_output_prompt,
        )

    def execute(self, user_query: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Execute the Tree of Thoughts process.

        Returns:
            Dict[str, Any]: The result of the Tree of Thoughts process.

        Raises:
            ValueError: If user_query is empty.
        """

        return self.tree_of_thoughts.solve(
            user_input=user_query,
            chat_history=chat_history,
            num_thoughts=self.num_thoughts,
            max_steps=self.num_iterations,
            best_states_count=2,  # Adjust as needed
        )
