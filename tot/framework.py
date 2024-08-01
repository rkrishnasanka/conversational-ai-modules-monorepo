import openai
import logging
from typing import List, Tuple, Dict
from embedding_utils import get_openai_embedding, cosine_similarity
from thought_generator import generate_thoughts
from state_evaluator import evaluate_states
from solution_parser import parse_solution, generate_output

class Framework:
    """
    A framework class for solving problems using a breadth-first search (BFS) approach with thought generation and evaluation.
    
    Attributes:
        api_key (str): The API key for accessing the OpenAI service.
        logger (logging.Logger): Logger instance for logging information and debugging.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the framework with the provided API key.
        
        Args:
            api_key (str): The API key for accessing the OpenAI service.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.logger = logging.getLogger(__name__)

    def solve(self, problem: str, k: int = 3, T: int = 3, b: int = 2) -> Tuple[str, List[str]]:
        """
        Solve the given problem using the BFS approach with thought generation and evaluation.
        
        Args:
            problem (str): The problem statement to be solved.
            k (int): The number of thoughts to generate at each step. Default is 3.
            T (int): The number of steps to perform in the BFS. Default is 3.
            b (int): The number of best states to keep at each step. Default is 2.
        
        Returns:
            Tuple[str, List[str]]: The best state found and the path of thoughts leading to it.
        """
        self.logger.info(f"Starting to solve problem: {problem}")
        self.logger.info(f"Parameters: k={k}, T={T}, b={b}")
        return self._bfs(problem, k, T, b)

    def _bfs(self, problem: str, k: int, T: int, b: int) -> Tuple[str, List[str]]:
        """
        Perform a breadth-first search to solve the problem.
        
        Args:
            problem (str): The problem statement to be solved.
            k (int): The number of thoughts to generate at each step.
            T (int): The number of steps to perform in the BFS.
            b (int): The number of best states to keep at each step.
        
        Returns:
            Tuple[str, List[str]]: The best state found and the path of thoughts leading to it.
        """
        initial_state = problem
        states = [(initial_state, [])]
        self.logger.info(f"Starting BFS with initial state: {initial_state}")

        for step in range(T):
            self.logger.info(f"Step {step + 1}/{T}")
            new_states = []
            for state, path in states:
                self.logger.debug(f"Generating thoughts for state: {state}")
                new_thoughts = generate_thoughts(state, k)
                self.logger.debug(f"Generated thoughts: {new_thoughts}")
                new_states.extend([(f"{state} -> {thought}", path + [thought]) for thought in new_thoughts])

            if not new_states:
                self.logger.warning("No new states generated. Stopping early.")
                break

            self.logger.info(f"Evaluating {len(new_states)} new states")
            values = evaluate_states([state for state, _ in new_states], problem)
            
            if len(values) != len(new_states):
                self.logger.warning(f"Mismatch between number of states ({len(new_states)}) and evaluations ({len(values)})")
                values = values[:len(new_states)]
            
            states = sorted(zip(new_states, values), key=lambda x: x[1], reverse=True)[:b]
            states = [state for state, _ in states]
            self.logger.info(f"Selected top {b} states: {[state[0] for state in states]}")

        if not states:
            self.logger.warning("No states remaining after search.")
            return (problem, ["No additional thoughts generated"])
        
        best_state, best_path = states[0]
        self.logger.info(f"BFS completed. Best state: {best_state}")
        self.logger.debug(f"Path to best state: {best_path}")
        return best_state, best_path

    def parse_solution(self, best_state: str, path: List[str]) -> Dict:
        """
        Parse the solution from the best state and path of thoughts.
        
        Args:
            best_state (str): The best state found.
            path (List[str]): The path of thoughts leading to the best state.
        
        Returns:
            Dict: The parsed solution as a dictionary.
        """
        return parse_solution(best_state, path)

    def generate_output(self, parsed_solution: Dict) -> str:
        """
        Generate the final output from the parsed solution.
        
        Args:
            parsed_solution (Dict): The parsed solution as a dictionary.
        
        Returns:
            str: The final output as a string.
        """
        return generate_output(parsed_solution)
