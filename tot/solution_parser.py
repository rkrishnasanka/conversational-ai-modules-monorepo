import json
from typing import Dict, List

def parse_solution(best_state: str, path: List[str]) -> Dict:
    """
    Parse the solution from the best state and path of thoughts.

    Args:
        best_state (str): The best state found.
        path (List[str]): The path of thoughts leading to the best state.

    Returns:
        Dict: The parsed solution as a dictionary.
    """
    steps = [{"step": i + 1, "thought": thought} for i, thought in enumerate(path)]
    
    return {
        "problem": best_state.split(" -> ")[0] if " -> " in best_state else best_state,
        "final_solution": best_state.split(" -> ")[-1] if " -> " in best_state else best_state,
        "thought_process": steps
    }

def generate_output(parsed_solution: Dict) -> str:
    """
    Generate the final output from the parsed solution.

    Args:
        parsed_solution (Dict): The parsed solution as a dictionary.

    Returns:
        str: The final output as a JSON string.
    """
    return json.dumps(parsed_solution, indent=2)
