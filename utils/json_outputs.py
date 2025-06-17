from typing import Any, Dict, List


def validate_llm_output_keys(llm_output: Dict[str, Any], reference_dict: Dict[str, Any]) -> List[str]:
    """Validate that all keys in the reference_dict are present in the llm_output.

    Args:
        llm_output (Dict[str, Any]): A dictionary containing the output of the LLM (the json must be loaded as a dictionary before sending here)
        reference_dict (Dict[str, Any]): A dictionary containing the reference keys

    Returns:
        List[str]: _description_
    """

    def find_missing_keys(d1: Dict[str, Any], d2: Dict[str, Any], parent_key: str = "") -> List[str]:
        missing = []
        for key in d1:
            full_key = f"{parent_key}.{key}" if parent_key else key
            if key not in d2:
                missing.append(full_key)
            elif isinstance(d1[key], dict) and isinstance(d2.get(key), dict):
                missing.extend(find_missing_keys(d1[key], d2[key], full_key))
        return missing

    return find_missing_keys(reference_dict, llm_output)
