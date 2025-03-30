from pathlib import Path
import yaml
from typing import Dict
from tog.config import PROMPTS_DIR  # Import from config file

class PromptManager:
    """
    A class to manage prompts stored in YAML files in the prompts directory.
    Each YAML file should contain at least 'system' and 'user' keys.
    """

    PROMPTS_DIR = PROMPTS_DIR  # Use the path from config

    @classmethod
    def set_prompt_dir(cls, prompt_dir: str):
        """
        Set the directory where prompt files are stored.

        Args:
            prompt_dir: Path to the directory containing prompt files
        """
        cls.PROMPTS_DIR = prompt_dir

    @classmethod
    def get_prompt(cls, filename: str, directory: str = None) -> Dict[str, str]:
        """
        Get a prompt from a YAML file.

        Args:
            filename: Name of the YAML file (with or without .yaml extension)

        Returns:
            Dictionary containing prompt data with 'system' and 'user' keys

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            KeyError: If the YAML file doesn't contain required keys
        """

        # Use the default prompt directory if none is provided
        if directory is None:
            prompt_dir = cls.PROMPTS_DIR
        else:
            prompt_dir = directory

        # Ensure the filename has .yaml extension
        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"

        filepath = Path(prompt_dir) / filename

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                prompt_data = yaml.safe_load(file)

            # Validate that required keys are present
            if "system" not in prompt_data or "user" not in prompt_data:
                raise KeyError(f"Prompt file {filename} must contain 'system' and 'user' keys")

            return prompt_data

        except KeyError as e:
            raise KeyError(f"KeyError: {e}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
    @classmethod
    def list_prompts(cls) -> list:
        """
        List all available prompt files in the prompts directory.

        Returns:
            List of prompt filenames (without extension)
        """
        prompt_files = []
        # Use pathlib to list files
        for file_path in Path(cls.PROMPTS_DIR).iterdir():
            if file_path.is_file() and file_path.suffix == ".yaml":
                prompt_files.append(file_path.stem)  # stem is filename without extension
        return prompt_files

    def __repr__(self):
        return f"PromptManager(prompt_dir={self.PROMPTS_DIR})"

    def __str__(self):
        return f"PromptManager(prompt_dir={self.PROMPTS_DIR})"

# Usage
if __name__ == "__main__":
    print(PromptManager.get_prompt("extraction_prompt"))
    print("-" * 50)
    print(PromptManager.list_prompts())
