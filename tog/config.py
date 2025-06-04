from pathlib import Path


PROMPTS_DIR = Path("tog/prompts")
KG_DIR = Path("tog/data")

# Default model configuration
DEFAULT_MODEL = "gpt-4o"

# Exploration parameters
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_PATHS = 5
DEFAULT_MAX_ENTITIES_PER_ROUND = 3
DEFAULT_MAX_RELATIONS = 3