import openai
from typing import List
import logging

logger = logging.getLogger(__name__)

def generate_thoughts(state: str, k: int) -> List[str]:
    """
    Generate thoughts or considerations for the given state.

    Args:
        state (str): The current state of the problem.
        k (int): The number of thoughts to generate.

    Returns:
        List[str]: A list of generated thoughts.
    """
    prompt = f"Given the current state of the problem:\n\n{state}\n\nGenerate {k} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant generating thoughts for problem-solving."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"Your response should be in the following format:\n1. [First thought]\n2. [Second thought]\n...\n{k}. [Last thought]"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            temperature=0.7
        )

        thoughts_text = response.choices[0].message['content'].strip()
        thoughts = [thought.split('. ', 1)[1] for thought in thoughts_text.split('\n') if '. ' in thought]
        return thoughts[:k]
    except Exception as e:
        logger.error(f"Error in thought generation: {e}")
        return [f"Error in thought generation: {e}"] * k
