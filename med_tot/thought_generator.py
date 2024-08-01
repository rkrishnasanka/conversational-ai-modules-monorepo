import openai
from typing import List

class ThoughtGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_thoughts(self, current_state: str, num_thoughts: int) -> List[str]:
        prompt = f"Given the current state of the problem:\n\n{current_state}\n\nGenerate {num_thoughts} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant generating thoughts for problem-solving."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Your response should be in the following format:\n1. [First thought]\n2. [Second thought]\n...\n{num_thoughts}. [Last thought]"}
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
            return thoughts[:num_thoughts]  # Ensure we return exactly num_thoughts thoughts
        except Exception as e:
            print(f"Error in thought generation: {e}")
            return [f"Error in thought generation: {e}"] * num_thoughts
