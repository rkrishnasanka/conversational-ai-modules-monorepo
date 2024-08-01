import openai
from typing import List

class StateEvaluator:
    """
    Evaluates problem-solving states for relevance and usefulness using OpenAI's GPT-3.5-turbo model.
    """
    def __init__(self, api_key: str):
        """
        Initialize the StateEvaluator with the OpenAI API key.

        Args:
            api_key (str): The OpenAI API key for accessing the GPT model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def evaluate_states(self, states: List[str]) -> List[float]:
        """
        Evaluate a list of states and provide relevance ratings.

        Args:
            states (List[str]): List of states to be evaluated.

        Returns:
            List[float]: List of relevance ratings for each state.
        """
        prompt = f"Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful."
        
        states_text = "\n".join(f"{i+1}. {state}" for i, state in enumerate(states))
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant evaluating problem-solving states."},
            {"role": "user", "content": f"{prompt}\n\n{states_text}"},
            {"role": "user", "content": "Provide only the numerical ratings, one per line:"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                n=1,
                temperature=0.3
            )

            ratings_text = response.choices[0].message['content'].strip()
            ratings = [float(rating) for rating in ratings_text.split('\n') if rating.replace('.', '').isdigit()]
            
            # Ensure we have a rating for each state
            if len(ratings) < len(states):
                ratings.extend([0.0] * (len(states) - len(ratings)))
            return ratings[:len(states)]
        except Exception as e:
            print(f"Error in state evaluation: {e}")
            return [0.0] * len(states)
