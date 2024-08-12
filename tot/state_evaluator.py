import openai
from typing import List
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam

class StateEvaluator:
    """
    Evaluates problem-solving states using OpenAI's GPT model.
    """

    def __init__(self, api_key: str, evaluation_prompt: str, number_of_iterations: int = 3):
        """
        Initialize the StateEvaluator.

        Args:
            api_key (str): OpenAI API key.
            evaluation_prompt (str, optional): Custom prompt for state evaluation.
        """
        self.api_key = api_key
        self.number_of_iterations = number_of_iterations
        
        # Default evaluation prompt if not provided
        self.evaluation_prompt = evaluation_prompt or """
        Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful.

        {states_text}

        Provide only the numerical ratings, one per line:
        """

    def evaluate_states(self, states: List[str]) -> List[float]:
        """
        Evaluate a list of states using the OpenAI API.

        Args:
            states (List[str]): List of states to evaluate.

        Returns:
            List[float]: List of ratings for each state.
        """
        # Format states for the prompt
        states_text = "\n".join(f"{i+1}. {state}" for i, state in enumerate(states))
        
        prompt = self.evaluation_prompt.format(states_text=states_text)
        
        messages = [
            ChatCompletionSystemMessageParam(role= "system", content= "You are a helpful assistant evaluating problem-solving states."),
            ChatCompletionUserMessageParam(role= "user", content= prompt)
        ]
        
        # Call OpenAI API for state evaluation
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            temperature=0.3
        )

        # Extract and process ratings from the response
        ratings_text = response.choices[0].message.content
        if ratings_text is None:
            return [0.0] * len(states)
        else:
            ratings_text = ratings_text.strip()
            ratings = [float(rating) for rating in ratings_text.split('\n') if rating.replace('.', '').isdigit()]
        
        # Ensure we have a rating for each state
        if len(ratings) < len(states):
            ratings.extend([0.0] * (len(states) - len(ratings)))
        return ratings[:len(states)]
