from typing import List, Literal, Optional

import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)


class StateEvaluator:
    def __init__(self, api_key: str, evaluation_prompt: Optional[str] = None):
        """StateEvaluator class for evaluating problem-solving states.

        Args:
            api_key (str): _description_
            evaluation_prompt (Optional[str], optional): _description_. Defaults to None.
        """
        self.api_key = api_key
        self.evaluation_prompt = evaluation_prompt or self.default_evaluation_prompt()

    @staticmethod
    def default_evaluation_prompt() -> str:
        """Return the default evaluation prompt for the state evaluator."""

        return """
        Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful.

        {states_text}

        Provide only the numerical ratings, one per line:
        """

    def evaluate_states(self, states: List[str]) -> List[float]:
        """Evaluate the problem-solving states based on their relevance and usefulness.

        Args:
            states (List[str]): A list of problem-solving states to evaluate.

        Returns:
            List[float]: A list of numerical ratings for each state.
        """
        states_text = "\n".join(
            f"{i+1}. {state}" for i, state in enumerate(states))

        prompt = self.evaluation_prompt.format(states_text=states_text)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant evaluating problem-solving states."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        response = openai.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=100, n=1, temperature=0.3
        )

        ratings_text = response.choices[0].message.content
        if ratings_text is None:
            return [0.0] * len(states)
        else:
            ratings_text = ratings_text.strip()
            ratings = [float(rating) for rating in ratings_text.split(
                "\n") if rating.replace(".", "").isdigit()]

        if len(ratings) < len(states):
            ratings.extend([0.0] * (len(states) - len(ratings)))
        return ratings[: len(states)]
