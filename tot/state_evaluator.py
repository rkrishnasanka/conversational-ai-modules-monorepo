from typing import List, Optional

from openai import AzureOpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


class StateEvaluator:
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        evaluation_prompt: Optional[str] = None,
    ):
        """
        StateEvaluator class using Azure OpenAI to evaluate problem-solving states.

        Args:
            api_key (str): Azure OpenAI API key.
            azure_endpoint (str): Your Azure resource endpoint.
            api_version (str): API version (e.g., "2023-12-01-preview").
            deployment_name (str): Name of the deployed model.
            evaluation_prompt (Optional[str], optional): Custom prompt. Defaults to None.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.evaluation_prompt = evaluation_prompt or self.default_evaluation_prompt()

    @staticmethod
    def default_evaluation_prompt() -> str:
        return """
        Evaluate the following states in terms of their relevance and usefulness for addressing the problem. 
        Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful.

        {states_text}

        Provide only the numerical ratings, one per line:
        """

    def evaluate_states(self, states: List[str]) -> List[float]:
        """
        Evaluate the problem-solving states based on their relevance and usefulness.

        Args:
            states (List[str]): List of states to evaluate.

        Returns:
            List[float]: Corresponding numerical ratings.
        """
        states_text = "\n".join(f"{i+1}. {state}" for i, state in enumerate(states))
        prompt = self.evaluation_prompt.format(states_text=states_text)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful assistant evaluating problem-solving states."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=150,
            temperature=0.3,
        )

        ratings_text = response.choices[0].message.content
        if not ratings_text:
            return [0.0] * len(states)

        ratings = [
            float(r.strip())
            for r in ratings_text.strip().split("\n")
            if r.strip().replace(".", "", 1).isdigit()
        ]

        # Ensure the output length matches the input states
        if len(ratings) < len(states):
            ratings.extend([0.0] * (len(states) - len(ratings)))
        return ratings[:len(states)]
