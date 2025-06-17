from typing import List, Optional

from openai import AzureOpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


class ThoughtGenerator:
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        thought_generation_prompt: Optional[str] = None,
    ):
        """
        ThoughtGenerator class for generating thoughts using Azure OpenAI.

        Args:
            api_key (str): Azure OpenAI API key.
            azure_endpoint (str): Azure resource endpoint URL.
            api_version (str): API version (e.g., "2023-12-01-preview").
            deployment_name (str): Deployed model name (e.g., "gpt-4o").
            thought_generation_prompt (Optional[str]): Optional custom prompt.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.thought_generation_prompt = (
            thought_generation_prompt or self.default_thought_generation_prompt()
        )

    @staticmethod
    def default_thought_generation_prompt() -> str:
        return """
        Given the current state of the problem:

        {current_state}

        Generate {num_thoughts} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem.

        Your response should be in the following format:
        1. [First thought]
        2. [Second thought]
        ...
        {num_thoughts}. [Last thought]
        """

    def generate_thoughts(self, current_state: str, num_thoughts: int) -> List[str]:
        """
        Generate thoughts based on the current problem-solving state.

        Args:
            current_state (str): The current state of the problem.
            num_thoughts (int): Number of thoughts to generate.

        Returns:
            List[str]: A list of generated thoughts.
        """
        prompt = self.thought_generation_prompt.format(
            current_state=current_state, num_thoughts=num_thoughts
        )

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful assistant generating thoughts for problem-solving."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.7,
        )

        thoughts_text = response.choices[0].message.content
        if not thoughts_text:
            raise Exception("No thoughts generated.")

        thoughts = [
            thought.split(". ", 1)[1]
            for thought in thoughts_text.strip().split("\n")
            if ". " in thought
        ]
        return thoughts[:num_thoughts]
