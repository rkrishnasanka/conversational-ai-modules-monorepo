from typing import List, Literal, Optional

import openai
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)


class ThoughtGenerator:
    def __init__(self, api_key: str, thought_generation_prompt: Optional[str] = None):
        """ThoughtGenerator class for generating thoughts for problem-solving.

        Args:
            api_key (str):
            thought_generation_prompt (Optional[str], optional): _description_. Defaults to None.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.thought_generation_prompt = thought_generation_prompt or self.default_thought_generation_prompt()

    @staticmethod
    def default_thought_generation_prompt() -> str:
        """Return the default thought generation prompt for the thought generator."""

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
        """Generate thoughts for problem-solving based on the current state.

        Args:
            current_state (str): The current state of the problem.
            num_thoughts (int): The number of thoughts to generate.

        Raises:
            Exception: If no thoughts are generated.

        Returns:
            List[str]: A list of generated thoughts.
        """
        prompt = self.thought_generation_prompt.format(
            current_state=current_state, num_thoughts=num_thoughts)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant generating thoughts for problem-solving."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            # max_tokens=100,
            n=1,
            temperature=0.7,
        )

        thoughts_text = response.choices[0].message.content
        if thoughts_text is not None:
            thoughts_text = thoughts_text.strip()
            thoughts = [thought.split(". ", 1)[1] for thought in thoughts_text.split(
                "\n") if ". " in thought]
            return thoughts[:num_thoughts]
        else:
            raise Exception("No thoughts generated.")
