from typing import Optional

import openai
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)



class IntentClassifier:
    def __init__(self, api_key: str, classification_prompt: Optional[str] = None):
        """
        Classify the intent of the user input using OpenAI's GPT model.

        Args:
            user_input (str): The user's input to classify.

        Returns:
            str: The classified intent as a string (1-5).
        """
        openai.api_key = api_key
        self.classification_prompt = classification_prompt or self.default_classification_prompt()

    @staticmethod
    def default_classification_prompt():
        return """
        Classify the user's intent based on the following input:

        User Input: {user_input}

        Possible intents:
        1. Phatic communication (greetings, farewells, etc.)
        2. Profanity or vulgar input
        3. SQL injection attempt
        4. Information request
        5. Other (not related to available data)

        Respond with only the number corresponding to the intent.
        """

    def classify_intent(self, user_input: str) -> str:
        """Classify the intent of the user input.

        Args:
            user_input (str): The user's input to classify.

        Returns:
            str: The classified intent as a string (1-5).
        """
        prompt = self.classification_prompt.format(user_input=user_input)
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant classifying user intent."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        response = openai.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=100, n=1, temperature=0.3
        )

        choice = response.choices[0]
        message_content = choice.message.content
        if message_content is not None:
            intent = message_content.strip()
            return intent
        else:
            return "4"  # Default to "Information request" if no content is received
