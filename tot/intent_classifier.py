from typing import Optional
import openai
from openai import AzureOpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


class IntentClassifier:
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        classification_prompt: Optional[str] = None
    ):
        """
        Classify the intent of user input using Azure OpenAI's GPT deployment.

        Args:
            api_key (str): Azure OpenAI API key.
            api_base (str): Azure OpenAI API base URL.
            api_version (str): API version (e.g., "2023-12-01-preview").
            deployment_name (str): The name of your deployed GPT model.
            classification_prompt (str, optional): Custom prompt template.
        """
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.deployment_name = deployment_name
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
        """
        Classify the intent of the user input.

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

        response = self.client.chat.completions.create(
            model=self.deployment_name,  # this should be the deployment name, not a model ID like "gpt-4"
            messages=messages,
            max_tokens=50,
            temperature=0.3,
        )

        message_content = response.choices[0].message.content
        return message_content.strip() if message_content else "4"
