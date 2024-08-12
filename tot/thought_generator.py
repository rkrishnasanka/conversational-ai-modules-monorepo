import openai
from typing import List
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam

class ThoughtGenerator:
    """
    Generates thoughts for problem-solving using OpenAI's GPT model.
    """

    def __init__(self, api_key: str, thought_generation_prompt: str):
        """
        Initialize the ThoughtGenerator.

        Args:
            api_key (str): OpenAI API key.
            thought_generation_prompt (str, optional): Custom prompt for thought generation.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        
        # Default thought generation prompt if not provided
        self.thought_generation_prompt = thought_generation_prompt or """
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
        Generate thoughts based on the current state.

        Args:
            current_state (str): The current state of the problem.
            num_thoughts (int): Number of thoughts to generate.

        Returns:
            List[str]: List of generated thoughts.
        """
        prompt = self.thought_generation_prompt.format(
            current_state=current_state,
            num_thoughts=num_thoughts
        )
        
        messages = [
            ChatCompletionSystemMessageParam(role= "system", content= "You are a helpful assistant generating thoughts for problem-solving."),
            ChatCompletionUserMessageParam(role="user", content= prompt)
        ]
        
        # Call OpenAI API for thought generation
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            temperature=0.7
        )

        # Extract and process thoughts from the response
        thoughts_text = response.choices[0].message.content #['content'].strip()
        if thoughts_text is not None:
            thoughts_text = thoughts_text.strip()
            thoughts = [thought.split('. ', 1)[1] for thought in thoughts_text.split('\n') if '. ' in thought]
            return thoughts[:num_thoughts]  # Ensure we return exactly num_thoughts thoughts
        else:
            raise Exception("No thoughts generated.")
