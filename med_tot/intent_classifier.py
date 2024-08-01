import openai

class IntentClassifier:
    """
    A class to classify the intent of a user's input using OpenAI's GPT-3.5-turbo model.
    """
    def __init__(self, api_key: str):
        """
        Initialize the IntentClassifier with the provided OpenAI API key.

        Args:
            api_key (str): The OpenAI API key for accessing the GPT model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def classify_intent(self, user_input: str) -> str:
        """
        Classify the intent of the user's input.

        Args:
            user_input (str): The input text from the user.

        Returns:
            str: The classified intent number as a string.
        """
        prompt = f"""
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

        messages = [
            {"role": "system", "content": "You are a helpful assistant classifying user intent."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1,
                n=1,
                temperature=0.3
            )

            intent = response.choices[0].message['content'].strip()
            return intent
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return "4"  # Default to information request in case of error
