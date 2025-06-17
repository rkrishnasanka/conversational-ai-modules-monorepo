import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from state_machine.bot.prompts import (
    get_classification_prompt,
    get_evaluation_prompt,
    get_json_output_prompt,
    get_sample_data,
    get_thought_generation_prompt,
)
from state_machine.bot.session import initialize_session_data, update_session_data
from tot.tree_of_thoughts_executor import ToTExecutorInputs, TreeOfThoughtsExecutor


class CannabisRecommendationBot:
    """
    The CannabisRecommendationBot class handles interactions with the user, manages session data,
    and utilizes the Tree of Thoughts framework to provide cannabis product recommendations based on user inputs.
    """

    def __init__(self):
        api_key = self._get_api_key()
        self.executor = TreeOfThoughtsExecutor(
            ToTExecutorInputs(
                api_key=api_key,
                json_output_prompt=get_json_output_prompt(),
                classification_prompt=get_classification_prompt(),
                thought_generation_prompt=get_thought_generation_prompt(),
                evaluation_prompt=get_evaluation_prompt(),
                sample_csv_data=get_sample_data(),
            )
        )
        self.session_data = initialize_session_data()
        self.asked_questions = set()

    @staticmethod
    def _get_api_key() -> str:
        """
        Retrieve the API key from environment variables.
        Raises:
            ValueError: If the API key is not found.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key

    def process_user_input(self, user_input: str, chat_history: List[Tuple[str, str]]) -> str:
        """
        Process the user input by executing the Tree of Thoughts framework and returning the bot's response.

        Args:
            user_input (str): The user's input or query.

        Returns:
            str: The bot's formatted response.
        """
        self._log_user_input(user_input)

        try:
            tot_input = self._prepare_tot_input(user_input)
            tot_output = self.executor.execute(user_query=tot_input, chat_history=chat_history)
        except Exception as e:
            logging.error(f"Error occurred while executing Tree of Thoughts: {str(e)}")
            return "I'm sorry, but I'm having trouble processing your request right now. Could you please try again?"

        if not tot_output or not isinstance(tot_output, dict) or "response" not in tot_output:
            logging.error(f"Invalid response received from Tree of Thoughts executor: {tot_output}")
            return "I apologize, but I received an invalid response. Could you please rephrase your question?"

        update_session_data(self.session_data, tot_output)
        return self._format_response(tot_output)

    def _prepare_tot_input(self, user_input: str) -> str:
        """
        Prepare the input string for the Tree of Thoughts executor.

        Args:
            user_input (str): The user's input or query.

        Returns:
            str: Formatted input string for the executor.
        """
        return f"User Query: {user_input}\n"

    def _log_user_input(self, user_input: str):
        """
        Log user inputs and initialize the session data if it is the first input.

        Args:
            user_input (str): The user's input or query.
        """
        if not self.session_data["user_context"]["initial_query"]:
            self.session_data["user_context"]["initial_query"] = {"text": user_input, "intent": ""}

        self.session_data["user_context"]["interactions"].append(
            {"user_input": user_input, "timestamp": datetime.now().isoformat()}
        )

    def _format_response(self, tot_output: Dict[str, Any]) -> str:
        """
        Format the response based on the type of output from the Tree of Thoughts executor.

        Args:
            tot_output (Dict[str, Any]): Output from the executor.

        Returns:
            str: Formatted response.
        """
        response = tot_output["response"]
        if response["type"] == "question":
            return self._format_question(response)
        elif response["type"] == "recommendation":
            return self._format_recommendation(tot_output)
        return response["text"]

    def _format_question(self, response: Dict[str, Any]) -> str:
        """
        Format the bot's question with available options.

        Args:
            response (Dict[str, Any]): The response object containing question and options.

        Returns:
            str: Formatted question with options.
        """
        question = response["text"]
        options = response.get("options", [])
        if options:
            options_str = "\n".join(f"{chr(97 + i)}. {opt}" for i, opt in enumerate(options))
            return f"{question}\n\n{options_str}"
        return question

    def _format_recommendation(self, tot_output: Dict[str, Any]) -> str:
        """
        Format the recommendation response.

        Args:
            tot_output (Dict[str, Any]): Output from the executor including recommendations.

        Returns:
            str: Formatted recommendation.
        """
        recommendation = tot_output.get("recommendation", {})
        specific_products = ", ".join(recommendation.get("specific_products", []))
        return f"""
        Based on our conversation, here's our cannabis product recommendation:

        Product Type: {recommendation.get('product_type', 'Not specified')}
        Cannabinoid Profile: {recommendation.get('cannabinoid_profile', 'Not specified')}
        Usage Instructions: {recommendation.get('usage_instructions', 'Not specified')}
        Specific Product Recommendations: {specific_products}

        {tot_output.get('explanation', 'No additional explanation provided.')}

        Please consult with a healthcare professional before starting any new cannabis regimen.
        """
