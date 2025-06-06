import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from tot.intent_classifier import IntentClassifier
from tot.sample_data_manager import SampleDataManager
from tot.state_evaluator import StateEvaluator
from tot.thought_generator import ThoughtGenerator


class TreeOfThoughts:
    """
    A class to implement the Tree of Thoughts methodology for processing user input and generating analysis.
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        sample_data: str,
        classification_prompt: str,
        thought_generation_prompt: Optional[str],
        state_evaluation_prompt: str,
        json_output_prompt: str,
    ):
        """
        Initialize the TreeOfThoughts class with Azure OpenAI client and required components.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.deployment_name = deployment_name

        self.sample_data_manager = SampleDataManager(sample_data)
        self.intent_classifier = IntentClassifier(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            classification_prompt=classification_prompt,
        )
        self.thought_generator = ThoughtGenerator(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            thought_generation_prompt=thought_generation_prompt,
        )
        self.state_evaluator = StateEvaluator(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            evaluation_prompt=state_evaluation_prompt,
        )

        self.json_output_prompt = json_output_prompt
        self.logger = logging.getLogger(__name__)

    def solve(
        self,
        user_input: str,
        chat_history: List[Tuple[str, str]],
        num_thoughts: int = 3,
        max_steps: int = 3,
        best_states_count: int = 2,
    ) -> Dict[str, Any]:
        if not user_input:
            raise ValueError("user_input must not be empty")

        self.logger.info(f"Starting solve method with user input: {user_input}")
        intent = self.intent_classifier.classify_intent(user_input)
        self.logger.info(f"Classified intent: {intent}")

        initial_state = f"User Input: {user_input}\nChat History: {chat_history}"
        self.logger.info("Starting tree search")
        best_state, thought_path = self._tree_search(initial_state, num_thoughts, max_steps, best_states_count)
        self.logger.info("Tree search completed")
        return self._generate_json_output(best_state, thought_path, user_input)

    def _tree_search(
        self, initial_state: str, num_thoughts: int, max_steps: int, best_states_count: int
    ) -> Tuple[str, List[str]]:
        self.logger.info(
            f"Starting tree search with num_thoughts={num_thoughts}, max_steps={max_steps}, best_states_count={best_states_count}"
        )
        states = [(initial_state, [])]  # (state, path)

        for step in range(max_steps):
            self.logger.info(f"Tree search step {step + 1}")
            new_states = []
            for state, path in states:
                new_thoughts = self.thought_generator.generate_thoughts(state, num_thoughts)
                self.logger.debug(f"Generated {len(new_thoughts)} new thoughts")
                new_states.extend([(f"{state}\nThought: {thought}", path + [thought]) for thought in new_thoughts])

            if not new_states:
                self.logger.warning("No new states generated. Breaking tree search.")
                break

            values = self.state_evaluator.evaluate_states([state for state, _ in new_states])
            states = sorted(zip(new_states, values), key=lambda x: x[1], reverse=True)[:best_states_count]
            states = [state for state, _ in states]
            self.logger.info(f"Selected {len(states)} best states")

        self.logger.info("Tree search completed")
        return states[0] if states else (initial_state, [])

    def _generate_json_output(self, final_state: str, thought_path: List[str], user_query: str) -> Dict[str, Any]:
        self.logger.info("Generating JSON output")
        sample_data = self.sample_data_manager.get_sample_data() if self.sample_data_manager else ""

        prompt = self.json_output_prompt.format(
            user_query=user_query, final_state=final_state, thought_path=thought_path, sample_data=sample_data
        )

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful assistant generating detailed JSON output based on analysis results.",
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        self.logger.info("Sending request to Azure OpenAI API")
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.2,
        )
        self.logger.info("Received response from Azure OpenAI API")

        content = response.choices[0].message.content
        
        # print(f"Response content: {content}")
        
        if content is None:
            raise ValueError("No content received in response to the Azure OpenAI completion request")

        json_string = content.strip()
        json_string = re.sub("```json|```", "", json_string).strip()

        try:
            json_output = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            self.logger.error(f"Response text: {json_string}")
            raise

        self.logger.info("JSON output generated successfully")
        return json_output
