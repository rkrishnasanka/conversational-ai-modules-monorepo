import openai
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from sample_data_manager import SampleDataManager
from intent_classifier import IntentClassifier
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam


class TreeOfThoughts:
    """
    A class to implement the Tree of Thoughts methodology for processing user input and generating analysis.

    This class orchestrates the process of intent classification, thought generation, state evaluation,
    and final output generation based on user input and sample data.
    """

    def __init__(
        self, 
        api_key: str, 
        sample_data: Optional[str], 
        classification_prompt: str, 
        thought_generation_prompt: Optional[str] = None, 
        state_evaluation_prompt: Optional[str] = None, 
        json_output_prompt: Optional[str] = None
    ):
        """
        Initialize the TreeOfThoughts instance.

        Args:
            api_key (str): API key for OpenAI services.
            sample_data_manager (SampleDataManager): Instance for handling sample data.
            intent_classifier (IntentClassifier): Instance for classifying user intents.
            thought_generator (ThoughtGenerator): Instance for generating thoughts.
            state_evaluator (StateEvaluator): Instance for evaluating states.
            json_output_prompt (str, optional): Custom prompt for JSON output generation.
        """
        if not api_key or not sample_data or not classification_prompt or not thought_generation_prompt or not state_evaluation_prompt:
            raise ValueError("API key and all component instances must be provided")

        
        openai.api_key = api_key

        if sample_data:
            self.sample_data_manager = SampleDataManager(sample_data)
        self.intent_classifier = IntentClassifier(
            api_key=api_key,
            classification_prompt=classification_prompt,
        )

        if thought_generation_prompt:
            thought_generation_prompt = "TODO: Implement default thought generation prompt"
            raise NotImplementedError("Default thought generation prompt not implemented")
        
        self.thought_generator: ThoughtGenerator = ThoughtGenerator(
            api_key=api_key,
            thought_generation_prompt=thought_generation_prompt
        )

        if state_evaluation_prompt:
            state_evaluation_prompt = "TODO: Implement default state evaluation prompt"
            raise NotImplementedError("Default state evaluation prompt not implemented")
        
        self.state_evaluator: StateEvaluator = StateEvaluator(
            api_key=api_key,
            evaluation_prompt=state_evaluation_prompt
        )
        
        if not json_output_prompt:
            json_output_prompt = "TODO: Implement default JSON output prompt"
            raise NotImplementedError("Default JSON output prompt not implemented")
        
        self.json_output_prompt = json_output_prompt
        
        self.logger = logging.getLogger(__name__)


    def solve(
        self, 
        user_input: str, 
        chat_history: List[str], 
        num_thoughts: int = 3, 
        max_steps: int = 3, 
        best_states_count: int = 2
    ) -> Dict[str, Any]:
        """
        Processes the user input, analyzes the intent, and performs a tree search to generate the output.

        Args:
            user_input (str): The user's query or input.
            chat_history (List[str]): The history of the chat conversation.
            num_thoughts (int, optional): Number of thoughts to generate at each step. Defaults to 3.
            max_steps (int, optional): Maximum number of steps for the tree search. Defaults to 3.
            best_states_count (int, optional): Number of best states to retain at each step. Defaults to 2.

        Returns:
            Dict[str, Any]: A dictionary containing the summary, quantitative data, qualitative data, 
                            user requested columns, and intent.

        Raises:
            ValueError: If the user_input is empty.
        """
        if not user_input:
            raise ValueError("user_input must not be empty")

        self.logger.info(f"Starting solve method with user input: {user_input}")
        intent = self.intent_classifier.classify_intent(user_input)
        self.logger.info(f"Classified intent: {intent}")

        # Handle special intents
        if intent in ['1', '2', '3', '5']:
            error_messages = {
                '1': "The user has made a phatic communication.",
                '2': "The user has used profanity or vulgar language.",
                '3': "The user has attempted an SQL injection.",
                '5': "The user's query is not related to the available data."
            }
            self.logger.warning(f"Intent classified as {intent}. Returning error message.")
            return {
                "summary": error_messages[intent],
                "quantitative_data": {},
                "qualitative_data": {},
                "user_requested_columns": [],
                "intent": ["phatic_communication", "profanity", "sql_injection", "other"][int(intent) - 1],
            }

        initial_state = f"User Input: {user_input}\nChat History: {chat_history}"
        self.logger.info("Starting tree search")
        best_state, thought_path = self._tree_search(initial_state, num_thoughts, max_steps, best_states_count)
        self.logger.info("Tree search completed")
        return self._generate_json_output(best_state, thought_path)

    def _tree_search(
        self, 
        initial_state: str, 
        num_thoughts: int, 
        max_steps: int, 
        best_states_count: int
    ) -> Tuple[str, List[str]]:
        """
        Performs a tree search to find the best state and thought path.

        Args:
            initial_state (str): The initial state of the search.
            num_thoughts (int): Number of thoughts to generate at each step.
            max_steps (int): Maximum number of steps for the search.
            best_states_count (int): Number of best states to retain at each step.

        Returns:
            Tuple[str, List[str]]: The best state and the path of thoughts leading to it.
        """
        self.logger.info(f"Starting tree search with num_thoughts={num_thoughts}, max_steps={max_steps}, best_states_count={best_states_count}")
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

            # Evaluate and select best states
            values = self.state_evaluator.evaluate_states([state for state, _ in new_states])
            states = sorted(zip(new_states, values), key=lambda x: x[1], reverse=True)[:best_states_count]
            states = [state for state, _ in states]
            self.logger.info(f"Selected {len(states)} best states")

        self.logger.info("Tree search completed")
        return states[0] if states else (initial_state, [])

    def _generate_json_output(
        self, 
        final_state: str, 
        thought_path: List[str]
    ) -> Dict[str, Any]:
        """
        Generates the final JSON output based on the best state and thought path.

        Args:
            final_state (str): The best state found by the tree search.
            thought_path (List[str]): The path of thoughts leading to the best state.

        Returns:
            Dict[str, Any]: A dictionary containing the generated output.
        """
        self.logger.info("Generating JSON output")
        sample_data = self.sample_data_manager.get_sample_data()

        # Extract user query from final state
        user_query = final_state.split('\n')[0].replace('User Input: ', '')

        if not self.json_output_prompt:
            raise ValueError("JSON output prompt is not provided")

        prompt = self.json_output_prompt.format(
            user_query=user_query,
            final_state=final_state,
            thought_path=thought_path,
            sample_data=sample_data
        )

        messages = [
            ChatCompletionSystemMessageParam(role= "system", content= "You are a helpful assistant generating detailed JSON output based on analysis results."),
            ChatCompletionUserMessageParam(role="user", content= prompt)
        ]

        self.logger.info("Sending request to OpenAI API")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            n=1,
            temperature=0.2
        )
        self.logger.info("Received response from OpenAI API")

        # Process and clean the JSON string
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content received in response to the openai completion request")

        json_string = content.strip()
        
        if json_string.startswith('```json'):
            json_string = json_string[7:]
        if json_string.endswith('```'):
            json_string = json_string[:-3]
        json_string = json_string.strip()

        try:
            json_output = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            self.logger.error(f"Response text: {json_string}")
            raise json.JSONDecodeError(msg="Error decoding JSON response from OpenAI", doc=e.doc, pos=e.pos)
    
        # Ensure 'user_requested_columns' is present in the output
        if 'user_requested_columns' not in json_output or not json_output['user_requested_columns']:
            json_output['user_requested_columns'] = []

        self.logger.info("JSON output generated successfully")
        return json_output
