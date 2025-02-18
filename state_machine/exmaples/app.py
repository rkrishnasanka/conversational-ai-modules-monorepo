import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import openai
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)


class State(Enum):
    """
    Enum representing the different states of the conversation.
    """

    INITIAL_INQUIRY = auto()
    MEDICAL_ASSESSMENT = auto()
    RECOMMENDATION = auto()
    CONCLUSION = auto()


@dataclass
class Response:
    """
    Data class for standardized responses.
    """

    bot_response: str
    follow_up_question: str
    sample_options: List[str] = field(default_factory=list)
    conversation_complete: bool = False
    state_complete: bool = False
    natural_language_queries: List[str] = field(default_factory=list)


class CannabisRecommendationSystem:
    """
    A system for providing medical cannabis recommendations based on user input and state management.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationSystem.

        Args:
            api_key (str): The API key for OpenAI.
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.state = State.INITIAL_INQUIRY
        self.context = {}
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge_base()
        self.generated_queries = []
        self.state_requirements = {
            State.INITIAL_INQUIRY: {
                "required_info": ["is_medical_query", "main_symptom"],
                "goal": "Understand the user's initial inquiry and identify the main symptom.",
                "initial_message": "Hi! I'm here to help recommend products for your needs. Could you tell me what brings you here today?",
            },
            State.MEDICAL_ASSESSMENT: {
                "required_info": [
                    "additional_symptoms",
                    "symptom_severity",
                    "symptom_duration",
                    "previous_treatments",
                    "medical_history",
                    "lifestyle_factors",
                ],
                "goal": "Gather comprehensive information about the user's health condition and symptoms.",
            },
            State.RECOMMENDATION: {
                "required_info": [
                    "suitable_products",
                    "usage_guidelines",
                    "precautions",
                    "expected_effects",
                    "user_preferences",
                ],
                "goal": "Generate natural language queries and provide user-friendly product recommendations.",
            },
            State.CONCLUSION: {
                "required_info": [
                    "user_satisfaction",
                    "understood_recommendations",
                    "remaining_concerns",
                    "next_steps",
                ],
                "goal": "Ensure recommendations are understood and provide clear next steps.",
            },
        }
        print(f"System initialized. Initial State: {self.state.name}")

    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base with cannabis-related information.
        """
        try:
            return {
                "product_types": ["Oil", "Vape", "Edible", "Pill", "Flower"],
                "effects": ["Calming", "Uplifting", "Balanced", "Sleep-aid", "Energizing", "Relaxing"],
                "onset_times": ["Quick", "Fast", "Medium", "Slow"],
                "durations": ["Short", "Medium", "Long"],
                "use_times": ["Any time", "Daytime", "Nighttime"],
                "common_issues": ["Pain", "Anxiety", "Sleep", "Nausea", "Mood", "Inflammation", "Headache"],
                "strengths": ["Mild", "Moderate", "Strong"],
                "active_compounds": ["Compound A", "Compound B", "Compound C", "Compound D", "Compound E"],
            }
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {}

    def generate_natural_language_queries(self) -> List[str]:
        """
        Generate natural language queries based on the gathered context.
        """
        try:
            prompt = f"""
            Generate 5 natural language queries based on the following user context:
            {json.dumps(self.context)}
            
            Rules for query generation:
            1. Use simple, clear language without medical jargon
            2. Focus on the main symptom and user preferences
            3. Include queries about product types, effects, and ratings
            4. Make queries suitable for conversion to SQL/Cypher queries
            5. Use the context information to make queries specific and relevant
            
            Format: Return only a JSON array of query strings
            """

            response = self.get_openai_response_text(
                prompt, "Generate natural language queries for product recommendations."
            )
            if response:
                try:
                    queries = json.loads(response)
                    if isinstance(queries, list):
                        print(f"Generated {len(queries)} natural language queries")
                        return queries
                except json.JSONDecodeError:
                    print("Failed to parse generated queries as JSON array")
            return []
        except Exception as e:
            print(f"Error generating natural language queries: {e}")
            return []

    def chat(self, user_input: str) -> Response:
        """
        Process user input and generate a response.
        """
        print(f"Processing user input: {user_input}")

        self.conversation_history.append(ChatCompletionUserMessageParam(content=user_input, role="user"))
        response = self.process_input(user_input)
        self.conversation_history.append(
            ChatCompletionSystemMessageParam(content=response.bot_response, role="assistant")
        )
        return response

    def process_input(self, user_input: str) -> Response:
        """
        Process the user input based on the current state and generate a response.
        """
        current_state_info = self.state_requirements[self.state]
        print(f"Processing input in state: {self.state.name}")

        response = self.get_llm_response(user_input, current_state_info)
        if not response:
            return self.get_fallback_response()

        self.context.update(response.get("extracted_info", {}))
        print(f"Current context: {json.dumps(self.context, indent=2)}")

        if self.state == State.RECOMMENDATION:
            queries = self.generate_natural_language_queries()
            response["natural_language_queries"] = queries
            self.generated_queries = queries
            print(f"Generated queries: {json.dumps(queries, indent=2)}")

        missing_info = [info for info in current_state_info["required_info"] if info not in self.context]
        print(f"Missing info: {missing_info}")

        if not missing_info or response.get("state_complete", False):
            print(f"State {self.state.name} complete")
            self.transition_to_next_state()

        if self.state == State.CONCLUSION:
            response["conversation_complete"] = True

        return Response(
            bot_response=response.get("bot_response", "I'm here to assist you."),
            follow_up_question=response.get("follow_up_question", ""),
            sample_options=response.get("sample_options", []),
            conversation_complete=response.get("conversation_complete", False),
            state_complete=response.get("state_complete", False),
            natural_language_queries=response.get("natural_language_queries", []),
        )

    def get_openai_response_text(self, prompt: str, system_message: str) -> Optional[str]:
        """
        Get raw text response from OpenAI API with error handling.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    ChatCompletionSystemMessageParam(content=system_message, role="system"),
                    ChatCompletionUserMessageParam(content=prompt, role="user"),
                ],
                temperature=0.7,
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return None
        except Exception as e:
            print(f"Error getting OpenAI response: {e}")
            return None

    def get_llm_response(self, user_input: str, state_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a response using the OpenAI language model with improved error handling.
        """
        try:
            prompt = f"""
            Analyze the following user input for a product recommendation system:
            User input: "{user_input}"
            Current state: {self.state.name}
            State goal: {state_info['goal']}
            Required information: {json.dumps(state_info['required_info'])}
            Current context: {json.dumps(self.context)}
            Knowledge base: {json.dumps(self.knowledge_base)}
            
            Respond with a JSON object containing:
            {{
                "bot_response": "A friendly, helpful response using simple language",
                "extracted_info": {{"key": "value"}},
                "follow_up_question": "A natural follow-up question if needed",
                "sample_options": ["Option 1", "Option 2", "Option 3"],
                "state_complete": true/false
            }}
            """

            response_text = self.get_openai_response_text(
                prompt, "You are a friendly product recommendation assistant. Use simple language."
            )

            if response_text:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}\nResponse text: {response_text}")
                    return None
            return None
        except Exception as e:
            print(f"Error in get_llm_response: {e}")
            return None

    def get_fallback_response(self) -> Response:
        """
        Provide a fallback response in case of errors.
        """
        return Response(
            bot_response="I'm sorry, I didn't understand that. Can you please clarify?",
            follow_up_question="What else would you like to share?",
            sample_options=[],
            conversation_complete=False,
            state_complete=False,
        )

    def transition_to_next_state(self):
        """
        Transition to the next state based on the current state.
        """
        if self.state == State.INITIAL_INQUIRY:
            self.state = State.MEDICAL_ASSESSMENT
        elif self.state == State.MEDICAL_ASSESSMENT:
            self.state = State.RECOMMENDATION
        elif self.state == State.RECOMMENDATION:
            self.state = State.CONCLUSION
        elif self.state == State.CONCLUSION:
            print("Conversation concluded. Thank you!")
        print(f"Transitioned to state: {self.state.name}")


import os

from dotenv import load_dotenv


class CannabisRecommendationApp:
    """
    Application class for running the Cannabis Recommendation System.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationApp.
        """
        self.system = CannabisRecommendationSystem(api_key)
        print("Application initialized")

    def run(self):
        """
        Run the Cannabis Recommendation Application.
        """
        print("Welcome to the Product Recommendation Assistant!")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    print("Please type something to continue.")
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Thank you for using the Product Recommendation Assistant. Goodbye!")
                    break

                response = self.system.chat(user_input)
                print(f"\nAssistant: {response.bot_response}")

                if self.system.state == State.RECOMMENDATION and response.natural_language_queries:
                    print("\nGenerated Queries:")
                    for i, query in enumerate(response.natural_language_queries, 1):
                        print(f"{i}. {query}")

                if response.follow_up_question and self.system.state != State.CONCLUSION:
                    print(f"\n{response.follow_up_question}")
                    if response.sample_options:
                        print("\nSample responses:")
                        for i, option in enumerate(response.sample_options, 1):
                            print(f"{i}. {option}")

                if response.conversation_complete:
                    print("\nThank you for using the Product Recommendation Assistant. Take care!")
                    break

            except KeyboardInterrupt:
                print("\nGoodbye! Thank you for using the Product Recommendation Assistant.")
                break
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                print("I apologize, but something went wrong. Let's continue our conversation.")


def main():
    """
    Main function to run the Cannabis Recommendation Application.
    """
    try:
        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("API key not found in environment variables")
            raise ValueError("API key not found")

        print("Starting application")
        app = CannabisRecommendationApp(api_key)
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        print("An error occurred while starting the application. Please check your configuration.")


if __name__ == "__main__":
    main()
