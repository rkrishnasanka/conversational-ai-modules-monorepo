import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from tot.tree_of_thoughts_executor import ToTExecutorInputs, TreeOfThoughtsExecutor

# Define the log directory and file path
log_directory = r"logs"
log_file = os.path.join(log_directory, "cannabis_bot.log")

# Create the log directory if it does not exist
os.makedirs(log_directory, exist_ok=True)

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TreeOfThoughtsOutputs:
    description: str
    options: List[Dict[str, str]]
    final_recommendation: Optional[str] = None


class CannabisRecommendationBot:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.executor = TreeOfThoughtsExecutor(
            ToTExecutorInputs(
                api_key=api_key,
                json_output_prompt=self.get_json_output_prompt(),
                classification_prompt=self.get_classification_prompt(),
                thought_generation_prompt=self.get_thought_generation_prompt(),
                evaluation_prompt=self.get_evaluation_prompt(),
                sample_csv_data=self.get_sample_data(),
            )
        )
        self.session_data = self.initialize_session_data()
        self.asked_questions = set()

    def initialize_session_data(self) -> Dict[str, Any]:
        return {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_context": {"initial_query": {}, "interactions": [], "final_summary": {}},
            "rag_analysis": {"contextual_analysis": {}, "recommendations": {}},
            "cannabis_preferences": {},
        }

    def get_json_output_prompt(self) -> str:
        return """
        User Query: {user_query}
        Final State: {final_state}
        Thought Path: {thought_path}
        Sample Data: {sample_data}

        Generate a JSON object with the following structure:
        {{
            "response": {{
                "text": "The next response or question to the user",
                "type": "question" or "recommendation" or "information",
                "options": ["List", "of", "relevant", "options"],
                "entities": ["List", "of", "key", "entities"],
                "intent": "Identified intent"
                "options": ["List", "of", "relevant", "options"],

            }},
            "recommendation": {{
                "product_type": "Recommended product type",
                "cannabinoid_profile": "Recommended cannabinoid profile",
                "usage_instructions": "Instructions for use",
                "specific_products": ["List", "of", "specific", "product", "recommendations"]
            }},
            "explanation": "Detailed explanation of the response or recommendation",
            "follow_up_questions": ["List", "of", "potential", "follow-up", "questions"],
            "contextual_analysis": "In-depth contextual analysis based on the conversation",
            "relationships": "Identified relationships between entities in the context"
        }}

        If a recommendation is not ready, set the "recommendation" field to null.
        Ensure the response is relevant to the conversation history and helps gather necessary information for a cannabis recommendation or provides appropriate information/recommendations.
        Base your responses and recommendations on the context of cannabis products and user preferences.
        Provide meaningful and diverse options for questions, tailored to the specific query and context.
        Generate a list of potential follow-up questions based on the current context and information gaps.
        Use the provided algorithm framework to guide the questioning process and product recommendations.
        """

    def get_classification_prompt(self) -> str:
        return """
        Classify the user's intent based on the following input:

        User Input: {user_input}

        Possible intents:
        1. Seeking cannabis product recommendation
        2. Providing information about preferences or needs
        3. Inquiring about cannabis effects or usage
        4. Other (not related to cannabis recommendations)

        Respond with only the number corresponding to the intent.
        """

    def get_thought_generation_prompt(self) -> str:
        return """
        Given the current state of the problem:

        {current_state}

        Generate {num_thoughts} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem.

        Your response should be in the following format:
        1. [First thought]
        2. [Second thought]
        ...
        {num_thoughts}. [Last thought]
        
        generate potential follow-up questions, responses, or recommendations in the cannabis recommendation process. 
        Use the following sample data structure to guide your questions:

        Consider the following aspects based on the sample data columns:
        1. Product Category: Ask about preferred consumption methods (e.g., Tincture, Vaporizer, Edible, Capsule, Flower)
        2. CBD and THC content: Inquire about desired cannabinoid ratios or potency
        3. Onset: Ask about how quickly the user needs the effects to start (e.g., Immediate, Fast, Medium, Slow)
        4. Duration: Inquire about how long they want the effects to last (e.g., Short, Medium, Long)
        5. Effects: Ask about desired effects (e.g., Non-euphoric, Euphoric, Balanced, Sedating, Energizing, Relaxing)
        6. TimeOfUse: When they plan to use the product (e.g., Any, Day, Night)

        Generate thoughts about what information is still needed, what cannabis product recommendations might be appropriate, or what information should be provided to the user based on the conversation so far.
        Ensure that follow-up questions and options are diverse, not repetitive, and tailored to the specific context of the conversation.
        For each question, provide a list of relevant options for the user to choose from, based on the unique values in the sample data.
        """

    def get_evaluation_prompt(self) -> str:
        return """
        Evaluate the potential next steps in the cannabis recommendation conversation based on the following criteria:
        1. Relevance to the user's stated needs and preferences for cannabis products
        2. Alignment with the algorithm framework for cannabis product recommendations
        3. Completeness of information gathered for making a cannabis recommendation
        4. Clarity and helpfulness of the response for the user
        5. Progression towards a suitable product recommendation

        Provide a score from 1-10 for each potential next step, where 10 is the best.
        """

    def get_sample_data(self) -> str:
        return """Product,Category,CBD,THC,Onset,Duration,Effects,TimeOfUse
        CBD Oil Tincture,Tincture,500,10,Fast,Medium,Non-euphoric,Any
        THC Vape Pen,Vaporizer,0,250,Immediate,Short,Euphoric,Any
        1:1 CBD:THC Gummies,Edible,100,100,Slow,Long,Balanced,Any
        CBN Sleep Capsules,Capsule,50,10,Medium,Long,Sedating,Night
        Sativa Flower,Flower,20,180,Immediate,Medium,Energizing,Day
        Indica Flower,Flower,30,220,Immediate,Medium,Relaxing,Night"""

    def process_user_input(self, user_input: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        if not self.session_data["user_context"]["initial_query"]:
            self.session_data["user_context"]["initial_query"] = {"text": user_input, "intent": ""}

        self.session_data["user_context"]["interactions"].append(
            {"user_input": user_input, "timestamp": datetime.now().isoformat()}
        )

        tot_input = f"User Query: {user_input}\n"

        try:
            tot_output = self.executor.execute(user_query=tot_input, chat_history=chat_history)
        except Exception as e:
            logging.error(f"Error occurred while executing Tree of Thoughts: {str(e)}")
            return {}

        if not tot_output or not isinstance(tot_output, dict) or "response" not in tot_output:
            logging.error(f"Invalid response received from Tree of Thoughts executor: {tot_output}")
            return {}

        self.update_session_data(tot_output)

        return tot_output

    def update_session_data(self, tot_output: Dict[str, Any]):
        response = tot_output["response"]
        self.session_data["user_context"]["interactions"][-1]["bot_response"] = response

        if response["type"] == "question":
            self.asked_questions.add(response["text"])

        if response["type"] == "recommendation":
            self.session_data["user_context"]["final_summary"] = {
                "text": tot_output["explanation"],
                "key_insights": response.get("entities", []),
                "recommendation": tot_output["recommendation"],
            }

        self.session_data["rag_analysis"]["contextual_analysis"] = {
            "text": tot_output["contextual_analysis"],
            "entities": response.get("entities", []),
            "relationships": tot_output["relationships"],
        }
        if tot_output.get("recommendation"):
            self.session_data["rag_analysis"]["recommendations"] = tot_output["recommendation"]

        for entity in response.get("entities", []):
            if entity not in self.session_data["cannabis_preferences"]:
                self.session_data["cannabis_preferences"][entity] = True

    # def format_response(self, tot_output: Dict[str, Any]) -> str:
    #     response = tot_output["response"]

    #     if response["type"] == "question":
    #         return self.format_question(response)
    #     elif response["type"] == "recommendation":
    #         return self.format_recommendation(tot_output)
    #     else:
    #         return response["text"]

    # def format_question(self, response: Dict[str, Any]) -> str:
    #     question = response["text"]
    #     options = response.get("options", [])

    #     if options:
    #         options_str = "\n".join(f"{chr(97 + i)}. {opt}" for i, opt in enumerate(options))
    #         return f"{question}\n\n{options_str}"
    #     else:
    #         return question

    def format_recommendation(self, tot_output: Dict[str, Any]) -> str:
        recommendation = tot_output.get("recommendation")
        if not recommendation:
            return "I'm sorry, but I don't have enough information to make a recommendation yet. Let me ask you a few more questions."

        specific_products = recommendation.get("specific_products", [])
        if specific_products and isinstance(specific_products[0], dict):
            specific_products_str = ", ".join(product.get("name", "Unknown") for product in specific_products)
        else:
            specific_products_str = ", ".join(map(str, specific_products))

        return f"""
        Based on our conversation, here's our cannabis product recommendation:

        Product Type: {recommendation.get('product_type', 'Not specified')}
        Cannabinoid Profile: {recommendation.get('cannabinoid_profile', 'Not specified')}
        Usage Instructions: {recommendation.get('usage_instructions', 'Not specified')}
        Specific Product Recommendations: {specific_products_str}

        {tot_output.get('explanation', 'No additional explanation provided.')}

        Please note that this is a general recommendation. Always consult with a healthcare professional before starting any new cannabis regimen, and ensure you're aware of the legal status of cannabis products in your area.
        """


def main():
    bot = CannabisRecommendationBot()
    print("Welcome to the Cannabis Recommendation Bot!")
    print("How can I assist you in finding the right cannabis product today?")
    chat_history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nThank you for using the Cannabis Recommendation Bot.")
            print("Here's a summary of our conversation:")
            print_session_summary(bot.session_data)
            print("\nGoodbye!")
            break

        response = bot.process_user_input(user_input, chat_history)
        if not response:
            print("I'm sorry, but I'm having trouble processing your request right now. Could you please try again?")

        recommendation_result = ""
        if response["response"]["type"] == "recommendation":
            recommendation_result = bot.format_recommendation(response)

        # response = json.loads(response)
        description = response.get("response", "").get("text", "")
        options = response.get("response", "").get("options", "")
        formatted_options = [{chr(97 + i): opt} for i, opt in enumerate(options)]

        result = TreeOfThoughtsOutputs(description, formatted_options, recommendation_result)
        print(f"Description: {result.description}")
        print(f"Options: {result.options}")

        chat_history.append((user_input, result))
        print(f"chat history: {chat_history}")

        print("\nRAG Analysis:")
        print(json.dumps(bot.session_data["rag_analysis"], indent=4))


def print_session_summary(session_data):
    summary = {
        "User Preferences": session_data.get("cannabis_preferences", {}),
        "Final Recommendation": session_data.get("rag_analysis", {}).get(
            "recommendations", "No recommendation provided"
        ),
        "Key Insights": session_data.get("user_context", {}).get("final_summary", {}).get("key_insights", []),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
