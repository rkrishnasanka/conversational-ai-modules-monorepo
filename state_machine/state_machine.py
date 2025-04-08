# app2.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pydantic import BaseModel
import os
from utils.llm import get_default_llm

class SystemConfig(BaseModel):
    """Configuration model for the recommendation system"""
    name: str
    description: str
    states: Dict[str, Dict[str, Any]]
    knowledge_base: Dict[str, List[str]]
    initial_message: str

@dataclass
class Response:
    """Standardized response structure"""
    bot_response: str
    follow_up_question: str
    sample_options: List[str] = field(default_factory=list)
    conversation_complete: bool = False
    state_complete: bool = False
    generated_queries: List[str] = field(default_factory=list)

class RecommendationSystem:
    """
    A flexible recommendation system that can be configured for different use cases.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the recommendation system with a specific configuration.
        
        Args:
            config (SystemConfig): System configuration including states and knowledge base
        """
        self.config = config
        
        # Initialize LLM using get_default_llm from llm.py
        self.llm = get_default_llm(use_azure=True)
        
        self.current_state = list(config.states.keys())[0]  # Start with first state
        self.context = {}
        self.conversation_history = []
        print(f"System initialized. Initial State: {self.current_state}")
        
    def chat(self, user_input: str) -> Response:
        """Process user input and generate a response."""
        print(f"Processing user input: {user_input}")
        response = self.process_input(user_input)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response.bot_response})
        return response

    def process_input(self, user_input: str) -> Response:
        """Process user input based on current state."""
        current_state_info = self.config.states[self.current_state]
        print(f"Processing input in state: {self.current_state}")
        print(f"State goal: {current_state_info['goal']}")
        
        # Generate LLM response
        response = self._get_llm_response(user_input, current_state_info)
        if not response:
            return self._get_fallback_response()

        # Update context with extracted information
        self.context.update(response.get("extracted_info", {}))
        print(f"Current context: {json.dumps(self.context, indent=2)}")
        
        # Generate queries if specified in state config
        queries = []
        if current_state_info.get("generate_queries", False):
            queries = self._generate_queries()
            print(f"Generated queries: {json.dumps(queries, indent=2)}")
        
        # Check missing required information
        missing_info = [info for info in current_state_info['required_info'] if info not in self.context]
        print(f"Missing required information: {missing_info}")
        
        # Check if state is complete
        if response.get("state_complete", False) or not missing_info:
            print(f"State {self.current_state} complete")
            self._transition_to_next_state()
        
        return Response(
            bot_response=response.get("bot_response", "I'm here to help."),
            follow_up_question=response.get("follow_up_question", ""),
            sample_options=response.get("sample_options", []),
            conversation_complete=self.current_state == list(self.config.states.keys())[-1],
            state_complete=response.get("state_complete", False),
            generated_queries=queries
        )

    def _get_llm_response(self, user_input: str, state_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get response using the LLM from llm.py."""
        try:
            prompt = f"""
            System: {self.config.name}
            Current state: {self.current_state}
            State goal: {state_info['goal']}
            Required information: {json.dumps(state_info['required_info'])}
            Current context: {json.dumps(self.context)}
            User input: "{user_input}"
            
            Respond with a JSON object containing:
            {{
                "bot_response": "A friendly, helpful response",
                "extracted_info": {{"key": "value"}},
                "follow_up_question": "A natural follow-up question if needed",
                "sample_options": ["Option 1", "Option 2", "Option 3"],
                "state_complete": true/false
            }}
            """
            
            messages = [
                {"role": "system", "content": self.config.description},
                {"role": "user", "content": prompt}
            ]
            
            # Using langchain chat model instead of direct OpenAI calls
            response = self.llm.invoke(messages)
            content = response.content
            
            # Parse the JSON response
            try:
                parsed_response = json.loads(content)
                print(f"LLM Response received and parsed successfully")
                return parsed_response
            except json.JSONDecodeError:
                print(f"Error parsing LLM response as JSON: {content}")
                # Try to extract JSON from a markdown code block if present
                if "```json" in content and "```" in content.split("```json", 1)[1]:
                    json_content = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    parsed_response = json.loads(json_content)
                    print(f"Successfully extracted JSON from markdown code block")
                    return parsed_response
                return None
            
        except Exception as e:
            print(f"Error in LLM response: {e}")
            return None

    def _generate_queries(self) -> List[str]:
        """Generate natural language queries based on context."""
        print("Generating natural language queries...")
        prompt = f"""
        Generate 5 natural language queries based on:
        Context: {json.dumps(self.context)}
        Knowledge base: {json.dumps(self.config.knowledge_base)}
        
        Return only a JSON array of query strings.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You generate relevant search queries based on context."},
                {"role": "user", "content": prompt}
            ]
            
            # Using langchain chat model
            response = self.llm.invoke(messages)
            content = response.content
            
            # Parse the JSON response
            try:
                queries = json.loads(content)
                if not isinstance(queries, list):
                    if isinstance(content, str) and "[" in content and "]" in content:
                        # Try to extract JSON array if present in the text
                        array_text = content[content.find("["):content.rfind("]")+1]
                        queries = json.loads(array_text)
                    else:
                        queries = []
                print(f"Generated {len(queries)} natural language queries")
                return queries
            except json.JSONDecodeError:
                print(f"Error parsing queries as JSON: {content}")
                # Try to extract JSON from a markdown code block if present
                if "```json" in content and "```" in content.split("```json", 1)[1]:
                    json_content = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    queries = json.loads(json_content)
                    if isinstance(queries, list):
                        print(f"Successfully extracted queries from markdown code block")
                        return queries
                return []
                
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []

    def _get_fallback_response(self) -> Response:
        """Provide fallback response."""
        print("Providing fallback response due to error")
        return Response(
            bot_response="I'm sorry, I didn't understand that. Could you please rephrase?",
            follow_up_question="What would you like to share?",
            sample_options=[],
            conversation_complete=False,
            state_complete=False
        )

    def _transition_to_next_state(self):
        """Transition to next state."""
        states = list(self.config.states.keys())
        current_index = states.index(self.current_state)
        if current_index < len(states) - 1:
            previous_state = self.current_state
            self.current_state = states[current_index + 1]
            print(f"Transitioned from {previous_state} to {self.current_state}")