import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAI

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""
    summary: str
    numerical_data: Dict[str, str]
    categorical_data: Dict[str, str]
    descriptive_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str

# Default system prompt for the LLM.
DEFAULT_SYSTEM_PROMPT = (
    "You are a data analyst specializing in cannabis product information. You excel at understanding product measurements and translating descriptive terms into specific ranges."
)

def get_prompt(instruction: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Generates the prompt for the LLM."""
    SYSTEM_PROMPT = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    return f"[INST]{SYSTEM_PROMPT}{instruction}[/INST]"

def summarize(
    user_input: str,
    chat_history: List[Tuple[str, str]],
    column_descriptions_dictionary: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    descriptive_columns: List[str],
    llm: Union[ChatOpenAI, OpenAI, AzureChatOpenAI],
) -> SummarizedInput:
    """Summarizes the user input and returns structured data."""
    column_descriptions = list(column_descriptions_dictionary.items())

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a precise data analyst for cannabis product queries.
            
Your task is to convert natural language queries into an EXACT structured format.

1. CBD/THC Content Rules:
   HIGH or STRONG CBD = EXACTLY ">15" 
   LOW or MILD CBD = EXACTLY "<5"
   MEDIUM CBD = EXACTLY "5-15"
   Exact numbers = Use as given (e.g., ">20", "<=30")

2. Required Output Structure:
Return ONLY a JSON object with EXACTLY this structure:
{{
    "summary": "Brief description of request",
    "numerical_data": {{
        "CBD": ">15",  # For high CBD, always use >15
        "THC": "value if mentioned"
    }},
    "categorical_data": {{
        "Category": "product type if mentioned"
    }},
    "descriptive_data": {{
        "Description": "notable properties",
        "MedicalBenefitsReported": "benefits if mentioned"
    }},
    "user_requested_columns": [
        "Product",  # Always include these
        "CBD",
        "URL", 
        "Description",
        "MedicalBenefitsReported"
    ],
    "user_intent": "search"  # Always use "search"
}}

Database Info:
- Numerical: {numerical_columns}
- Categorical: {categorical_columns}
- Descriptive: {descriptive_columns}
- Details: {column_descriptions}
- History: {chat_history}

CRITICAL RULES:
1. Always output valid JSON with ALL fields
2. Always use ">15" for high CBD
3. Always include all 5 columns listed
4. Always set user_intent to "search"\n"""),
        ("human", user_input),
    ])

    # Generate the summarized input
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    summarized_input_str = str(chain.invoke({"user_input": user_input}))

    print(f"summarized_input_str: {summarized_input_str}")
    print("-" * 72)

    # Parse and validate the LLM output
    try:
        summarized_input_dict = json.loads(summarized_input_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM output as JSON")
        summarized_input_dict = {
            "summary": user_input,
            "numerical_data": {},
            "categorical_data": {},
            "descriptive_data": {},
            "user_requested_columns": [],
            "user_intent": "search"
        }

    logger.info("-" * 26)
    logger.info(f"user input: {user_input}")
    logger.info(f"Summarized input: {summarized_input_dict}")

    # Ensure CBD content is properly captured
    if "numerical_data" not in summarized_input_dict:
        summarized_input_dict["numerical_data"] = {}
    
    if (
        "high" in user_input.lower() 
        and "cbd" in user_input.lower()
        and ("CBD" not in summarized_input_dict["numerical_data"] 
        or not summarized_input_dict["numerical_data"]["CBD"])
    ):
        summarized_input_dict["numerical_data"]["CBD"] = ">15"

    # Ensure all required fields exist
    required_fields = {
        "summary": user_input,
        "numerical_data": {"CBD": ">15"} if "high" in user_input.lower() and "cbd" in user_input.lower() else {},
        "categorical_data": {},
        "descriptive_data": {},
        "user_requested_columns": ["Product", "CBD", "URL", "Description", "MedicalBenefitsReported"],
        "user_intent": "search"
    }

    for field, default in required_fields.items():
        if field not in summarized_input_dict or not summarized_input_dict[field]:
            summarized_input_dict[field] = default

    # Add required columns if missing
    for col in required_fields["user_requested_columns"]:
        if col not in summarized_input_dict["user_requested_columns"]:
            summarized_input_dict["user_requested_columns"].append(col)

    return SummarizedInput(
        summary=summarized_input_dict["summary"],
        numerical_data=summarized_input_dict["numerical_data"],
        categorical_data=summarized_input_dict["categorical_data"],
        descriptive_data=summarized_input_dict["descriptive_data"],
        user_requested_columns=summarized_input_dict["user_requested_columns"],
        user_intent=summarized_input_dict["user_intent"]
    )