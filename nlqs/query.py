import json
import logging
import chromadb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAI

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    quantitative_data: Dict[str, str]
    qualitative_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


# Default system prompt for the LLM.
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional medical assistant, adept at handling inquiries related to medical products."
)


# Generates a prompt for the LLM based on the instruction and system prompt.
def get_prompt(instruction: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Generates the prompt for the LLM.

    Args:
        instruction (str): The instruction for the LLM.
        system_prompt (str, optional): The system prompt for the LLM. Defaults to DEFAULT_SYSTEM_PROMPT.

    Returns:
        str: The prompt for the LLM.

    """
    SYSTEM_PROMPT = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    return f"[INST]{SYSTEM_PROMPT}{instruction}[/INST]"


# Function to identify qualitative and quantitative data and user intent
def summarize(
    user_input: str,
    chat_history: List[Tuple[str, str]],
    column_descriptions_dictionary: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    llm: Union[ChatOpenAI, OpenAI],
) -> SummarizedInput:
    """Summarizes the user input and returns the summary, quantitative data, and qualitative data, along with the user requested columns in a JSON format.

    Args:
        user_input (str): The user input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions (dict[str, str]): The column descriptions.
        numerical_columns (list[str]): The numerical columns.
        categorical_columns (list[str]): The categorical columns.
        llm (Union[ChatOpenAI, OpenAI]): The LLM object.(Contains the details of the language we are using.)

    Returns:
        dict: {
            "summary": str,
            "quantitative_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "qualitative_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "user_requested_columns": list,
            "user_intent":str,
        }
    """

    column_descriptions = list(column_descriptions_dictionary.items())

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You will receive a user input and the chat history. Your task is to:
                
                1. **Single-Word Queries**: If the user input is a single word or very short (e.g., one or two words), provide a direct response if possible. If the query is unclear, prompt the user to elaborate.
                - Example response: "It seems you're asking about something specific. Could you provide more details?"

                2. **Structured Analysis**: For all other inputs, analyze the user input and identify key details based on our available data and chat history.
                
                3. Summarize the input, classifying the data into qualitative and quantitative categories.
                
                4. Identify relevant columns from which we can provide an answer. Pay close attention to the user's intent and specific mentions of data columns:
                - Are they seeking information about products, medications, treatments, or other relevant categories?
                - If the user is seeking information about a product, also provide the URL of the product if available.
                - Look for explicit mentions of column names, synonyms, or phrases that indicate the type of information requested. If the user specifies certain attributes or metrics, consider these as user-requested columns.

                5. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.

                6. Output the result in a JSON format.

                7. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)
                
                The output JSON should have the following structure:
                `
                    "summary": "summary of the user input",
                    "quantitative_data":
                                        ` 
                                        "column name": "Data mentioned about that column by the user. Example- < 4",
                                        "column name": "Data mentioned about that column by the user. Example- > 6.215",
                                        "column name": "Data mentioned about that column by the user. Example- >= 3.14 or <= 2.718",
                                        `,
                    "qualitative_data": 
                                        ` 
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        `,
                    "user_requested_columns": "List of columns the user wants data from. If none, leave it as an empty list. Always add product and url to this column.",
                    "user_intent": "The user's intent. If none, leave it as an empty string.",
                `
                
                The data we have and chat history:
                Data:{column_descriptions}\n\n 
                numerical columns in the data: {numerical_columns}\n\n 
                descriptive columns in the data: {categorical_columns}\n\n 
                chat history: {chat_history}

                Now, summarize the user input, chat history and provide the structured output in JSON format.
                """,
            ),
            ("human", f"{user_input}"),
        ]
    )

    # print(f"prompt: {prompt}")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    summarized_input_str = str(chain.invoke({"user_input": user_input}))

    print(f"summarized_input_str: {summarized_input_str}")

    print("------------------------------------------------------------------------")

    try:
        # Attempt to parse the summarized input as JSON
        summarized_input_dict = json.loads(summarized_input_str)
    except json.JSONDecodeError:
        # If parsing fails, return an empty SummarizedInput
        summarized_input_dict = {}

    logger.info("--------------------------")
    logger.info(f"user input: {user_input}")
    logger.info(f"Summarized input: {summarized_input_dict}")

    summarized_input = SummarizedInput(
        summary=summarized_input_dict.get("summary", ""),
        quantitative_data=summarized_input_dict.get("quantitative_data", {}),
        qualitative_data=summarized_input_dict.get("qualitative_data", {}),
        user_requested_columns=summarized_input_dict.get("user_requested_columns", []),
        user_intent=summarized_input_dict.get("user_intent", ""),
    )

    return summarized_input


def generate_quantitaive_serach_query(quantitaive_data: Dict[str, str], table_name: str, primary_key: str) -> str:
    """Creates an SQL query from a dictionary of quantitative data.

    Args:
        quantitaive_data (dict): A dictionary of quantitative data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query.
    """
    if not quantitaive_data:
        return ""  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in quantitaive_data.items():
        # Handle different comparison operators
        if "<" in condition:
            operator = "<"
        elif ">" in condition:
            operator = ">"
        elif "<=" in condition:
            operator = "<="
        elif ">=" in condition:
            operator = ">="
        elif "=" in condition:
            operator = "="
        else:
            operator = "LIKE"  # Default to LIKE for other conditions

        # Extract the value from the condition
        value = condition.replace(operator, "").strip()

        # Construct the query part
        query_part = f"{column} {operator} {value}"
        query_parts.append(query_part)

    # Combine the query parts with AND
    query_constraints = " AND ".join(query_parts)

    query = f"select {primary_key} from {table_name} where {query_constraints}"
    return query


def qualitative_search(collection: chromadb.Collection, data: Dict[str, str], primary_key: str) -> List[int]:
    """Performs a similarity search on the database and returns up to 5 similar results per column.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search.
        data (Dict[str, str]): A dictionary of qualitative data to search for.
        primary_key (str): The primary key column name in the database.

    Returns:
        List[int]: A list of unique IDs from the search results.
    """
    ids_per_column = {}

    for column, condition in data.items():
        query_result: chromadb.QueryResult = collection.query(
            query_texts=condition, n_results=5, where={"column_name": column}
        )

        if query_result["metadatas"]:
            ids_for_column = set()
            for result in query_result["metadatas"]:
                for item in result:
                    id_value = item.get(primary_key)
                    if id_value is not None:
                        ids_for_column.add(int(id_value))
            ids_per_column[column] = list(ids_for_column)

    print(f"ids_per_column: {ids_per_column}")

    # Flatten the list of lists into a single list of unique IDs
    all_ids = list(set([id_val for sublist in ids_per_column.values() for id_val in sublist]))
    return all_ids


# def qualitaive_search(collection: chromadb.Collection, data: Dict[str, str], primary_key: str) -> List[str]:
#     """Performs a similarity search on the database and returns all similar results.

#     Args:
#         collection (chromadb.Collection): The ChromaDB collection to search.
#         data (Dict[str, str]): A dictionary of qualitative data to search for.
#         primary_key (str): The primary key column name in the database.

#     Returns:
#         List[str]: A dictionary containing the search results.
#     """
#     all_ids = []

#     for column, condition in data.items():
#         query_result = collection.query(query_texts=condition, n_results=10, where={"column_name": column})

#         if query_result:
#             ids_for_column = set()  # Use a set to store unique IDs for this column
#             for result in query_result["metadatas"]:
#                 for item in result:
#                     id_value = item.get(primary_key)
#                     if id_value is not None:
#                         ids_for_column.add(str(id_value))  # Convert to string for comparison
#             all_ids.append(ids_for_column)

#     # Find the intersection of IDs across all columns
#     common_ids = set.intersection(*all_ids) if all_ids else set()

#     return list(common_ids)
