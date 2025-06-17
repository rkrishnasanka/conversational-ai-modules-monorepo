import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from sqlalchemy import column

from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import ColumnType, DataCollectionMetadata, VectorDBDriver
from utils.json_outputs import validate_llm_output_keys

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


class InputIntent(TypedDict):
    """Class to represent the input intent."""

    summary: str
    user_intent: str
    qualitative_statements: List[str]
    quantitative_statements: List[str]


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    numerical_data: Dict[str, str]
    categorical_data: Dict[str, str]
    descriptive_data: Dict[str, str]
    identifier_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


REFERENCE_SUMMARIZED_INTENT_DICT = {
    "summary": "",
    "user_intent": "",
    "qualitative_statements": [],
    "quantitative_statements": [],
}


REFERENCE_SUMMARIZED_OUTPUT_DICT = {
    "qualitative_data": {},
    "quantitative_data": {},
    "user_requested_columns": [],
}


# Default system prompt for the LLM.
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional data assistant, adept at handling inquiries across various domains and data types."
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


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from markdown code blocks or return the original text.
    
    Args:
        response_text (str): The response text that may contain JSON in markdown code blocks
        
    Returns:
        str: The extracted JSON string
    """
    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no markdown code blocks found, return the original text
    return response_text.strip()


# Function to identify qualitative and quantitative data and user intent
def summarize(
    user_input: str,
    chat_history: List[Tuple[str, str]],
    column_descriptions_dictionary: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    descriptive_columns: List[str],
    llm: Union[ChatOpenAI, OpenAI],
    vectordb: VectorDBDriver,
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
            "numerical_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "categorical_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "descriptive_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "user_requested_columns": list,
            "user_intent":str,
        }
    """

    # Updated NLQS Algorithm:
    # 1. Extract a list of qualitative and quantitative statements from the user input along with the user intent.
    # 2. Identify the relevant columns from the data based on the statements extracted and available column descriptions.
    # 3. Generate a structured output in JSON format with the summary, numerical data, categorical data, descriptive data,
    # user requested columns, and user intent.
    intent_classification_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You will receive a user input and the chat history. Your task is to:
                
                1. **Single-Word Queries**: If the user input is a single word or very short (e.g., one or two words), provide a direct response if possible. If the query is unclear, prompt the user to elaborate.
                - Example response: "It seems you're asking about something specific. Could you provide more details?"

                2. **Structured Analysis**: For all other inputs, analyze the user input and identify key details based on our available data and chat history.
                  3. Summarize the input, classifying the statements made by the user into qualitative and quantitative categories.
                
                4. Identify relevant columns from which we can provide an answer. Pay close attention to the user's intent and specific mentions of data columns:
                - Are they seeking information about specific entities, categories, or data types?
                - Look for explicit mentions of column names, synonyms, or phrases that indicate the type of information requested. If the user specifies certain attributes or metrics, consider these as user-requested columns.

                5. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.

                6. Output the result in a JSON format.

                7. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)
                  The output JSON should have the following structure:
                `
                    "summary": "summary of the user input",
                    "qualitative_statements":
                                        [
                                            "statement 1",
                                            "statement 2",
                                            "statement 3"
                                        ],
                    "quantitative_statements":
                                        [
                                            "statement 1",
                                            "statement 2",
                                            "statement 3"
                                        ],                    
                    "user_intent": "The user's intent. If none, leave it as an empty string.",
                `
                
                chat history: {chat_history}

                Now, summarize the user input, chat history and provide the structured output in JSON format.
                """,
            ),
            ("human", f"{user_input}"),
        ]
    )

    output_parser = StrOutputParser()
    chain = intent_classification_prompt | llm | output_parser

    summarized_input_intent_raw = str(chain.invoke({"user_input": user_input}))

    print(f"summarized_input_intent: {summarized_input_intent_raw}")
    print("------------------------------------------------------------------------")
    
    # Extract JSON from markdown code blocks if present
    summarized_input_intent_json = extract_json_from_response(summarized_input_intent_raw)
    
    # Attempt to parse the summarized input as JSON
    try:
        # Attempt to parse the summarized input as JSON
        summarized_input_dict = json.loads(summarized_input_intent_json)

        missing_keys = validate_llm_output_keys(
            llm_output=summarized_input_dict, reference_dict=REFERENCE_SUMMARIZED_INTENT_DICT
        )

        if len(missing_keys) > 0:
            logger.error(f"Missing keys in summarized_input_dict: {missing_keys}")
            raise ValueError("Missing keys in summarized_input_dict")
        else:
            # Insert into typed dict for summarized intent
            summarized_input_intent = InputIntent(
                summary=summarized_input_dict["summary"],
                user_intent=summarized_input_dict["user_intent"],
                qualitative_statements=summarized_input_dict["qualitative_statements"],
                quantitative_statements=summarized_input_dict["quantitative_statements"],
            )

    except json.JSONDecodeError as e:
        # If parsing fails, return an empty SummarizedInput
        logger.error(f"Error parsing summarized_input_intent for user input: {user_input}. Error: {str(e)}")
        logger.error(f"Raw response: {summarized_input_intent_raw}")
        logger.error(f"Extracted JSON: {summarized_input_intent_json}")
        summarized_input_intent = InputIntent(
            summary="", user_intent="", qualitative_statements=[], quantitative_statements=[]
        )

    column_descriptions = list(column_descriptions_dictionary.items())

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You will receive a user input and the chat history, and the set of qualitative and quantitative statements within the input. Your task is to:
                
                1. **Single-Word Queries**: If the user input is a single word or very short (e.g., one or two words), provide a direct response if possible. If the query is unclear, prompt the user to elaborate.
                - Example response: "It seems you're asking about something specific. Could you provide more details?"

                2. **Structured Analysis**: For all other inputs, analyze the user input and identify key details based on our available data and chat history.
                
                3. Summarize the input, classifying the data into qualitative and quantitative categories.
                  4. Identify relevant columns from which we can provide an answer. Pay close attention to the user's intent and specific mentions of data columns:
                - Are they seeking information about specific entities, categories, or data types?
                - Look for explicit mentions of column names, synonyms, or phrases that indicate the type of information requested. If the user specifies certain attributes or metrics, consider these as user-requested columns.

                5. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.

                6. Output the result in a JSON format.

                7. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)
                
                The output JSON should have the following structure:
                `
                    "qualitative_data": 
                                        ` 
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        `,
                    "quantitative_data":
                                        ` 
                                        "column name": "Data mentioned about that column by the user. Example- < 4",
                                        "column name": "Data mentioned about that column by the user. Example- > 6.215",
                                        "column name": "Data mentioned about that column by the user. Example- >= 3.14 or <= 2.718",
                                        `,
                    
                    "user_requested_columns": "List of columns the user wants data from. If none, leave it as an empty list.",
                `
                
                The data we have and chat history: 
                Data:{column_descriptions}\n\n 
                Qualitative statements: {summarized_input_intent['qualitative_statements']}\n\n
                Quantitative statements: {summarized_input_intent['quantitative_statements']}\n\n
                numerical columns in the data: {numerical_columns}\n\n 
                categorical columns in the data: {categorical_columns}\n\n
                descriptive columns in the data: {descriptive_columns}\n\n 
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

    summarized_input_str_raw = str(chain.invoke({"user_input": user_input}))

    print(f"summarized_input_str: {summarized_input_str_raw}")

    print("------------------------------------------------------------------------")

    # Extract JSON from markdown code blocks if present
    summarized_input_str_json = extract_json_from_response(summarized_input_str_raw)

    try:
        # Attempt to parse the summarized input as JSON
        summarized_input_dict = json.loads(summarized_input_str_json)

        missing_keys = validate_llm_output_keys(
            llm_output=summarized_input_dict, reference_dict=REFERENCE_SUMMARIZED_OUTPUT_DICT
        )

        if len(missing_keys) > 0:
            logger.error(f"Missing keys in summarized_input_dict: {missing_keys}")
            raise ValueError("Missing keys in summarized_input_dict")

    except json.JSONDecodeError as e:
        # If parsing fails, return an empty SummarizedInput
        logger.error(f"Error parsing summarized_input_dict for user input: {user_input}. Error: {str(e)}")
        logger.error(f"Raw response: {summarized_input_str_raw}")
        logger.error(f"Extracted JSON: {summarized_input_str_json}")
        summarized_input_dict = {}

    logger.info("--------------------------")
    logger.info(f"user input: {user_input}")
    logger.info(f"Summarized input: {summarized_input_dict}")

    # Validate the qualitative and quantitative columns against the available data columns /
    # pick the most relevant columns
    numerical_data = {}
    categorical_data = {}
    descriptive_data = {}
    identifier_data = {}

    # TODO: In the future this should also find the corresponding table from the databse
    # For now, we will use the default table and default db

    # Go through each of the qualitative and quantitative maps check if the column is present in the data
    for column_name, description in summarized_input_dict.get("quantitative_data", {}).items():
        if column_name not in column_descriptions_dictionary:
            closest_column_name, column_type = vectordb.get_closest_column_from_description(
                approximate_column_name=column_name,
                users_description=description,
                sample_data_strings=[],
                database_name=DEFAULT_DB_NAME,
                table_name=DEFAULT_TABLE_NAME,
            )

            if closest_column_name not in column_descriptions_dictionary:
                raise ValueError(f"Closest column name '{closest_column_name}' not found in chroma columns collection.")

            # Add the column to the corresponding dictionary
            if column_type == ColumnType.NUMERICAL:
                numerical_data[closest_column_name] = description
            elif column_type == ColumnType.CATEGORICAL:
                categorical_data[closest_column_name] = description
            elif column_type == ColumnType.DESCRIPTIVE:
                descriptive_data[closest_column_name] = description
            elif column_type == ColumnType.IDENTIFIER:
                identifier_data[closest_column_name] = description
            else:
                raise ValueError(f"Invalid column type '{column_type}' for column '{closest_column_name}'")

        else:
            # Add the column to the corresponding dictionary
            numerical_data[column_name] = description

    for column_name, description in summarized_input_dict.get("qualitative_data", {}).items():
        if column_name not in column_descriptions_dictionary.keys():
            closest_column_name, column_type = vectordb.get_closest_column_from_description(
                approximate_column_name=column_name,
                users_description=description,
                sample_data_strings=[],
                database_name=DEFAULT_DB_NAME,
                table_name=DEFAULT_TABLE_NAME,
            )

            if closest_column_name not in column_descriptions_dictionary:
                raise ValueError(f"Closest column name '{closest_column_name}' not found in chroma columns collection.")

            if column_type == ColumnType.NUMERICAL:
                numerical_data[closest_column_name] = description
            elif column_type == ColumnType.CATEGORICAL:
                categorical_data[closest_column_name] = description
            elif column_type == ColumnType.DESCRIPTIVE:
                descriptive_data[closest_column_name] = description
            elif column_type == ColumnType.IDENTIFIER:
                identifier_data[closest_column_name] = description
            else:
                raise ValueError(f"Invalid column type '{column_type}' for column '{closest_column_name}'")

        else:
            # Add the column to the corresponding dictionary after checking the column type
            column_type = vectordb.get_column_type(column_name, DEFAULT_TABLE_NAME, DEFAULT_DB_NAME)
            if column_type == ColumnType.NUMERICAL:
                numerical_data[column_name] = description
            elif column_type == ColumnType.CATEGORICAL:
                categorical_data[column_name] = description
            elif column_type == ColumnType.DESCRIPTIVE:
                descriptive_data[column_name] = description
            elif column_type == ColumnType.IDENTIFIER:
                identifier_data[column_name] = description
            else:
                raise ValueError(f"Invalid column type '{column_type}' for column '{column_name}'")

    for column_name, description in summarized_input_dict.get("quantitative_data", {}).items():
        if column_name not in column_descriptions_dictionary.keys():
            closest_column_name, column_type = vectordb.get_closest_column_from_description(
                approximate_column_name=column_name,
                users_description=description,
                sample_data_strings=[],
                database_name=DEFAULT_DB_NAME,
                table_name=DEFAULT_TABLE_NAME,
            )

            if closest_column_name not in column_descriptions_dictionary:
                raise ValueError(f"Closest column name '{closest_column_name}' not found in chroma columns collection.")

            if column_type == ColumnType.NUMERICAL:
                numerical_data[closest_column_name] = description

            else:
                logger.warning(
                    f"Warning ! column type '{column_type}' for column '{closest_column_name}' is not numerical as expected."
                )

                if column_type == ColumnType.CATEGORICAL:
                    categorical_data[closest_column_name] = description
                elif column_type == ColumnType.DESCRIPTIVE:
                    descriptive_data[closest_column_name] = description
                elif column_type == ColumnType.IDENTIFIER:
                    identifier_data[closest_column_name] = description
                else:
                    raise ValueError(f"Invalid column type '{column_type}' for column '{closest_column_name}'")

        else:
            # Add the column to the corresponding dictionary
            column_type = vectordb.get_column_type(column_name, DEFAULT_TABLE_NAME, DEFAULT_DB_NAME)
            if column_type == ColumnType.NUMERICAL:
                numerical_data[column_name] = description

            else:
                logger.warning(
                    f"Warning ! column type '{column_type}' for column '{column_name}' is not numerical as expected."
                )

                if column_type == ColumnType.CATEGORICAL:
                    categorical_data[column_name] = description
                elif column_type == ColumnType.DESCRIPTIVE:
                    descriptive_data[column_name] = description                
                elif column_type == ColumnType.IDENTIFIER:
                    identifier_data[column_name] = description
                else:
                    raise ValueError(f"Invalid column type '{column_type}' for column '{column_name}'")

    summazied_user_requested_columns: List[str] = summarized_input_dict.get("user_requested_columns", [])
    get_validated_user_requested_columns(vectordb, summazied_user_requested_columns, DEFAULT_TABLE_NAME, DEFAULT_DB_NAME)

    summarized_input = SummarizedInput(
        summary=summarized_input_intent["summary"],
        numerical_data=numerical_data,
        categorical_data=categorical_data,
        descriptive_data=descriptive_data,
        identifier_data=identifier_data,
        user_requested_columns=summarized_input_dict.get("user_requested_columns", []),
        user_intent=summarized_input_intent["user_intent"],
    )

    return summarized_input


def get_validated_user_requested_columns(
    vectordb_driver: VectorDBDriver, summazied_user_requested_columns: List[str], table_name: str, db_name: str
) -> List[str]:
    """Validates the user requested columns and returns a list of valid columns.

    Args:
        summazied_user_requested_columns (List[str]): The user requested columns.
        table_name (str): The table name.
        db_name (str): The database name.

    Returns:
        List[str]: A list of valid user requested columns.
    """
    if not summazied_user_requested_columns:
        return []
    if len(summazied_user_requested_columns) < 1:
        return []

    ret = []
    # Go through each of the columns
    for column_name in summazied_user_requested_columns:
        # Check if the columns is present in the data
        exists = vectordb_driver.check_if_column_name_exists(column_name, table_name, db_name)

        # If it exists, add it to the list, continue to the next column
        if exists:
            ret.append(column_name)
            continue

        # If the column does not exist, find the closest column name
        closest_column_name, column_type = vectordb_driver.get_closest_column_from_description(
            approximate_column_name=column_name,
            users_description="",
            sample_data_strings=[],
            database_name=db_name,
            table_name=table_name,
        )

        # If the closest column name is not found, print a warning and continue to the next column
        if not closest_column_name:
            logger.warning(f"Closest column name not found for column '{column_name}'")
            continue

        # Now add the closest column name to the list
        ret.append(closest_column_name)

    return ret


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
