from pathlib import Path
from typing import List, Dict, Tuple
import json
from xml.dom.minidom import Document
import logging
from dataclasses import dataclass
import chromadb
from discord_bot.parameters import CHROMA_COLLECTION_NAME, OPENAI_API_KEY, SQLITE_DB_FILE, SQL_TABLE_NAME, LOGGER_FILE
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from nlqs.database.sqlite import fetch_data_from_sqlite
from pydantic.v1 import SecretStr

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to save logs
file_handler = logging.FileHandler(LOGGER_FILE)

# Create a formatter to format the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    quantitative_data: Dict[str, str]
    qualitative_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


# class Query:

#     def __init__(self) -> None:
#         self.column_descriptions, self.numerical_columns, self.categorical_columns = retrieve_descriptions_and_types_from_db()


def get_chroma_collections() -> chromadb.Collection:
    """Gets the chroma collection.

    Returns:
        Chroma: Chroma collection.
    """
    chroma_client: chromadb.ClientAPI = chromadb.PersistentClient()
    collection_name = CHROMA_COLLECTION_NAME
    collections = [col.name for col in chroma_client.list_collections()]
    print(collections)

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists, getting existing collection...")
        chroma_collection = chroma_client.get_collection(collection_name)
    else:
        print("Creating new collection...")
        collection = chroma_client.create_collection(collection_name)

        data = fetch_data_from_sqlite(Path(SQLITE_DB_FILE), SQL_TABLE_NAME)

        data["combined_text"] = data[
            ["Product", "Category", "PackageID", "MedicalBenefitsReported", "Description"]
        ].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        texts = data["combined_text"].tolist()
        data["meta_data"] = data[
            [
                "Category",
                "CustomerRating",
                "RepeatPurchaseFrequency",
                "Location",
                "Room",
                "Batch",
                "CBD",
                "THC",
                "CBDA",
                "CBG",
                "CBN",
                "THCA",
                "URL",
            ]
        ].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        metadata = data["meta_data"].tolist()

        for text, pro, meta in zip(texts, data["Product"], metadata):
            chroma_collection = collection.add(
                documents=text,
                ids=pro,
                metadatas={
                    "product details: 'Category','CustomerRating','PurchaseFrequency','Location', 'Room', 'Batch', 'CBD','THC','CBDA','CBG','CBN','THCA', 'URL' ": meta
                },
            )

        chroma_collection = chroma_client.get_collection(collection_name)
    return chroma_collection


# Initializes the ChatOpenAI LLM model
llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=OPENAI_API_KEY, max_tokens=1000)

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
) -> SummarizedInput:
    """Summarizes the user input and returns the summary, quantitative data, and qualitative data, along with the user requested columns in a JSON format.

    Args:
        user_input (str): The user input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions (dict[str, str]): The column descriptions.
        numerical_columns (list[str]): The numerical columns.
        categorical_columns (list[str]): The categorical columns.

    Returns:
        dict: {
            "summary": str,
            "quantitative_data": {
                "column name": str,
                "column name": str,
                "column name": str,
            },
            "qualitative_data": {
                "column name": str,
                "column name": str,
                "column name": str,
            },
            "user_requested_columns": list,
            "user_intent":str,
        }
    """

    column_descriptions = list(column_descriptions_dictionary.items())

    # Summarize the user input
    instruction = f"""
    You will receive a user input and the chat history. Your task is to:
    1. Analyze the user input and identify key details based on our available data and chat history.
    2. Summarize the input, classifying the data into qualitative and quantitative categories.
    3. Identify Relevant Columns:
        - Determine which columns from the data are needed to provide an answer.
        - Pay close attention to the user's intent and specific mentions of data columns:
        - Are they seeking information about products, medications, treatments, or other relevant categories?
        - If the user is seeking information about a product, provide the URL of the product if available.
        - Look for explicit mentions of column names, synonyms, or phrases indicating the type of information requested. 
        - If the user specifies certain attributes or metrics, consider these as user-requested columns.
    4. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.
    5. Output the result in a JSON format.
    6. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)
    
    The output JSON should have the following structure:

        "summary": "summary of the user input",
        "quantitative_data":
                            " 
                               "column name": "data mentioned in the user input",
                               "column name": "data mentioned in the user input",
                               "column name": "data mentioned in the user input",
                             ",
        "qualitative_data": 
                            " 
                               "column name": "data mentioned in the user input",
                               "column name": "data mentioned in the user input",
                               "column name": "data mentioned in the user input",
                             ",
        "user_requested_columns": "List of columns the user wants data from. If none, leave it as an empty list.",
        "user_intent": "The user's intent. If none, leave it as an empty string.",
    
    The data we have and chat history:
    User input: {user_input}\n\n 
    Data:{column_descriptions}\n\n 
    numerical columns in the data: {numerical_columns}\n\n 
    descriptive columns in the data: {categorical_columns}\n\n 
    Chat history: {chat_history}

    Now, summarize the user input and provide the structured output in JSON format.
    """
    system_prompt = "You are an expert in summarization and expressing key ideas succinctly."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "user_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)

    summarized_input_str = llm_chain.run({"chat_history": chat_history, "user_input": user_input})
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


# Function to perform a similarity search
def similarity_search(collection: chromadb.Collection, user_input: str) -> str:
    """Performs a similarity search on the database and returns the first similar result.

    Args:
        user_input (str): the user input.

    Returns:
        str: the first similar result.

    """
    result = collection.query(query_texts=user_input, n_results=1, include=["documents", "metadatas"])
    if result:
        result_str = str(result)
        result_str = result_str.replace("{", "")
        result_str = result_str.replace("}", '"')
        logger.info(f"Result: {result_str}")
        return result_str
    else:
        logger.info("No similar result found.")
        return ""


# Function to generate a response based on the user input
def generate_query(
    user_input: str,
    summarized_input: SummarizedInput,
    chat_history: List[Tuple[str, str]],
    column_descriptions: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> str:
    """Generates an SQL query based on the user input and chat history.

    Args:
        user_input (str): the user input.
        summarized_input (dict): the summarized input.
        chat_history (list[(str, str)]): the chat history.
        column_descriptions (dict): the column descriptions.
        numerical_columns (list[str]): the numerical columns.
        categorical_columns (list[str]): the categorical columns.

    Returns:
        str: execute_query function executes the SQL query
    """
    quantitative_data = list(summarized_input.quantitative_data.items())
    qualitative_data = list(summarized_input.qualitative_data.items())
    user_requested_columns = summarized_input.user_requested_columns

    instruction = f"""
    Generate an SQLite query based on the user input and other data. For numerical columns, use exact matches. 
    For descriptive columns, use 'LIKE' for partial matches but handle possible spelling mistakes and close matches. 
    insert ORDER BY CustomerRating DESC LIMIT 3 if needed. 
    
    Generate the query according to the user input, chat history, and database schema. 
    Ensure that the query is robust, handles various user input scenarios, and incorporates appropriate conditions.
    Answer just the query without any explanation and code. 

    The data we have:
    numerical columns in the data: {numerical_columns}\n\n 
    descriptive columns in the data: {categorical_columns}\n\n 
    The columns in the database were {', '.join(column_descriptions.keys())}\n\n
    Table name: {SQL_TABLE_NAME}\n\n
    User input: {user_input}\n\n
    quantitative data in the user input: {quantitative_data}\n\n
    qualitative data in the user input: {qualitative_data}\n\n
    user requested columns: {user_requested_columns}\n\n
    Chat history: {chat_history}\n\n

    Generate the SQLite query below:
    """

    system_prompt = (
        "You are an expert in SQL queries. Create robust queries based on the user requirements and database schema."
    )
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "user_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)

    query = llm_chain.run({"chat_history": chat_history, "user_input": user_input}).strip()
    logger.info(f"Query: {query}")

    return query
