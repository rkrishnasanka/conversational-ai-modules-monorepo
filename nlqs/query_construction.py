from typing import Dict, List, Union
from unittest.mock import DEFAULT
import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAI

from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import VectorDBDriver

logger = logging.getLogger(__name__)


def join_fragments(fragments: List[str], joiner: str = "AND") -> str:
    """Joins a list of query fragments into a single query.

    Args:
        fragments (List[str]): A list of query fragments to join.
        joiner (str, optional): The joiner to use between fragments. Defaults to "AND".

    Returns:
        str: The joined query.
    """
    return f" {joiner} ".join(fragments)


def parse_descriptive_numerical_condition(
    column_name: str, 
    descriptive_condition: str, 
    llm: Union[ChatOpenAI, OpenAI, AzureChatOpenAI]
) -> str:
    """Convert descriptive numerical conditions to SQL conditions using LLM.
    
    Args:
        column_name (str): The name of the column
        descriptive_condition (str): Descriptive text like "high CBD content" or "low price"
        llm: The LLM instance to use
        
    Returns:
        str: A numerical condition like "> 15" or "<= 100" or empty string if cannot parse
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a data query assistant. Your task is to convert descriptive numerical conditions into specific SQL numerical conditions.

Given a column name and a descriptive condition, you need to:
1. Determine if this is asking for a numerical comparison
2. If yes, convert it to a proper SQL condition format (>, <, >=, <=, =)
3. Make reasonable assumptions about thresholds based on common sense and typical values for the column

Examples:
- "high [column]" → > [typical high threshold]
- "low [column]" → < [typical low threshold]
- "above average [column]" → > [average value]
- "expensive" for a price column → > [typical expensive threshold]
- "small quantities" → < [typical small value]

Important rules:
1. Only return the operator and number (e.g., > 10, <= 50, = 0)
2. Do NOT use quotes around your response
3. If you cannot determine a reasonable numerical condition, return "UNABLE_TO_PARSE"
4. Make reasonable assumptions about typical value ranges for the column
5. Consider the context of the descriptive term (high/low/above/below/etc.)

Column name: {column_name}
Descriptive condition: {descriptive_condition}

Return only the numerical condition (without quotes) or "UNABLE_TO_PARSE":
        """),
        ("human", f"Column: {column_name}, Condition: {descriptive_condition}")
    ])
    
    try:
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        
        result = chain.invoke({
            "column_name": column_name,
            "descriptive_condition": descriptive_condition
        }).strip()
        
        # Remove any quotes that might be present
        result = result.strip('"').strip("'")
        
        logger.info(f"LLM converted '{descriptive_condition}' for column '{column_name}' to: '{result}'")
        
        # Validate the result format
        if result == "UNABLE_TO_PARSE":
            return ""
        
        # Check if result matches expected pattern (operator + number)
        import re
        if re.match(r'^(>=|<=|>|<|=)\s*\d+(\.\d+)?$', result.replace(" ", "")):
            return result
        else:
            logger.warning(f"LLM returned invalid format: {result}")
            return ""
            
    except Exception as e:
        logger.error(f"Error parsing descriptive condition with LLM: {e}")
        return ""

def construct_quantitaive_search_query_fragments(
    quantitaive_data: Dict[str, str], 
    llm: Union[ChatOpenAI, OpenAI, AzureChatOpenAI, None] = None
) -> List[str]:
    """Creates an SQL query from a dictionary of quantitative data.

    Args:
        quantitaive_data (dict): A dictionary of quantitative data in the form {'column_name': 'condition'}.
        llm: Optional LLM instance for parsing descriptive conditions

    Returns:
        List[str]: List of SQL query fragments.
    """
    if not quantitaive_data:
        return []  # Return an empty list if the dictionary is empty

    query_parts = []
    for column, condition in quantitaive_data.items():

        # Remove the whitespace from the condition
        condition_cleaned = condition.replace(" ", "")

        # Check if this looks like a descriptive condition rather than numerical
        operators = ["<=", ">=", "<", ">", "="]
        has_operator = any(op in condition_cleaned for op in operators)
        
        if not has_operator and llm is not None:
            # This looks like a descriptive condition, try to parse it with LLM
            logger.info(f"Attempting to parse descriptive condition: '{condition}' for column '{column}'")
            parsed_condition = parse_descriptive_numerical_condition(column, condition, llm)
            
            if parsed_condition:
                condition_cleaned = parsed_condition.replace(" ", "")
                logger.info(f"Successfully parsed to: '{parsed_condition}'")
            else:
                logger.warning(f"Could not parse descriptive condition: '{condition}' for column '{column}'. Skipping.")
                continue
        elif not has_operator:
            logger.warning(f"Invalid condition: {condition} for column {column}. No LLM provided for parsing.")
            continue

        # Handle different comparison operators
        if "<=" in condition_cleaned:
            operator = "<="
        elif ">=" in condition_cleaned:
            operator = ">="
        elif "<" in condition_cleaned:
            operator = "<"
        elif ">" in condition_cleaned:
            operator = ">"
        elif "=" in condition_cleaned:
            operator = "="
        else:
            logger.warning(f"Invalid condition: {condition_cleaned}")
            continue

        # Extract the value from the condition
        value = condition_cleaned.replace(operator, "").strip()

        # Validate that value is numeric
        try:
            float(value)  # Test if it's a valid number
        except ValueError:
            logger.warning(f"Non-numeric value in condition: {value}")
            continue

        # Special handling for CBD column to convert mg/g values
        if column == "CBD":
            # Use CAST and REPLACE to handle the mg/g unit conversion in SQLite
            query_part = f"CAST(REPLACE(REPLACE({column}, ' mg/g', ''), ',', '.') AS DECIMAL) {operator} {value}"
        else:
            # Normal numeric comparison for other columns
            query_part = f"{column} {operator} {value}"

        query_parts.append(query_part)

    return query_parts


def construct_categorical_search_query_fragments(categorical_data: Dict[str, str]) -> List[str]:
    """Creates an SQL query from a dictionary of categorical data.

    Args:
        categorical_data (dict): A dictionary of categorical data in the form {'column_name': 'condition'}.

    Returns:
        List[str]: List of SQL query fragments.
    """
    if not categorical_data:
        return []  # Return an empty list if the dictionary is empty

    query_parts = []
    for column, condition in categorical_data.items():
        # Construct the query part
        query_part = f"{column} = '{condition}'"
        query_parts.append(query_part)

    return query_parts


def construct_identifier_search_query_fragments(identifier_data: Dict[str, str]) -> List[str]:
    """Creates an SQL query from a dictionary of identifier data.

    Args:
        identifier_data (dict): A dictionary of identifier data in the form {'column_name': 'condition'}.

    Returns:
        List[str]: List of SQL query fragments.
    """
    if not identifier_data:
        return []  # Return an empty list if the dictionary is empty

    query_parts = []
    for column, condition in identifier_data.items():
        # Check if the condition is numeric or string
        try:
            # Try to convert to int/float - if successful, it's numeric
            float(condition)
            # If numeric, don't use quotes
            query_part = f"{column} = {condition}"
        except ValueError:
            # If not numeric, treat as string and add quotes
            # Also handle case-insensitive matching for location names
            query_part = f"LOWER({column}) = LOWER('{condition}')"
        
        query_parts.append(query_part)
        logger.info(f"Generated identifier query fragment: {query_part}")

    return query_parts


def construct_descriptive_search_query_fragments(
    descriptive_data: Dict[str, str], vectordb_driver: VectorDBDriver
) -> Dict[str, List[str]]:
    """Creates an SQL query from a dictionary of descriptive data.

    Args:
        descriptive_data (Dict[str, str]):  A dictionary of descriptive data in the form {'column_name': 'condition'}.
        vectordb_driver: The vector database driver

    Returns:
        Dict[str, List[str]]: A dictionary of the generated SQL query fragments where the key is the column name.
    """

    results = vectordb_driver.qualitative_dataset_search(
        data=descriptive_data, db_name=DEFAULT_DB_NAME, table_name=DEFAULT_TABLE_NAME
    )

    if not results:
        return {}  # Return an empty dict if no results

    ret = {}

    for column, pk_column_name_value_pairs in results.items():
        query_parts = []
        # Construct a dictionary of primary key column names and values
        temp_storage: Dict[str, List[str]] = {}

        # Store the value_pairs in the temp_storage
        for pk_column_name, value in pk_column_name_value_pairs:
            if pk_column_name not in temp_storage:
                temp_storage[pk_column_name] = []
            temp_storage[pk_column_name].append(value)

        # Construct the query part for each primary key column
        for pk_column_name, values in temp_storage.items():
            values_list = ", ".join(f"{value}" for value in values)
            query_part = f"{pk_column_name} IN ({values_list})"
            query_parts.append(query_part)

        ret[column] = query_parts

    return ret


def construct_final_search_query(where_query_fragments: List[str], table_name: str) -> List[str]:
    """Construct the search query using the fragments (database and table names)

    Args:
        where_query_fragments (List[str]): The where conditions that need to be appended
        table_name (str): The name of the table

    Returns:
        List[str]: The list of queries
    """

    # Construct the final query
    if not where_query_fragments:
        return []

    queries = [f"SELECT * FROM {table_name} WHERE {fragment};" for fragment in where_query_fragments]
    return queries
