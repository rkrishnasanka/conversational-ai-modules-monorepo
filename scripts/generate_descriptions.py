import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from scripts.parameters import (
    CHROMA_COLLECTION_NAME,
    OUTPUT_COLUMNS,
    SQL_TABLE_NAME,
    SQLITE_DB_FILE,
    SUPABASE_DATABASE_NAME,
    SUPABASE_HOST,
    SUPABASE_PASSWORD,
    SUPABASE_PORT,
    SUPABASE_USER,
    URL_COLUMN,
)
from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import OPENAI_API_KEY


# 1. pass the data in the databse
# 2. for each column in the table create a sample data for 5 non empty rows and remove '{|}'
# 3. pass the column name, it's data type and the sample data into the llm to generate desccriptions
# 4. in the instruction for llm, i passed some predifined descriptions to make the llm know how to write descriptions.
# these predifined descriptions will not effect any future changes..
def get_column_descriptions(dataframe: pd.DataFrame) -> Dict[str, str]:
    """
    Generates the descriptions for each columns.

    Args:
        dataframe(pd.Dataframe): data in the database converted into a pandas dataframe.

    Returns:
        descriptions(Dict[str, str]): a dictionary of column descriptions.
    """

    print("Generating column descriptions...")

    # Initialize an empty dictionary to store column descriptions
    descriptions = {}

    for column in dataframe.columns:
        # for each column get data and create sample data using five non empty rows and reomve any specials characters.
        col_data = dataframe[column]
        # get the data type of the column
        col_type = col_data.dtype
        # get the sample data from five non empty rows for the column
        sample_data = dataframe[column].dropna().sample(min(5, len(dataframe[column]))).tolist()
        # mapping column name and the column data
        sample_data_str = ", ".join(map(str, sample_data))
        # removing { } because the llm cannot handle some special characters.
        sample_data_str = re.sub("{|}", "", sample_data_str)

        # Prepare the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                The following is a description of a dataset column:
                Column Name: {column}
                Data Type: {col_type}
                
                Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                
                Sample Data: {sample_data_str}
                
                Use the following format:

                For example:
                    "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                    "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.",
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.",
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches."
                """,
                ),
                ("user", "{user_input}"),
            ]
        )

        llm = ChatOpenAI(
            model="gpt-4",
            api_key=SecretStr(OPENAI_API_KEY),
            temperature=0.0,
            verbose=True,
        )

        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        # Generate the description
        response = chain.invoke(
            {
                "column": column,
                "col_type": col_type,
                "sample_data_str": sample_data_str,
                "user_input": "Please provide a detailed description of each column in the given dataset.",
            }
        )

        # Extract the description from the response
        description = response.strip()

        # Add the description to the dictionary
        descriptions[column] = description

    # Return the dictionary of column descriptions
    return descriptions


def store_descriptions_in_db(
    descriptions: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    db_driver: Union[SQLiteDriver, PostgresDriver],
):
    """
    Stores the generated column descriptions in the database.

    Args:
        descriptions (Dict[str,str]): description of each column along with the column name.
        numerical_columns (List[str]): numerical columns in the database.
        categorical_columns (List[str]): categorical columns in the database
        db_driver (Union[SQLiteDriver, PostgresDriver]): Database driver.
    """

    # Create table for column descriptions
    db_driver.execute_query(
        """
        CREATE TABLE IF NOT EXISTS column_descriptions (
            column_name TEXT PRIMARY KEY,
            description TEXT
        )
        """
    )

    for column, description in descriptions.items():
        db_driver.execute_query(
            f"""
            INSERT OR REPLACE INTO column_descriptions (column_name, description)
            VALUES ({column}, {description})
            """
        )

    # Create table for numerical and categorical columns
    db_driver.execute_query(
        """
        CREATE TABLE IF NOT EXISTS column_types (
            column_name TEXT PRIMARY KEY,
            column_type TEXT
        )
        """
    )

    for column in numerical_columns:
        db_driver.execute_query(
            f"""
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES ({column}, "numerical")
            """
        )

    for column in categorical_columns:
        db_driver.execute_query(
            f"""
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES ({column}, "categorical")
            """
        )


def generate_column_description(df: pd.DataFrame, db_driver: Union[SQLiteDriver, PostgresDriver]):
    """Generates and stores column descriptions in the database.

    Args:
        df (pd.DataFrame): data in the database converted into a pandas dataframe.
        db_driver (Union[SQLiteDriver, PostgresDriver]): Database driver.
    """

    # Get column descriptions
    column_descriptions = get_column_descriptions(dataframe=df)

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Store descriptions and column types in the database
    store_descriptions_in_db(
        descriptions=column_descriptions,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        db_driver=db_driver,
    )

    print(column_descriptions)
    print("Column descriptions and column types stored in the database.")


if __name__ == "__main__":
    # SQLite configuration
    connection_config = SQLiteConnectionConfig(
        db_file=Path(SQLITE_DB_FILE), dataset_table_name=SQL_TABLE_NAME, uri_column="URL", output_columns=OUTPUT_COLUMNS
    )

    # Postgres configuration
    connection_config = PostgresConnectionConfig(
        host=SUPABASE_HOST,
        port=int(SUPABASE_PORT),
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        database_name=SUPABASE_DATABASE_NAME,
        dataset_table_name=SQL_TABLE_NAME,
        uri_column=URL_COLUMN,
    )

    connection_driver = None

    if isinstance(connection_config, SQLiteConnectionConfig):
        connection_driver = SQLiteDriver(connection_config)
    elif isinstance(connection_config, PostgresConnectionConfig):
        connection_driver = PostgresDriver(connection_config)
    elif connection_driver is None:
        raise ValueError("Initialize or enter connection config..")

    connection_driver.connect()

    df = connection_driver.fetch_data_from_database(table_name=connection_config.dataset_table_name)
    generate_column_description(df, connection_driver)
