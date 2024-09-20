import re
from typing import Dict, List, Optional, Union

import chromadb
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver
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


def get_chroma_collection(
    collection_name: str,
    client,
    db_driver: Union[SQLiteDriver, PostgresDriver],
    primary_key: Optional[str],
) -> chromadb.Collection:
    """Gets the chroma collection.

    Args:
        collection_name (str): Name of the collection.
        client (chromadb.Client): Chroma client.
        db_driver (Union[SQLiteDriver, PostgresDriver]): Database driver.
        primary_key (Optional[str]): Primary key column name.

    Returns:
        Chroma: Chroma collection.
    """

    collections = [col.name for col in client.list_collections()]

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists, getting existing collection...")
        chroma_collection = client.get_collection(collection_name)
    else:
        print(f"Collection '{collection_name}' does not exists, Creating new collection...")
        collection = client.create_collection(collection_name)

        data = db_driver.fetch_data_from_database(db_driver.db_config.dataset_table_name)

        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

        if data is None:
            raise ValueError("No data found in the database.")

        if not primary_key:
            primary_key = data.columns[0]

        for index, row in data.iterrows():
            # Extract the primary key value
            pri_key = str(row[primary_key])

            for column in categorical_columns:
                # Extract the text for the current column and row
                text = [str(row[column])]

                # Create the ID for the current column and row
                id = f"{column}_{pri_key}"

                print(f"id: {id}")

                # Create the metadata dictionary
                meta = {
                    "id": pri_key,
                    "table_name": db_driver.db_config.dataset_table_name,
                    "column_name": column,
                }

                # Add the data to the Chroma collection
                chroma_collection = collection.add(
                    documents=text,
                    ids=id,
                    metadatas=meta,
                )

        chroma_collection = client.get_collection(collection_name)
    return chroma_collection


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
