import json
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
def get_column_descriptions(dataframe: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    print("Generating column descriptions...")

    # Initialize an empty dictionary to store column descriptions and types
    descriptions = {}

    for column in dataframe.columns:
        # For each column, get data and create sample data from five non-empty rows, removing special characters.
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = dataframe[column].dropna().sample(min(5, len(dataframe[column]))).tolist()
        sample_data_str = ", ".join(map(str, sample_data))
        sample_data_str = re.sub("{|}", "", sample_data_str)

        # Prepare the prompt for LLM
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
                    
                    For example:
                    "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.", "descriptive",
                    "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.", "categorical",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.", "numerical",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches.", "descriptive",

                    The output format should be:
                    {{
                        "column_name": "Product",
                        "description": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                        "column_type": "descriptive"
                    }}
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

        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

        print(f"response: {match}")

        if match:
            json_response = match.group(1)
            json_response = json.loads(json_response)

            column_name = json_response["column_name"]
            description = json_response["description"]
            column_type = json_response["column_type"]

            # Store the description and type in the dictionary
            descriptions[column_name] = {
                "description": description,
                "column_type": column_type,
            }

    # Return the dictionary of column descriptions and types
    return descriptions


def store_descriptions_in_db(
    descriptions: Dict[str, Dict[str, str]],  # Updated to hold descriptions and types
    db_driver: Union[SQLiteDriver, PostgresDriver],
):
    # Create table to store column names, descriptions, and types in one table
    db_driver.execute_query(
        """
        CREATE TABLE IF NOT EXISTS column_metadata (
            column_name TEXT PRIMARY KEY,
            description TEXT,
            column_type TEXT
        )
        """
    )

    for column, metadata in descriptions.items():
        description = metadata["description"]
        column_type = metadata["column_type"]

        # Insert or replace each column's name, description, and type into the table
        db_driver.execute_query(
            """
            INSERT OR REPLACE INTO column_metadata (column_name, description, column_type)
            VALUES (?, ?, ?)
            """,
            (column, description, column_type),
        )

    print("Column metadata (name, description, type) stored in the database.")


def get_chroma_collection(
    collection_name: str,
    client,
    db_driver: Union[SQLiteDriver, PostgresDriver],
    primary_key: Optional[str],
) -> chromadb.Collection:
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
    # Get column descriptions along with types
    column_descriptions = get_column_descriptions(dataframe=df)

    # Store descriptions and column types in the database
    store_descriptions_in_db(
        descriptions=column_descriptions,
        db_driver=db_driver,
    )

    print(column_descriptions)
    print("Column descriptions and column types stored in the database.")
