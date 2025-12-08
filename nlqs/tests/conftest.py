import os
import random
import sqlite3
from pathlib import Path
from typing import Callable, List
from unittest.mock import Mock, patch
import uuid

import pandas as pd
import psycopg2
import pytest
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver
from utils.llm import get_default_embedding_function


@pytest.fixture(scope="function")
def sqlite_driver():
    sqlite_config = SQLiteConnectionConfig(db_file=Path("test_database.db"), dataset_table_name=DEFAULT_TABLE_NAME)

    db_driver = SQLiteDriver(sqlite_config=sqlite_config)

    return db_driver


@pytest.fixture(scope="function")
def setup_sqlite_database():
    """Setup method to create a test database and driver instance."""
    test_db_file = Path("test_database.db")

    # Create a test database and table
    with sqlite3.connect(test_db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE {DEFAULT_TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
            """
        )
        cursor.execute(f"INSERT INTO {DEFAULT_TABLE_NAME} (name, value) VALUES ('John', 10.5)")
        cursor.execute(f"INSERT INTO {DEFAULT_TABLE_NAME} (name, value) VALUES ('Jane', 20.0)")

        # Create a table for column descriptions
        cursor.execute(
            """
            CREATE TABLE column_descriptions (
                column_name TEXT PRIMARY KEY,
                description TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("id", "Unique identifier"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("name", "Name"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("value", "Value"),
        )

        # Create a table for column types
        cursor.execute(
            """
            CREATE TABLE column_types (
                column_name TEXT PRIMARY KEY,
                column_type TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("id", "numerical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("name", "categorical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("value", "numerical"),
        )

        # Create a table with multiple primary keys
        cursor.execute(
            """
            CREATE TABLE test_table_multiple_pk (
                id1 INTEGER,
                id2 INTEGER,
                name TEXT,
                PRIMARY KEY (id1, id2)
            )
            """
        )

    yield

    # Cleanup method to remove the test database file
    test_db_file.unlink(missing_ok=True)


@pytest.fixture(scope="function")
def pg_config():
    return PostgresConnectionConfig(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database_name="postgres",
        dataset_table_name=DEFAULT_TABLE_NAME,
        uri_column="url",
    )


@pytest.fixture(scope="function")
def setup_postgres_database():
    """Setup method to create a test database and driver instance using product_descriptions.csv data."""
    # Load the product descriptions CSV
    csv_path = Path("./examples/nlqs_demo/product_descriptions.csv")
    products_df = pd.read_csv(csv_path)
    
    with psycopg2.connect(
        host="localhost", port=5432, user="postgres", password="postgres", database="postgres"
    ) as conn:
        cursor = conn.cursor()
        
        # Create table with schema matching the CSV
        cursor.execute(
            f"""
            CREATE TABLE {DEFAULT_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                Location TEXT,
                Room TEXT,
                Product TEXT,
                Category TEXT,
                PackageID TEXT,
                Batch TEXT,
                CBD TEXT,
                THC TEXT,
                CBDA TEXT,
                CBG TEXT,
                CBN TEXT,
                THCA TEXT,
                CustomerRating INTEGER,
                MedicalBenefitsReported TEXT,
                RepeatPurchaseFrequency TEXT,
                URL TEXT,
                Description TEXT
            )
            """
        )
        
        # Insert data from CSV
        for _, row in products_df.iterrows():
            cursor.execute(
                f"""
                INSERT INTO {DEFAULT_TABLE_NAME} 
                (Location, Room, Product, Category, PackageID, Batch, CBD, THC, CBDA, CBG, CBN, THCA, 
                 CustomerRating, MedicalBenefitsReported, RepeatPurchaseFrequency, URL, Description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                tuple(row)
            )
        
        conn.commit()

    yield

    with psycopg2.connect(
        host="localhost", port=5432, user="postgres", password="postgres", database="postgres"
    ) as conn:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE {DEFAULT_TABLE_NAME}")


@pytest.fixture(scope="function")
def postgres_driver(pg_config, setup_postgres_database) -> PostgresDriver:
    return PostgresDriver(pg_config)


@pytest.fixture(scope="function")
def chroma_config():
    return ChromaDBConfig(persist_path=Path("./chroma"), is_local=True)


@pytest.fixture(scope="function")
def vectordb_driver(chroma_config, embedding_function):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)

    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    # Load the column and data description datasets
    column_info_df = pd.read_csv("./nlqs/tests/data/column_descriptions_with_embeddings.tsv", sep="\t")
    data_info_df = pd.read_csv("./nlqs/tests/data/data_descriptions_with_embeddings.tsv", sep="\t")
    table_info_df = pd.read_csv("./nlqs/tests/data/table_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_column_info(
        chroma_config,
        column_info_df,
    )

    VectorDBDriver.populate_nlqs_dataset_info(
        chroma_config,
        data_info_df,
    )

    VectorDBDriver.populate_nlqs_table_info(
        chroma_config,
        table_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config, embedding_function)

    return vectordb_driver


@pytest.fixture(scope="function")
def embedding_function() -> (
    Callable[[str], List[float]]
):

    # Initialize the Azure OpenAI Embedding model
    embedding_model = get_default_embedding_function(use_azure=True)

    # Create an embedding function
    embedding_function = embedding_model.embed_query

    return embedding_function
