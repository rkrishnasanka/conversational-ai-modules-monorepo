from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb

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
from nlqs.nlqs import ChromaDBConfig


def generate_chroma_collection(
    collection_name: str, client, db_driver: Union[SQLiteDriver, PostgresDriver]
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
        print(f"Collection '{collection_name}' does not exists, Creating a collection...")

        # raise ValueError("Chroma collection doesn't exist. Create a chroma collection!!")

        collection = client.create_collection(collection_name)

        data = db_driver.fetch_data_from_database(db_driver.db_config.dataset_table_name)

        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

        if data is None:
            raise ValueError("No data found in the database.")

        primary_key = db_driver.get_primary_key(db_driver.db_config.dataset_table_name)

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


if __name__ == "__main__":
    # SQLite configuration
    # connection_config = SQLiteConnectionConfig(
    #     db_file=Path(SQLITE_DB_FILE), dataset_table_name=SQL_TABLE_NAME, uri_column="URL", output_columns=OUTPUT_COLUMNS
    # )

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

    # ChromaDB configuration
    chroma_config = ChromaDBConfig()  # local chroma

    # remote config
    chroma_config = ChromaDBConfig(CHROMA_COLLECTION_NAME, is_local=False, host="localhost", port=8000)

    chroma_type = chroma_config.is_local
    if chroma_type:
        chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
    else:
        chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)

    # generate_chroma_collection(chroma_config.collection_name, chroma_client, connection_driver)
