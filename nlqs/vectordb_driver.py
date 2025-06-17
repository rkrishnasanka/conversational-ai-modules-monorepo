"""
VectorDB Data Schema

Column Info Collection (name: nlqs_column_info)
{
    "document": "description : This column contains the description of the product. It is a text field and can be used for exact or partial matches.",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
        "column_name": "Column name",
        "column_type": "descriptive"
    }
}

Dataset Collection (name: nlqs_descriptive_data)
{
    "document": "Raw data from the dataset",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
        "column_name": "Column name",
        "lookup_key_column_name": "Primary key column name",
        "lookup_key_column_value": "Primary key column value
    }
}

Table Descriptions Collection (name: nlqs_table_descriptions)
{
    "document": "Description of the table",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
    }
}
"""

from __future__ import annotations

import ast
import collections
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypedDict, Union

import chromadb
from chromadb import QueryResult
from chromadb.api import ClientAPI
from chromadb.config import Settings
from click import Option
from pandas import DataFrame
from tqdm import tqdm

DEFAULT_COLUMN_INFO_COLLECTION_NAME = "nlqs_column_info"
DEFAULT_DATASET_COLLECTION_NAME = "nlqs_descriptive_data"
DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME = "nlqs_table_descriptions"

DEFAULT_BATCH_SIZE = 10


class ColumnType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DESCRIPTIVE = "descriptive"
    IDENTIFIER = "identifier"

    @staticmethod
    def from_string(s: str) -> ColumnType:
        try:
            return ColumnType(s)
        except ValueError:
            raise ValueError(f"Invalid column type: {s}. Must be one of {[e.value for e in ColumnType]}")


class ColumnDescriptions(TypedDict):
    column_descriptions: Dict[str, str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    descriptive_columns: List[str]
    identifier_columns: List[str]


class DataCollectionMetadata(TypedDict):
    db_name: str
    table_name: str
    column_name: str
    lookup_key_column_name: str
    lookup_key_column_value: str


class ColumnInfoMetadata(TypedDict):
    db_name: str
    table_name: str
    column_name: str
    column_type: ColumnType


class TableDescriptionMetadata(TypedDict):
    db_name: str
    table_name: str


class ClosestDataResult(TypedDict):
    lookup_key: str
    column_value: Union[str, int]
    data: str


"""
This is a dictionary where the key is the column name and the value is a 
list of tuples where the first element is the lookup key and the second 
element is the column value.
"""
QualitativeSearchResult = Dict[str, List[Tuple[str, str]]]


"""
This is a tuple where the first element is the column name and the second
element is the column value.
"""
EqualsCondition = Tuple[str, str]


@dataclass
class ChromaDBConfig:
    table_description_collection_name: str = DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME
    column_info_collection_name: str = DEFAULT_COLUMN_INFO_COLLECTION_NAME
    dataset_collection_name: str = DEFAULT_DATASET_COLLECTION_NAME
    persist_path: Path = Path("./chroma")
    host: str = "localhost"
    port: int = 8000
    is_local: bool = True
    username: Optional[str] = None
    password: Optional[str] = None


def create_chroma_client(chroma_config: ChromaDBConfig) -> ClientAPI:
    """Create a Chroma client based on the ChromaDBConfig.

    Args:
        chroma_config (ChromaDBConfig): ChromaDB configuration

    Returns:
        ClientAPI: Chroma client
    """
    chroma_type = chroma_config.is_local
    if chroma_type:
        chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
    else:
        chroma_client = chromadb.HttpClient(
            port=chroma_config.port,
            host=chroma_config.host,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{chroma_config.username}:{chroma_config.password}",
            ),
        )

    return chroma_client


def build_and_conditions(
    db_name_filter: Optional[str] = None,
    table_name_filter: Optional[str] = None,
    equal_conditions: Optional[List[EqualsCondition]] = None,
) -> List[Dict[str, Any]]:
    """Build the and conditions for the ChromaDB query

    Args:
        db_name_filter (Optional[str]): Database name filter
        table_name_filter (Optional[str]): table name filter
        equal_conditions (Optional[List[Mapping[str, str]]]): Equal conditions

    Returns:
        List[Mapping[str, Mapping[str, str]]]: List of and conditions
    """
    and_conditions = []

    if db_name_filter:
        and_conditions.append({"db_name": {"$eq": db_name_filter}})
    if table_name_filter:
        and_conditions.append({"table_name": {"$eq": table_name_filter}})

    if equal_conditions:
        for equal_condition in equal_conditions:
            column_name = equal_condition[0]
            column_value = equal_condition[1]
            and_conditions.append({column_name: {"$eq": column_value}})

    return and_conditions


class VectorDBDriver:
    def __init__(self, chroma_config: ChromaDBConfig, embedding_function: Callable[[str], List[float]]):
        """Constructor for the VectorDBDriver

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            embedding_function (Callable[[str], List[float]]): Function to generate embeddings from a string
        """
        self.chroma_config = chroma_config

        self.chroma_client = create_chroma_client(chroma_config)

        self.embedding_function = embedding_function

        if self.check_nlqs_collections_exists():
            print("NLQS collections already exist.")
        else:
            raise ValueError("NLQS collections do not exist. Please initialize the collections.")

    def check_nlqs_collections_exists(
        self,
    ) -> bool:
        """Check if the NLQS collections exist, and return them if they do.

        Args:
            custom_column_data_collection_name (str): Custom column data collection name
            custom_dataset_collection_name (str): Custom dataset collection name

        Returns:
            Tuple[Optional[chromadb.Collection], Optional[chromadb.Collection]]: Tuple of custom column data collection and custom dataset collection
        """

        custom_column_data_collection = self.get_chroma_collection(self.chroma_config.column_info_collection_name)
        custom_dataset_collection = self.get_chroma_collection(self.chroma_config.dataset_collection_name)
        custom_table_description_collection = self.get_chroma_collection(
            self.chroma_config.table_description_collection_name
        )

        return (
            (custom_column_data_collection is not None)
            and (custom_dataset_collection is not None)
            and (custom_table_description_collection is not None)
        )

    def get_chroma_collection(
        self,
        collection_name: str,
    ) -> Optional[chromadb.Collection]:
        """Return the Chroma collection if it exists, otherwise raises an error.

        Args:
            collection_name (str): Collection name

        Raises:
            ValueError: If the collection does not exist

        Returns:
            chromadb.Collection: Chroma collection
        """

        try:
            chroma_collection = self.chroma_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists, getting existing collection...")

        except ValueError as e:
            print(f"Collection '{collection_name}' does not exists")

            return None

        return chroma_collection

    @property
    def column_info_collection(self) -> chromadb.Collection:
        """Get the column info collection.

        Returns:
            chromadb.Collection: Column info collection
        """

        collection = self.get_chroma_collection(self.chroma_config.column_info_collection_name)
        if collection is None:
            raise ValueError(f"Error: Collection '{self.chroma_config.column_info_collection_name}' does not exist.")
        return collection

    @property
    def dataset_collection(self) -> chromadb.Collection:
        """Get the dataset collection.

        Returns:
            chromadb.Collection: Dataset collection
        """

        collection = self.get_chroma_collection(self.chroma_config.dataset_collection_name)
        if collection is None:
            raise ValueError(f"Error: Collection '{self.chroma_config.dataset_collection_name}' does not exist.")
        return collection

    @property
    def table_description_collection(self) -> chromadb.Collection:
        """Get the table description collection.

        Returns:
            chromadb.Collection: Table description collection
        """

        collection = self.get_chroma_collection(self.chroma_config.table_description_collection_name)
        if collection is None:
            raise ValueError(
                f"Error: Collection '{self.chroma_config.table_description_collection_name}' does not exist."
            )
        return collection

    def retrieve_descriptions_and_types_from_db(
        self, db_name_filter: Optional[str] = None, table_name_filter: Optional[str] = None
    ) -> Optional[ColumnDescriptions]:
        """Retrieve the column descriptions and types from the database.

        Args:
            db_name_filter (Optional[str], optional): What the filter of the database name is . Defaults to None.
            table_name_filter (Optional[str], optional): What the filter of the table name is. Defaults to None.

        Raises:
            ValueError: If no metadata is found in the chromadb query result

        Returns:
            Optional[ColumnDescriptions]: Column descriptions dictionary
        """

        # Get the column info collection
        collection = self.column_info_collection

        and_conditions = build_and_conditions(db_name_filter, table_name_filter)

        if len(and_conditions) > 0:
            results = collection.get(where={"$and": and_conditions})
        else:
            results = collection.get()

        if not results["documents"]:
            return None

        ret = ColumnDescriptions(
            column_descriptions={},
            numerical_columns=[],
            categorical_columns=[],
            descriptive_columns=[],
            identifier_columns=[],
        )

        # Go throught the results and populate the dictionaries based on the column type
        if not results["metadatas"]:
            raise ValueError("No metadata found in the chromadb query result.")

        for index, metadata in enumerate(results["metadatas"]):
            column_name = str(metadata["column_name"])
            column_description = str(results["documents"][index])
            column_type = ColumnType(str(metadata["column_type"]))

            ret["column_descriptions"][column_name] = str(column_description)

            if column_type == ColumnType.NUMERICAL:
                ret["numerical_columns"].append(column_name)
            elif column_type == ColumnType.CATEGORICAL:
                ret["categorical_columns"].append(column_name)
            elif column_type == ColumnType.DESCRIPTIVE:
                ret["descriptive_columns"].append(column_name)
            elif column_type == ColumnType.IDENTIFIER:
                ret["identifier_columns"].append(column_name)

        return ret

    def check_if_column_name_exists(self, column_name: str, table_name: str, db_name: str) -> bool:
        """Check if the column name exists in the database.

        Args:
            column_name (str): Column name
            table_name (str): Table name
            db_name (str): Database name

        Returns:
            bool: True if the column name exists, False otherwise
        """

        result = self.column_info_collection.get(
            where={
                "$and": [
                    {"column_name": {"$eq": column_name}},
                    {"table_name": {"$eq": table_name}},
                    {"db_name": {"$eq": db_name}},
                ]
            }
        )

        # Check if the result is empty
        if not result["documents"]:
            return False
        else:
            return True

    def get_closest_column_from_description(
        self,
        approximate_column_name: str,
        users_description: str,
        sample_data_strings: List[str],
        database_name: str,
        table_name: str,
    ) -> Tuple[str, ColumnType]:
        """Get the closest column name from the description provided by the user.

        Args:
            approximate_column_name (str): Approximate column name
            users_description (str): User's description
            sample_data_strings (List[str]): Sample data strings

        Returns:
            str: Closest column name
        """

        # Step 1: Lookup and get the closest column name from the collection using a
        # combination of the user's description and sample data strings
        # Step 2: Replace the approximate column name with the closest column name

        # Step 1: Lookup and get the closest column name from the collection using a
        # combination of the user's description and sample data strings
        column_info_collection = self.column_info_collection
        if not column_info_collection:
            raise ValueError("Column info collection does not exist.")

        # Create a description package
        description_package = f"""
        Closest Column Name: {approximate_column_name}
        User's Description: {users_description}
        Sample Data: {', '.join(sample_data_strings)}
        """

        # TODO: Figure out which embedding to use
        embedding = self.embedding_function(description_package)

        # Use the formatted string to find the closest column
        results = column_info_collection.query(
            query_embeddings=[embedding],
            where={
                "$and": [
                    {"table_name": {"$eq": table_name}},
                    {"db_name": {"$eq": database_name}},
                ]
            },
            n_results=1,
        )

        if not results:
            raise ValueError(f"No matching column found for {approximate_column_name}.")

        metadatas = results["metadatas"]
        if not metadatas:
            raise ValueError(f"No metadata found in chromadb query result for {approximate_column_name}.")

        # TODO: Figure out if this is the correct way to get the closest column name
        closest_column_name = metadatas[0][0]["column_name"]
        column_type = metadatas[0][0]["column_type"]

        if type(closest_column_name) is not str:
            raise ValueError(
                f"Closest column name is not a string for {approximate_column_name}. Extracted Info: {closest_column_name}."
            )

        return closest_column_name, ColumnType(column_type)

    def get_closest_data_from_description(
        self,
        column_name: str,
        description: str,
        database_name: str,
        table_name: str,
    ) -> List[ClosestDataResult]:

        # Do a chromadb query to get the closest data from the dataset collection
        # filter by the database name, table names, and column name

        results = self.dataset_collection.query(
            query_texts=[description],
            where={
                "$and": [
                    {"db_name": {"$eq": database_name}},
                    {"table_name": {"$eq": table_name}},
                    {"column_name": {"$eq": column_name}},
                ]
            },
            n_results=5,
        )

        ret = []
        if not results["documents"] or not results["metadatas"]:
            return ret

        for index, document in enumerate(results["documents"]):
            metadata = results["metadatas"][index][0]
            lookup_key = metadata["lookup_key_column_name"]
            lookup_value = metadata["lookup_key_column_value"]
            if not isinstance(lookup_key, int):
                continue
            if not isinstance(lookup_value, (str, int)):
                continue
            ret.append({"lookup_key": lookup_key, "column_value": lookup_value, "data": document})

        # Return the lookup key, column value and the actual data
        return ret

    def get_column_type(self, column_name: str, table_name: str, db_name: str) -> ColumnType:
        """Get the column type from the database.

        Args:
            column_name (str): Column name
            table_name (str): Table name
            db_name (str): Database name

        Returns:
            ColumnType: Column type

        Raises:
            ValueError: If the column is not found in the database
            ValueError: If no metadata is found in the chromadb query result
        """
        # Query the column info collection to get the column type
        results = self.column_info_collection.get(
            where={
                "$and": [
                    {"column_name": {"$eq": column_name}},
                    {"table_name": {"$eq": table_name}},
                    {"db_name": {"$eq": db_name}},
                ]
            }
        )

        if not results["metadatas"]:
            raise ValueError(f"Column {column_name} not found in the database.")

        metadatas = results["metadatas"]
        if not metadatas:
            raise ValueError(f"No metadata found in chromadb query result for {column_name}.")

        column_type = metadatas[0]["column_type"]

        return ColumnType(column_type)

    def store_column_info_in_db(
        self,
        column_name: str,
        description: str,
        column_type: ColumnType,
    ) -> None:
        """Store the column information in the database.

        Args:
            column_name (str): Column name
            description (str): Column description
            column_type (ColumnType): Column type
        """

        raise NotImplementedError("This method is not implemented yet.")

    def qualitative_table_name_search(self, data: Dict[str, str]) -> List[str]:
        """Performs a similarity search on the database and returns up to 5 similar results.

        Args:
            data (Dict[str, str]): A dictionary of qualitative data to search for.

        Returns:
            List[str]: List of table names
        """
        raise NotImplementedError("This method is not implemented yet.")

    def qualitative_db_name_search(self, data: Dict[str, str]) -> List[str]:
        """Performs a similarity search on the database and returns up to 5 similar results.

        Args:
            data (Dict[str, str]): A dictionary of qualitative data to search for.

        Returns:
            List[str]: List of database names
        """
        raise NotImplementedError("This method is not implemented yet.")

    def qualitative_dataset_search(
        self, data: Dict[str, str], table_name: str, db_name: str
    ) -> QualitativeSearchResult:
        """Performs a similarity search on the database and returns up to 5 similar results per column.

        Args:
            data (Dict[str, str]):  A dictionary of qualitative data to search for.
            table_name (str): Table name
            db_name (str): Database name

        Returns:
            QualitativeSearchResult: Column wise search results which returns the primary key names and values to search for.
        """

        ids_per_column: QualitativeSearchResult = {}

        collection = self.dataset_collection

        # TODO: Get the filter criteria for the table name and db name
        for column_name, condition in data.items():
            embedding = self.embedding_function(str(condition))
            and_conditions = build_and_conditions(
                db_name_filter=db_name,
                table_name_filter=table_name,
                equal_conditions=[("column_name", column_name)],
            )
            query_result: QueryResult = collection.query(
                query_embeddings=[embedding],
                n_results=5,
                where={"$and": and_conditions},
                # where={"column_name": {"$eq": column_name}},
            )

            if query_result["metadatas"]:
                ids_for_column = set()
                for metadata_objects in query_result["metadatas"]:
                    for item in metadata_objects:
                        id_column_value = str(item.get("lookup_key_column_value"))
                        id_column_name = str(item.get("lookup_key_column_name"))
                        if id_column_value is not None:
                            ids_for_column.add((id_column_name, id_column_value))
                ids_per_column[column_name] = list(ids_for_column)

        return ids_per_column

    @staticmethod
    def initialize_nlqs_vectordb(
        chroma_config: ChromaDBConfig,
    ) -> None:
        """Initialize the NLQS VectorDB collections.

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            column_info_collection_name (str): Column info collection name
            dataset_collection_name (str): Dataset collection name
        """

        # Create a new driver instance
        chroma_client = create_chroma_client(chroma_config)

        # Create the NLQS collections
        try:
            chroma_client.create_collection(chroma_config.column_info_collection_name)
        except Exception as e:
            print(f"Error creating NLQS collections: {e}")

        try:
            chroma_client.create_collection(chroma_config.dataset_collection_name)
        except Exception as e:
            print(f"Error creating NLQS collections: {e}")

        try:
            chroma_client.create_collection(chroma_config.table_description_collection_name)
        except Exception as e:
            print(f"Error creating NLQS collections: {e}")

    @staticmethod
    def purge_nlqs_vectordb(
        chroma_config: ChromaDBConfig,
    ) -> None:
        """Purge the NLQS VectorDB collections

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
        """

        chroma_client = create_chroma_client(chroma_config)

        # Get Collection names
        collections = [col.name for col in chroma_client.list_collections()]
        try:
            if chroma_config.column_info_collection_name in collections:
                chroma_client.delete_collection(chroma_config.column_info_collection_name)
            if chroma_config.dataset_collection_name in collections:
                chroma_client.delete_collection(chroma_config.dataset_collection_name)
            if chroma_config.table_description_collection_name in collections:
                chroma_client.delete_collection(chroma_config.table_description_collection_name)
        except Exception as e:
            print(f"Error purging NLQS collections: {e}")

    @staticmethod
    def populate_nlqs_dataset_info(
        chroma_config: ChromaDBConfig,
        dataset_info_df: DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Populate the NLQS VectorDB dataset collection

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            dataset_info_df (DataFrame): Dataset information
            batch_size (int, optional): In what quantities should they be populated. Defaults to DEFAULT_BATCH_SIZE.

        Raises:
            ValueError: If the column is not found in the DataFrame
        """

        # Test to ensure all the columns are present in the table_info_df
        colums_to_check = [
            "description",
            "db_name",
            "table_name",
            "column_name",
            "lookup_key_column_name",
            "lookup_key_column_value",
            "embedding",
        ]

        for column in colums_to_check:
            if column not in dataset_info_df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame. Cannot initialize the data")

        # Create a chroma client
        chroma_client = create_chroma_client(chroma_config)

        # Create the collection if it does not exist
        collection = chroma_client.get_collection(chroma_config.dataset_collection_name)

        print("Populating the dataset collection...")

        # Populate the dataset collection
        ids = [str(i + 1) for i in range(len(dataset_info_df))]

        for index in tqdm(range(0, len(dataset_info_df), batch_size)):
            documents = dataset_info_df["description"].astype(str).tolist()[index : index + batch_size]
            db_names = dataset_info_df["db_name"].astype(str).tolist()[index : index + batch_size]
            table_names = dataset_info_df["table_name"].astype(str).tolist()[index : index + batch_size]
            column_names = dataset_info_df["column_name"].astype(str).tolist()[index : index + batch_size]
            lookup_key_column_names = (
                dataset_info_df["lookup_key_column_name"].astype(str).tolist()[index : index + batch_size]
            )
            lookup_key_column_values = (
                dataset_info_df["lookup_key_column_value"].astype(str).tolist()[index : index + batch_size]
            )
            embeddings = dataset_info_df["embedding"].tolist()[index : index + batch_size]

            # Convert each string in the embeddings list to a list of floats
            embeddings = [ast.literal_eval(embedding) for embedding in embeddings]

            # Create metadata objects
            metadatas = []
            for i in range(len(documents)):
                metadatas.append(
                    {
                        "db_name": db_names[i],
                        "table_name": table_names[i],
                        "column_name": column_names[i],
                        "lookup_key_column_name": lookup_key_column_names[i],
                        "lookup_key_column_value": lookup_key_column_values[i],
                    }
                )

            # Populate the dataset collection
            collection.add(
                ids=ids[index : index + batch_size],
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # Print out the final count
        print(f"Number of items added to the collection: {collection.count()}")

    @staticmethod
    def populate_nlqs_table_info(
        chroma_config: ChromaDBConfig,
        table_info_df: DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Populate the NLQS VectorDB table description collection

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            table_info_df (DataFrame): Table information
            batch_size (int, optional): In what quantities should they be populated. Defaults to DEFAULT_BATCH_SIZE.

        Raises:
            ValueError: If the column is not found in the DataFrame
        """
        # Test to ensure all the columns are present in the table_info_df
        colums_to_check = ["description", "db_name", "table_name", "embedding"]

        for column in colums_to_check:
            if column not in table_info_df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame. Cannot initialize the data")

        # Create a chroma client
        chroma_client = create_chroma_client(chroma_config)

        # Create the collection if it does not exist
        collection = chroma_client.get_collection(chroma_config.table_description_collection_name)

        print("Populating the table description collection...")

        # Populate the table description collection
        ids = [str(i + 1) for i in range(len(table_info_df))]

        for index in tqdm(range(0, len(table_info_df), batch_size)):
            documents = table_info_df["description"].astype(str).tolist()[index : index + batch_size]
            db_names = table_info_df["db_name"].astype(str).tolist()[index : index + batch_size]
            table_names = table_info_df["table_name"].astype(str).tolist()[index : index + batch_size]
            embeddings = table_info_df["embedding"].tolist()[index : index + batch_size]

            # Convert each string in the embeddings list to a list of floats
            embeddings = [ast.literal_eval(embedding) for embedding in embeddings]

            # Create metadata objects
            metadatas = []
            for i in range(len(documents)):
                metadatas.append({"db_name": db_names[i], "table_name": table_names[i]})

            # Populate the table description collection
            collection.add(
                ids=ids[index : index + batch_size],
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # Print out the final count
        print(f"Number of items added to the collection: {collection.count()}")

    @staticmethod
    def populate_nlqs_column_info(
        chroma_config: ChromaDBConfig,
        column_info_df: DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Populate the NLQS VectorDB column info collection

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            column_info_df (DataFrame): Column information
            batch_size (int, optional): In what quantities should they be populated. Defaults to DEFAULT_BATCH_SIZE.

        Raises:
            ValueError: If the column is not found in the DataFrame
        """
        # Test to ensure all the columns are present in the column_info_df

        colums_to_check = ["description", "db_name", "table_name", "column_name", "column_type", "embedding"]

        for column in colums_to_check:
            if column not in column_info_df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame. Cannot initialize the data")

        chroma_client = create_chroma_client(chroma_config)

        # Create the collection if it does not exist
        collection = chroma_client.get_collection(chroma_config.column_info_collection_name)

        print("Populating the column info collection...")

        # Ids of the rows
        ids = [str(i + 1) for i in range(len(column_info_df))]

        for index in tqdm(range(0, len(column_info_df), batch_size)):
            descriptions = column_info_df["description"].astype(str).tolist()[index : index + batch_size]
            db_names = column_info_df["db_name"].astype(str).tolist()[index : index + batch_size]
            table_names = column_info_df["table_name"].astype(str).tolist()[index : index + batch_size]
            column_names = column_info_df["column_name"].astype(str).tolist()[index : index + batch_size]
            column_types = column_info_df["column_type"].astype(str).tolist()[index : index + batch_size]
            embeddings = column_info_df["embedding"].tolist()[index : index + batch_size]

            # Convert each string in the embeddings list to a list of floats
            embeddings = [ast.literal_eval(embedding) for embedding in embeddings]

            # Create metadata objects
            metadatas = []
            for i in range(len(descriptions)):
                metadatas.append(
                    {
                        "db_name": db_names[i],
                        "table_name": table_names[i],
                        "column_name": column_names[i],
                        "column_type": column_types[i],
                    }
                )

            # Populate the column info collection
            collection.add(
                ids=ids[index : index + batch_size],
                documents=descriptions,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # Print out the final count
        print(f"Number of items added to the collection: {collection.count()}")
