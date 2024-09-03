from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

import chromadb

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver


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

def execute_query(query: str) -> List[str]:
    """Executes the SQL query and returns the result.

    Args:
        query (str): the SQL query.

    Returns:
        List[str]: the result of the query.
    """
    try:
        connnection = sqlite3.connect("aegion.db")
        cursor = connnection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        connnection.commit()
        # result_str = str(result)
        # logger.info(f"Query executed successfully: {result_str}")
        # result_list = [item[0] for item in result]
        # return result_list if result_list else []
        return result if result else []
    except sqlite3.Error as e:
        error_message = f"Error executing SQL query: {e}"
        print(error_message)
        # logger.error(error_message)
        raise e
        # return []
    
def get_chroma_collection(
    collection_name: str,
    client,
    db_driver: SQLiteDriver,
    primary_key: Optional[str],
) -> chromadb.Collection:
    """Gets the chroma collection.

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

        # numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
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

def qualitaive_search(collection: chromadb.Collection, data: Dict[str, str]) -> List[str]:
    """Performs a similarity search on the database and returns all similar results.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search.
        data (Dict[str, str]): A dictionary of qualitative data to search for.

    Returns:
        Dict[str, List[str]: A dictionary containing the search results.
    """
    results = []

    for column, condition in data.items():
        query_result = collection.query(query_texts=condition, n_results=3, where={"column_name": column})

        if query_result:
            results.extend(query_result["metadatas"])  # Assuming metadatas is a list of dictionaries

    return results if results else []


# quantitaive_data = {'CustomerRating': '= 9'}
quantitaive_data = {}

quantitaive_query = generate_quantitaive_serach_query(quantitaive_data, "new_dataset", "id")
print(quantitaive_query)

quantitative_ids_uncleaned = execute_query(quantitaive_query)

quantitative_ids = [item[0] for item in quantitative_ids_uncleaned]

print(f"quantitative_ids: {quantitative_ids}")



chroma_client = chromadb.PersistentClient()

sqlite_config = SQLiteConnectionConfig(db_file=Path("aegion.db"), dataset_table_name="new_dataset")

connection_driver = SQLiteDriver(sqlite_config)
connection_driver.connect()


collections = get_chroma_collection("aegion", chroma_client, connection_driver, "id")


qualitative_data= {'Product': 'Puffco Peak Pro', 'MedicalBenefitsReported': 'User is asking about the medical benefits.'}

results = qualitaive_search(collections, qualitative_data)

# print(results)

# print("------------------")

qualitative_ids = []
for result in results:
    for item in result:
        qualitative_ids.append(int(item.get("id")))

print(qualitative_ids)

# Find the intersection of quantitative_ids and qualitative_ids
if not quantitative_ids:
    intersection_ids = qualitative_ids
elif not qualitative_ids:
    intersection_ids = quantitative_ids
else:
    intersection_ids = list(set(quantitative_ids) & set(qualitative_ids))

print(intersection_ids)

final_query = f"select * from new_dataset where id in ({','.join(str(id) for id in intersection_ids)})"

final_result = execute_query(final_query)

print(f"final result: {final_result}")


# if __name__ == "__main__":
#     result = execute_query("Show me the products for nausea relief with a rating of grater than 9.")
#     print(result)