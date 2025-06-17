# """
#     Generates benchmarks against the testcases.csv file and stores the results in benchmark_results.csv.
# """

# import csv
# import os
# import re
# from pathlib import Path

# import chromadb
# from langchain_openai import ChatOpenAI
# from pydantic.v1 import SecretStr

# from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
# from nlqs.nlqs import ChromaDBConfig, NLQSResult
# from nlqs.parameters import OPENAI_API_KEY
# from nlqs.query_construction import (
#     construct_quantitaive_search_query_fragments,
# )
# from nlqs.summarization import summarize

# # ChromaDB configuration
# chroma_config = ChromaDBConfig()

# # SQLite configuration
# sqlite_config = SQLiteConnectionConfig(
#     db_file=Path("aegion.db"), dataset_table_name="new_dataset", uri_column="URL", output_columns=[]
# )
# driver = SQLiteDriver(sqlite_config)

# driver.connect()

# primary_key = driver.get_primary_key(driver.db_config.dataset_table_name)

# # CSV file paths
# TEST_CASES_FILE = "test_cases.csv"
# BENCHMARK_RESULTS_FILE = "benchmark_results.csv"

# chroma_client = chromadb.PersistentClient()
# chroma_collection = get_chroma_collection(chroma_config.collection_name, chroma_client, driver, primary_key)

# llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", api_key=SecretStr(OPENAI_API_KEY), max_tokens=1000)


# # Main chat function
# def chat_benchmark(user_input, chat_history):
#     if not user_input:
#         return chat_history, "", []

#     column_descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

#     user_input = re.sub(r"{|}", "", user_input)
#     summarized_input = summarize(
#         user_input, chat_history, column_descriptions, numerical_columns, categorical_columns, llm
#     )

#     if not summarized_input:
#         summarized_input = summarize(
#             user_input, chat_history, column_descriptions, numerical_columns, categorical_columns, llm
#         )

#     if not summarized_input:
#         response = "unable to summarize the data."
#         log_data = [
#             user_input,
#             "none",
#             "none",
#             "none",
#             "none",
#             "none",
#             "",  # Placeholder for query
#             "",  # Placeholder for query result
#             "",  # Placeholder for similarity search result
#             "",  # Placeholder for response
#         ]
#         return chat_history, response, log_data

#     intent = summarized_input.user_intent
#     log_data = [
#         user_input,
#         summarized_input.summary,
#         summarized_input.quantitative_data,
#         summarized_input.qualitative_data,
#         summarized_input.user_requested_columns,
#         intent,
#         "",  # Placeholder for quantitative ids
#         "",  # Placeholder for qualitative ids
#         "",  # Placeholder for type of outputs
#         "",  # Placeholder for intersection ids
#         "",  # Placeholder for response
#     ]

#     if intent == "sql_injection":
#         response = "Sorry, I cannot process this request."
#         chat_history.append((user_input, response))
#         log_data[9] = response
#         return chat_history, response, log_data

#     if summarized_input.user_requested_columns:
#         quantitaive_data = summarized_input.quantitative_data
#         qualitative_data = summarized_input.qualitative_data

#         quantitaive_query = construct_quantitaive_search_query_fragments(
#             quantitaive_data, driver.db_config.dataset_table_name, primary_key
#         )
#         quantitative_ids_uncleaned = driver.execute_query(quantitaive_query)

#         quantitative_ids = [item[0] for item in quantitative_ids_uncleaned]
#         log_data[6] = quantitative_ids

#         qualitative_ids = qualitative_search(chroma_collection, qualitative_data, primary_key)
#         log_data[7] = qualitative_ids

#         # Find the intersection of quantitative_ids and qualitative_ids
#         if not quantitative_ids or not qualitative_ids:
#             intersection_ids = quantitative_ids or qualitative_ids
#             log_data[8] = "Intersection data found."
#         else:
#             intersection_ids = list(set(quantitative_ids) & set(qualitative_ids))
#             log_data[8] = "Intersection data found. Exact answer retrieved."

#         # Ensure intersection_ids is set to qualitative_ids if it's empty
#         if not intersection_ids:
#             intersection_ids = qualitative_ids
#             log_data[8] = "Union data found. A similar answer retrieved."

#         log_data[9] = intersection_ids

#         # Initial query to retrieve all columns based on the intersection IDs
#         final_query = f"SELECT * FROM {driver.db_config.dataset_table_name} WHERE {primary_key} IN ({','.join(str(id) for id in intersection_ids)})"

#         # Get the columns in the order they appear in the database
#         columns_database = driver.get_database_columns(driver.db_config.dataset_table_name)

#         # If output_columns is specified, modify the query to select only those columns
#         if driver.db_config.output_columns:
#             final_query = f"SELECT {','.join(col for col in driver.db_config.output_columns)} FROM {driver.db_config.dataset_table_name} WHERE {primary_key} IN ({','.join(str(id) for id in intersection_ids)})"
#             data_retreived = driver.execute_query(final_query)
#             # Since we now have a subset of columns, use output_columns directly
#             columns_to_use = driver.db_config.output_columns
#         else:
#             # Execute the query to retrieve the data with all columns
#             data_retreived = driver.execute_query(final_query)
#             columns_to_use = columns_database

#         # Initialize lists to hold records and URIs
#         records = []
#         uris = []

#         # Process the retrieved data
#         for row in data_retreived:
#             record = dict(zip(columns_to_use, row))
#             if driver.db_config.uri_column in record:
#                 uris.append(str(record[driver.db_config.uri_column]))
#                 del record[driver.db_config.uri_column]  # Remove the URI column data from the record
#             records.append(record)

#         # Create the result object
#         response = NLQSResult(records=records, uris=uris)

#         response = re.sub(r"{|}", "", str(response))
#     else:
#         response = "Sorry, I cannot process this request."

#     log_data[10] = response
#     chat_history.append((user_input, response))
#     return chat_history, log_data[9], log_data


# # Function to log data to CSV
# def log_to_csv(log_data, file_path=BENCHMARK_RESULTS_FILE):
#     file_exists = os.path.isfile(file_path)
#     with open(file_path, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(
#                 [
#                     "user_input",
#                     "summary",
#                     "quantitative_data",
#                     "qualitative_data",
#                     "user_requested_columns",
#                     "intent",
#                     "qualitative_ids",
#                     "quantitative_ids",
#                     "type_of_ouptut",
#                     "intersection_ids",
#                     "response",
#                 ]
#             )
#         writer.writerow(log_data)


# # Run benchmark
# def run_benchmark(test_cases_file=TEST_CASES_FILE, results_file=BENCHMARK_RESULTS_FILE):
#     with open(test_cases_file, mode="r") as file:
#         reader = csv.DictReader(file)
#         chat_history = []
#         for row in reader:
#             interaction_type = row["interaction_type"]
#             user_input = row["user_input"]
#             chat_history, response, log_data = chat_benchmark(user_input, chat_history)
#             log_to_csv(log_data, results_file)
#             print(f"Interaction Type: {interaction_type}, User Input: {user_input}, Response: {response}")


# if __name__ == "__main__":
#     run_benchmark()
