from nlqs.database.postgres import PostgresDriver, PostgresConnectionConfig
from nlqs.database.sqlite import SQLiteDriver, SQLiteConnectionConfig 
import re
from typing import Dict, List, Tuple, Union
import chromadb
from nlqs.description_generator import generate_column_description


from nlqs.database.sqlite import SQLiteDriver
from nlqs.query import (
    generate_query,
    get_chroma_collection,
    similarity_search,
    summarize,
)
from dataclasses import dataclass
from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from nlqs.parameters import OPENAI_API_KEY


@dataclass
class ChromaDBConfig:
    collection_name: str
    persist_path: Path
    is_local: bool = True
    

class NLQS:

    def __init__(self, connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig], chroma_config: ChromaDBConfig) -> None:
        # TODO - Figure out what the constructor parameters are
        if isinstance(connection_config, SQLiteConnectionConfig):
            self.connection_driver = SQLiteDriver(connection_config)
        elif isinstance(connection_config, PostgresConnectionConfig):
            self.connection_driver = PostgresDriver(connection_config)
    
        else:
            raise ValueError("Invalid connection configuration")

        # Initialize the connection to the database
        self.connection_driver.connect()

        # Create the chroma client
        chroma_client  = chromadb.PersistentClient()
        self.chroma_collection = get_chroma_collection(
            chroma_client=chroma_client, 
            collection_name=chroma_config.collection_name,
            db_driver=self.connection_driver,
            dataset_table_name=connection_config.dataset_table_name
        )

        # Create the llm object
        # Initializes the ChatOpenAI LLM model
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", api_key=SecretStr(OPENAI_API_KEY), max_tokens=1000)



        # TODO - Figure out if we need to create introspection table, and create
        pass

    def _create_introspection_table(self):
        driver = self.connection_driver
        
        # Step 1
        column_descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

        if column_descriptions == {}:
            # Step 2
            generate_column_description(
                df=self.connection_driver.fetch_data_from_database(
                    table_name=self.connection_driver.db_config.dataset_table_name),
                db_driver=self.connection_driver
            )
            column_descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

        return column_descriptions, numerical_columns, categorical_columns

    # Step 4
    def execute_nlqs_workflow(self,
        user_input: str,
        chat_history: List[Tuple[str, str]],
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """This function is where the whole interaction happens.
        It takes the user input and chat history as input and returns the response if the user's intent is either phatic_communication, profanity or sql_injection.
        Else it returns the query result or search similarity result and the updated chat history.

        Args:
            user_input (str): The user's input.
            chat_history (list[(str, str)]): The chat history.

        Returns:
            Tuple[str,List[Tuple[str, str]]]: The response and the updated chat history.
        """

        # Overview
        # Step 1 - retrieve descriptions and types from db. check if its empty. if not return the data. 
        # Step 2 - else if the retrived data was empty then generate new columns descriptions.
        # Step 3 - next get the chroma collection
        # Step 4 - pass all the retrieved data to the main_workflow method
        # Step 5 - check if the user input is empty if true retun none
        # Step 6 - Else remove the paranthesis from the user input.
        # Step 7 - generate a summary for the user input the required format.
        # Step 8 - check if the summary is empty. if true retry the generation of the summary, you can do this until five times 
        # (the above step is because we were getting errors while converting the generted summary to the json format.)
        # Step 9 - generate an sql query.
        # Step 10 - validate the generated query.
        # Step 11 - check if the query result is empty. if true then do a similarity search and retrieve the relevent info and return it.
        # Step 12 - else return the query result.

        # Step 0 - Create the pre-requisite objects

        # Database Connection
        driver = self.connection_driver
        # Chroma Collection
        chroma_collections = self.chroma_collection

        column_descriptions, numerical_columns, categorical_columns = self._create_introspection_table()

        # TODO - Figure out where you want to get them from
        column_descriptions_dict: Dict[str, str] = column_descriptions
        numerical_columns_list: List[str] = numerical_columns
        categorical_columns_list: List[str] = categorical_columns
        collections=chroma_collections


        # Step 5
        if not user_input.strip():
            response = ""

        # Step 6
        user_input = re.sub(r"{|}", "", user_input)

        # Step 7
        summarized_input = summarize(
            user_input=user_input, 
            chat_history=chat_history, 
            column_descriptions_dictionary=column_descriptions_dict, 
            numerical_columns=numerical_columns_list, 
            categorical_columns=categorical_columns_list,
            llm=self.llm
        )

        count = 0
        print(f"summarized_input: {summarized_input}")
        while not summarized_input.summary and count < 5:
            summarized_input = summarize(
                user_input=user_input, 
                chat_history=chat_history, 
                column_descriptions_dictionary=column_descriptions_dict, 
                numerical_columns=numerical_columns_list, 
                categorical_columns=categorical_columns_list, 
                llm=self.llm
            )
            count += 1
            if count == 5:
                response = "Summarization failed. Please try again."
                break

        intent = summarized_input.user_intent

        if intent == "sql_injection" or intent == "profanity":
            response = ""

        else:
            if summarized_input.user_requested_columns:

                # Step 9
                genenerted_query = generate_query(
                    user_input=user_input,
                    summarized_input=summarized_input,
                    chat_history=chat_history,
                    column_descriptions=column_descriptions_dict,
                    numerical_columns=numerical_columns_list,
                    categorical_columns=categorical_columns_list,
                    llm=self.llm,
                    dataset_table_name=driver.db_config.dataset_table_name
                )

                # Step 10
                if driver.validate_query(genenerted_query):
                    query_result = driver.execute_query(genenerted_query)

                    # Step 11
                    if query_result == "No results found.":
                        query_result = similarity_search(collections, user_input)
                    response = query_result
                else:
                    response = "error while generating query. Please try again."
            else:
                response = ""

        chat_history.append((user_input, response))
        return response, chat_history
        