import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd

from nlqs.database.abstract_driver import AbstractDriver

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class SQLiteConnectionConfig:
    db_file: Path
    dataset_table_name: str


class SQLiteDriver(AbstractDriver):
    def __init__(self, sqlite_config: SQLiteConnectionConfig):
        self.db_config = sqlite_config
        self._db_connection = None
        self.cursor = None

    def connect(self):
        try:
            self._db_connection = sqlite3.connect(self.db_config.db_file)
            self.cursor = self._db_connection.cursor()
            logger.info("Connected to SQLite database.")
            print(f"Connected to SQLite database.")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            # raise e
            print(f"Error connecting to database: {e}")
            raise e

    def disconnect(self):
        if self._db_connection:
            self._db_connection.close()
            logger.info("Disconnected from SQLite database.")

    def execute_query(self, query: str) -> str:
        """Executes the SQL query and returns the result.

        Args:
            query (str): the SQL query.

        Returns:
            str: the result of the query.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            self._db_connection.commit()
            result_str = str(result)
            logger.info(f"Query executed successfully: {result_str}")
            return result_str if result else "No results found."
        except sqlite3.Error as e:
            error_message = f"Error executing SQL query: {e}"
            logger.error(error_message)
            return error_message

    def retrieve_descriptions_and_types_from_db(self) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Retrieves descriptions and types from the SQLite database.

        Args:
            db_file (SQLite database, optional): SQLite database to store all the tables. Defaults to SQLITE_DB_FILE.

        Returns:
            Tuple[List[str], List[str], List[str]]: Return descriptions, numerical_columns, categorial_columns
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            # Retrieve descriptions
            self.cursor.execute("SELECT column_name, description FROM column_descriptions")
            description_rows = self.cursor.fetchall()
            descriptions = {row[0]: row[1] for row in description_rows}

            # Retrieve column types
            self.cursor.execute("SELECT column_name, column_type FROM column_types")
            type_rows = self.cursor.fetchall()
            numerical_columns = [row[0] for row in type_rows if row[1] == "numerical"]
            categorical_columns = [row[0] for row in type_rows if row[1] == "categorical"]

            return descriptions, numerical_columns, categorical_columns
        except sqlite3.Error as e:
            logger.error(f"Error retrieving descriptions and types: {e}")
            return {}, [], []
        

    def validate_query(self, query: str) -> bool:
        """Validates the generated SQL query against the database schema and returns True if valid, False otherwise.

        Args:
            query (str): sql query to be validated.
            db_file (str, optional): sqlite database file to be used. Defaults to SQLITE_DB_FILE.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        logger.info(f"Validating query: {query}")
        print(f"Validating query: {query}")

        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
            if not match:
                logger.error("Table name not found in the query.")
                print("Table name not found in the query.")
                return False
            table_name = match.group(1).strip()

            column_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
            if not column_match:
                logger.error("Column names not found in the query.")
                print("Column names not found in the query.")
                return False
            columns = column_match.group(1).strip().split(",")

            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not self.cursor.fetchone():
                logger.error(f"Table '{table_name}' not found in the database.")
                print(f"Table '{table_name}' not found in the database.")
                return False

            self.cursor.execute(f"PRAGMA table_info({table_name})")
            table_columns = [column[1] for column in self.cursor.fetchall()]
            for column in columns:
                column = column.strip()
                if column not in table_columns and column != "*":
                    logger.error(f"Column '{column}' not found in table '{table_name}'.")
                    print(f"Column '{column}' not found in table '{table_name}'.")
                    return False

            logger.info("Query validated successfully.")
            print("Query validated successfully.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error validating query: {e}")
            print(f"Error validating query: {e}")
            return False
        
    def check_table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in a SQLite database.

        Args:
            db_file (str): The path to the SQLite database file.
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        
        try:
            conn = self._db_connection
            # cursor = conn.cursor()
            self.cursor.execute(
                """
                SELECT name FROM sqlite_master WHERE type='table' AND name=?
                """,
                (table_name,),
            )
            result = self.cursor.fetchone()
            return bool(result)  # True if result is not None, False otherwise
        except sqlite3.Error as e:
            print(f"Error checking table existence: {e}")
            return False

    def fetch_data_from_database(self, table_name: str) -> pd.DataFrame:
        """Fetch data from a SQLite database table.

        Args:
            db_file (Path): Path to the SQLite database file.
            table_name (str): Name of the table to fetch data from.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the table, or Null dataframe if an error occurred.
        """
        try:
            if not self.check_table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            conn = self._db_connection
            query = f"SELECT * FROM {table_name}"
            if conn is None:
                raise ValueError("Database connection not established.")
            df = pd.read_sql_query(query, conn)
            conn.close()
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

        return df


    @property
    def db_connection(self) -> sqlite3.Connection:
        if self._db_connection is None:
            raise ValueError("Database connection not established.")
        
        return self._db_connection