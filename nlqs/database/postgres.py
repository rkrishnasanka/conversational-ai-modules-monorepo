import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2 import sql

from nlqs.database.abstract_driver import AbstractDriver

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class PostgresConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    database_name: str
    dataset_table_name: str
    uri_column: Optional[str] = None
    output_columns: Optional[List[str]] = None


class PostgresDriver(AbstractDriver):
    def __init__(self, pg_config: PostgresConnectionConfig):
        """

        Example:
        postgres_config = PostgresConnectionConfig(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database_name="aegion",
            dataset_table_name="new_dataset"
        )

        """
        self.db_config = pg_config
        self._db_connection = None
        self.cursor = None

    def connect(self):
        try:
            self._db_connection = psycopg2.connect(
                dbname=self.db_config.database_name,
                user=self.db_config.user,
                password=self.db_config.password,
                host=self.db_config.host,
                port=self.db_config.port,
            )
            self.cursor = self._db_connection.cursor()
            logger.info("Connected to PostgreSQL database.")
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        if self._db_connection:
            self._db_connection.close()
            logger.info("Disconnected from PostgreSQL database.")

    def execute_query(self, query: str) -> Optional[List[Tuple[Any]]]:
        """Executes the SQL query and returns the result.

        Args:
            query (str): the SQL query.

        Returns:
            Optional[List[Tuple]]: the result of the query as a list of tuples, or None if no results.
        """
        print(f"Executing query: {query}")

        if not query.strip():
            return None

        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            logger.info(f"Executing query: {query}")

            self.cursor.execute(query)

            # Only fetch results if the query is a SELECT statement
            if query.lower().startswith("select"):
                result = self.cursor.fetchall()
                self._db_connection.commit()
                logger.info(f"Query executed successfully: {result}")
                return result  # Return the full result here
            else:
                self._db_connection.commit()
                logger.info(f"Query executed successfully.")
                return None  # Return None for non-SELECT queries
        except psycopg2.Error as e:
            error_message = f"Error executing SQL query: {e}"
            print(error_message)
            logger.error(error_message)
            raise e

    def retrieve_descriptions_and_types_from_db(self) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
        """Retrieves descriptions and types from the PostgreSQL database.

        Args:
            db_file (PostgreSQL database, optional): PostgreSQL database to store all the tables. Defaults to POSTGRES_DB_CONFIG.

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
            self.cursor.execute("SELECT column_name, column_type FROM column_metadata")
            type_rows = self.cursor.fetchall()
            numerical_columns = [row[0] for row in type_rows if row[1] == "numerical"]
            categorical_columns = [row[0] for row in type_rows if row[1] == "categorical"]
            descriptive_columns = [row[0] for row in type_rows if row[1] == "descriptive"]

            return descriptions, numerical_columns, categorical_columns, descriptive_columns
        except psycopg2.Error as e:
            logger.error(f"Error retrieving descriptions and types: {e}")
            # roll back the failed sql query
            self._db_connection.rollback()
            return {}, [], [], []

    def get_database_columns(self, table_name: str) -> List[str]:
        """Returns the columns in the specified table in the order they appear in the database.

        Args:
            table_name (str): The name of the table from which to retrieve columns.

        Raises:
            ValueError: Error retrieving columns.

        Returns:
            List[str]: The columns in the database table in order.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            self.cursor.execute(
                sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"), [table_name]
            )
            columns_info = self.cursor.fetchall()
            columns_in_database = [column[0] for column in columns_info]  # Extract column names
            return columns_in_database
        except Exception as e:  # Catch the Exception here
            logger.error(f"Error retrieving columns: {e}")
            return []

    def validate_query(self, query: str) -> bool:
        """Validates the generated SQL query against the database schema and returns True if valid, False otherwise.

        Args:
            query (str): sql query to be validated.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        if not query.strip() or not query.lower().startswith("select"):
            return False

        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
            if not match:
                return False
            table_name = match.group(1).strip()

            column_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
            if not column_match:
                return False
            columns = column_match.group(1).strip().split(",")

            self.cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name=%s",
                (table_name,),
            )
            if not self.cursor.fetchone():
                return False

            self.cursor.execute(
                sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"), [table_name]
            )
            table_columns = [row[0] for row in self.cursor.fetchall()]
            for column in columns:
                column = column.strip()
                if column not in table_columns and column != "*":
                    return False

            return True
        except psycopg2.Error as e:
            logger.error(f"Error validating query: {e}")
            return False

    def fetch_data_from_database(self, table_name: str) -> pd.DataFrame:
        """Fetch data from a PostgreSQL database table.

        Args:
            db_file (Path): Path to the PostgreSQL database file.
            table_name (str): Name of the table to fetch data from.

        Returns:get_primary_key
            pd.DataFrame: A DataFrame containing the data from the table, or None if an error occurred.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            conn = self._db_connection
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)  # type: ignore
        except psycopg2.Error as e:
            logger.error(f"Error fetching data: {e}")
            if self._db_connection:
                self._db_connection.rollback()  # Rollback the transaction on error
            return pd.DataFrame()  # Return an empty DataFrame on error

        return df

    def get_primary_key(self, table_name: str) -> str:
        """
        Retrieves the primary key column name from a PostgreSQL table.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            str: The name of the primary key column.

        Raises:
            ValueError: If the database connection is not established,
                        if the table has no primary key, or
                        if the table has multiple primary keys.
            psycopg2.Error: If there is an error executing the SQL command.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            query = """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY';
            """
            self.cursor.execute(query, (table_name,))
            results = self.cursor.fetchall()  # Fetch all results

            if not results:
                raise ValueError(f"No primary key found for table '{table_name}'.")
            if len(results) > 1:
                raise ValueError(f"Multiple primary keys found for table '{table_name}'.")

            return results[0][0]  # Return the first primary key

        except psycopg2.Error as e:
            raise psycopg2.Error(f"Error getting primary key from table '{table_name}': {e}")

    @property
    def db_connection(self):
        if self._db_connection is None:
            raise ValueError("Database connection not established.")
        return self._db_connection
