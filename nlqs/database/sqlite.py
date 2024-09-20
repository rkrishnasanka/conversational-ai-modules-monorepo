import re
import sqlite3
import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence
from sqlalchemy import create_engine, text
from sqlalchemy.engine.row import Row
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from nlqs.database.abstract_driver import AbstractDriver

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class SQLiteConnectionConfig:
    db_file: Path
    dataset_table_name: str
    uri_column: Optional[str] = None
    output_columns: Optional[List[str]] = None


class SQLiteDriver(AbstractDriver):
    def __init__(self, sqlite_config: SQLiteConnectionConfig):
        self.db_config = sqlite_config
        self.engine = None
        self.Session = None

    def connect(self):
        try:
            if not self.db_config.db_file.exists():
                raise ValueError(f"Database file '{self.db_config.db_file}' does not exist.")
            # Create an engine with connection pool
            self.engine = create_engine(f"sqlite:///{self.db_config.db_file.absolute()}", echo=True, future=True)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Connected to SQLite database with connection pool.")
        except SQLAlchemyError as e:
            logger.error(f"Error connecting to database with connection pool: {e}")
            raise e

    def disconnect(self):
        # In SQLAlchemy, connections are returned to the connection pool after session.close()
        # There's no need to explicitly close the engine
        logger.info("Disconnected from SQLite database.")

    def execute_query(self, query: str) -> Optional[Sequence[Row[Any]]]:
        """Executes a SQL query on the SQLite database.

        Args:
            query (str): The SQL query to execute.

        Raises:
            ValueError: If the database connection is not established.
            e: If there is an error executing the SQL query.

        Returns:
            Optional[Sequence[Row[Any]]]: The result of the query, or None if the result is empty.
        """
        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")

        with self.Session() as session:
            try:

                result = session.execute(text(query)).fetchall()
                session.commit()
                logger.info(f"Query executed successfully: {result}")
                return result if result else None

            except SQLAlchemyError as e:
                session.rollback()
                error_message = f"Error executing SQL query: {e}"
                logger.error(error_message)
                raise e

    def retrieve_descriptions_and_types_from_db(self) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Retrieves descriptions and types from the SQLite database.

        Args:
            db_file (SQLite database, optional): SQLite database to store all the tables. Defaults to SQLITE_DB_FILE.

        Returns:
            Tuple[List[str], List[str], List[str]]: Return descriptions, numerical_columns, categorial_columns
        """
        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")
        # Retrieve descriptions
        with self.Session() as session:
            try:
                description_rows = session.execute(
                    text("SELECT column_name, description FROM column_descriptions")
                ).fetchall()
                descriptions = {row[0]: row[1] for row in description_rows}

                # Retrieve column types
                type_rows = session.execute(text("SELECT column_name, column_type FROM column_types")).fetchall()
                numerical_columns = [row[0] for row in type_rows if row[1] == "numerical"]
                categorical_columns = [row[0] for row in type_rows if row[1] == "categorical"]

                return descriptions, numerical_columns, categorical_columns
            except sqlite3.Error as e:
                logger.error(f"Error retrieving descriptions and types: {e}")
                return {}, [], []

    def get_database_columns(self, table_name: str) -> List[str]:
        """Returns the columns in the specified table in the order they appear in the database.

        Args:
            table_name (str): The name of the table from which to retrieve columns.

        Raises:
            ValueError: Error retrieving columns.

        Returns:
            List[str]: The columns in the database table in order.
        """
        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")

        with self.Session() as session:
            try:
                columns_info = session.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                columns_in_database = [column[1] for column in columns_info]  # The second field is the column name
                return columns_in_database
            except sqlite3.Error as e:
                logger.error(f"Error retrieving columns: {e}")
                return []

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

        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")

        with self.Session() as session:
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

                statement = session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                    {"table_name": table_name},
                )
                if not statement.fetchone():
                    logger.error(f"Table '{table_name}' not found in the database.")
                    print(f"Table '{table_name}' not found in the database.")
                    return False

                statement = session.execute(text(f"PRAGMA table_info({table_name})"))
                table_columns = [column[1] for column in statement.fetchall()]
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
        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")

        with self.Session() as session:
            try:
                result = session.execute(
                    text(
                        """
                    SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name
                    """
                    ),
                    {"table_name": table_name},
                ).fetchone()

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

            conn = self.engine
            query = f"SELECT * FROM {table_name}"
            if conn is None:
                raise ValueError("Database connection not established.")
            df = pd.read_sql_query(query, conn)
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

        return df

    def get_primary_key(self, table_name: str) -> str:
        """
        Retrieves the primary key column name from a SQLite table.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            str: The name of the primary key column.

        Raises:
            ValueError: If the database connection is not established or
                        if the table has no primary key.
            sqlite3.Error: If there is an error executing the SQL command.
        """
        if self.Session is None or self.engine is None:
            raise ValueError("Database connection not established.")

        with self.Session() as session:
            try:
                table_info = session.execute(text(f"PRAGMA table_info({table_name})")).fetchall()

                primary_key_columns = [row[1] for row in table_info if row[5] == 1]

                if len(primary_key_columns) == 0:
                    raise ValueError(f"No primary key found in the table '{table_name}'.")

                if len(primary_key_columns) > 1:
                    raise ValueError(f"Multiple primary keys found in the table '{table_name}'.")

                return primary_key_columns[0]

            except sqlite3.Error as e:
                raise sqlite3.Error(f"Error getting primary key from table '{table_name}': {e}")
