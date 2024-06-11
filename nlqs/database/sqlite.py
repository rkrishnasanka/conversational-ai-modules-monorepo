import sqlite3
from typing import Tuple
from discord_bot.parameters import SQLITE_DB_FILE
from nlqs.database.driver import AbstractDriver

class SQLiteDriver(AbstractDriver):

    def __init__(self, config):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    def execute_query(self, query):
        pass


def retrieve_descriptions_and_types_from_db(db_file: str= SQLITE_DB_FILE) -> Tuple:
    """ Retrieves descriptions and types from the SQLite database.

    Args:
        db_file (SQLite database, optional): SQLite database to store all the tables. Defaults to SQLITE_DB_FILE.

    Returns:
        tuple: Return descriptions, numerical_columns, categorial_columns
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # Retrieve descriptions
    c.execute('SELECT column_name, description FROM column_descriptions')
    description_rows = c.fetchall()
    descriptions = {row[0]: row[1] for row in description_rows}

    # Retrieve column types
    c.execute('SELECT column_name, column_type FROM column_types')
    type_rows = c.fetchall()
    numerical_columns = [row[0] for row in type_rows if row[1] == 'numerical']
    categorical_columns = [row[0] for row in type_rows if row[1] == 'categorical']

    conn.close()
    return descriptions, numerical_columns, categorical_columns

def execute_query(query:str) -> str:
    """ Executes the SQL query and returns the result.

    Args:
        query (str): the SQL query.

    Returns:
        str: the result of the query.
    """
    try:
        #SQLite connection
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        result = str(result)
        # logger.info(f"Query Result: {result}")

        return result if result else "No results found."
    except sqlite3.Error as e:
        error_message = f"Error executing SQL query: {e}"
        # logger.error(f"Error executing SQL query: {e}")
        return error_message