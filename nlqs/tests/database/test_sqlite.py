import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import DEFAULT_TABLE_NAME


def test_connect_successful(setup_sqlite_database, sqlite_driver):
    """Test successful connection to the database."""

    sqlite_driver.connect()
    assert sqlite_driver.engine is not None
    assert sqlite_driver.Session is not None


def test_connect_failed(sqlite_driver, setup_sqlite_database):
    """Test connection failure."""
    with pytest.raises(SQLAlchemyError):
        sqlite_driver.connect()


def test_disconnect_successful(setup_sqlite_database, sqlite_driver):
    """Test successful disconnection from the database."""

    sqlite_driver.connect()
    sqlite_driver.disconnect()
    # No need for assertions, SQLAlchemy handles disconnection internally


def test_execute_query_successful(setup_sqlite_database, sqlite_driver):
    """Test successful execution of a SQL query."""

    sqlite_driver.connect()
    result = sqlite_driver.execute_query("SELECT name FROM test_table WHERE value > 15")
    assert result[0][0] == "Jane"


def test_execute_query_with_error(setup_sqlite_database, sqlite_driver):
    """Test handling of errors during query execution."""

    sqlite_driver.connect()
    with pytest.raises(SQLAlchemyError):
        sqlite_driver.execute_query("SELECT * FROM nonexistent_table")


def test_retrieve_descriptions_and_types_from_db_successful(setup_sqlite_database, sqlite_driver):
    """Test successful retrieval of descriptions and types from the database."""

    sqlite_driver.connect()

    expected_descriptions = {"id": "Unique identifier", "name": "Name", "value": "Value"}
    expected_numerical_columns = ["id", "value"]
    expected_categorical_columns = ["name"]

    descriptions, numerical_columns, categorical_columns = sqlite_driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == expected_descriptions
    assert numerical_columns == expected_numerical_columns
    assert categorical_columns == expected_categorical_columns


def test_retrieve_descriptions_and_types_from_db_with_error(setup_sqlite_database, sqlite_driver):
    """Test handling of errors during descriptions and types retrieval."""

    sqlite_driver.connect()

    descriptions = {}
    numerical_columns = []
    categorical_columns = []

    # Drop the tables to simulate an error condition
    with sqlite_driver.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS column_descriptions"))
        conn.execute(text("DROP TABLE IF EXISTS column_types"))

    # Assert that an error is raised and handled gracefully
    with pytest.raises(SQLAlchemyError, match="no such table: column_descriptions"):
        descriptions, numerical_columns, categorical_columns = sqlite_driver.retrieve_descriptions_and_types_from_db()

    # Assert that empty values are returned
    assert descriptions == {}
    assert numerical_columns == []
    assert categorical_columns == []


def test_get_database_columns(setup_sqlite_database, sqlite_driver):
    """Test retrieval of database columns in order."""

    sqlite_driver.connect()
    columns = sqlite_driver.get_database_columns(DEFAULT_TABLE_NAME)
    assert columns == ["id", "name", "value"]


def test_validate_query_valid_query(setup_sqlite_database, sqlite_driver):
    """Test validation of a valid SQL query."""

    sqlite_driver.connect()
    is_valid = sqlite_driver.validate_query("SELECT name, value FROM test_table WHERE id = 1")
    assert is_valid


def test_validate_query_invalid_query(setup_sqlite_database, sqlite_driver):
    """Test validation of an invalid SQL query."""

    sqlite_driver.connect()
    is_valid = sqlite_driver.validate_query("SELECT * FROM nonexistent_table")
    assert not is_valid


def test_check_table_exists_existing_table(setup_sqlite_database, sqlite_driver):
    """Test checking for an existing table."""

    sqlite_driver.connect()
    assert sqlite_driver.check_table_exists(DEFAULT_TABLE_NAME)


def test_check_table_exists_nonexistent_table(setup_sqlite_database, sqlite_driver):
    """Test checking for a nonexistent table."""

    sqlite_driver.connect()
    assert not sqlite_driver.check_table_exists("nonexistent_table")


def test_fetch_data_from_database_successful(setup_sqlite_database, sqlite_driver):
    """Test successful data fetching from the database."""

    sqlite_driver.connect()
    df = sqlite_driver.fetch_data_from_database(DEFAULT_TABLE_NAME)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Check if two rows were fetched


def test_fetch_data_from_database_nonexistent_table(setup_sqlite_database, sqlite_driver):
    """Test data fetching from a nonexistent table."""

    sqlite_driver.connect()
    with pytest.raises(ValueError, match="Table 'nonexistent_table' does not exist in the database."):
        sqlite_driver.fetch_data_from_database("nonexistent_table")


def test_get_primary_key(setup_sqlite_database, sqlite_driver):
    """Test getting the primary key of a table."""

    sqlite_driver.connect()
    primary_key = sqlite_driver.get_primary_key(DEFAULT_TABLE_NAME)
    assert primary_key == "id"


def test_get_primary_key_no_primary_key(setup_sqlite_database, sqlite_driver):
    """Test getting the primary key when the table has no primary key."""

    sqlite_driver.connect()
    with pytest.raises(ValueError, match="No primary key found"):
        sqlite_driver.get_primary_key("test_table_no_pk")


def test_multithreading(setup_sqlite_database, sqlite_driver):
    """Test that multiple threads can access the database simultaneously."""

    sqlite_driver.connect()

    # Define a function to be executed in multiple threads
    def insert_data(name, value):
        with sqlite_driver.Session() as session:
            session.execute(
                text("INSERT INTO test_table (name, value) VALUES (:name, :value)"),
                {"name": name, "value": value},
            )
            session.commit()

    # Create and start multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=insert_data, args=(f"Thread-{i}", i * 10))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check if all data was inserted correctly
    with sqlite_driver.Session() as session:
        result = session.execute(text("SELECT * FROM test_table")).fetchall()
        assert len(result) == 7  # 2 initial rows + 5 from threads
