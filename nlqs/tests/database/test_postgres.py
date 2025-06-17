from unittest.mock import patch

import pandas as pd
import psycopg2
import pytest
from psycopg2 import sql

from nlqs.parameters import DEFAULT_TABLE_NAME


def test_connect_successful(postgres_driver):
    """Test successful connection to the database."""
    postgres_driver.connect()
    assert postgres_driver.db_connection is not None
    assert postgres_driver.cursor is not None


# def test_connec_with_error(postgres_driver):
#     """Test connection # failure with psycopg2.Error."""
#     postgres_driver.config.host = "invalid_host"
#     with pytest.raises(psycopg2.Error):
#         postgres_driver.connect()


def test_disconnect_successful(postgres_driver):
    """Test successful disconnection from the database."""
    postgres_driver.connect()
    postgres_driver.disconnect()
    assert postgres_driver.db_connection.closed == 1


def test_execute_query_successful(postgres_driver):
    """Test successful execution of a SQL query."""
    postgres_driver.connect()
    result = postgres_driver.execute_query("SELECT 1")
    assert result == [(1,)]


def test_execute_query_with_error(postgres_driver, setup_postgres_database):
    """Test handling of errors during query execution."""
    postgres_driver.connect()
    with pytest.raises(psycopg2.Error):
        postgres_driver.execute_query("SELECT * FROM nonexistent_table")


def test_get_database_columns_successful(postgres_driver, setup_postgres_database):
    """Test retrieval of database columns in order."""
    postgres_driver.connect()
    expected_columns = ["id", "name", "value"]
    columns = postgres_driver.get_database_columns(DEFAULT_TABLE_NAME)

    assert frozenset(columns) == frozenset(expected_columns)


def test_validate_query_valid_query(postgres_driver):
    """Test validation of a valid SQL query."""
    postgres_driver.connect()
    expected_return_value = [("id",), ("name",), ("value",)]
    is_valid = postgres_driver.validate_query("SELECT name, value FROM test_table WHERE id = 1")
    assert is_valid is True


def test_validate_query_invalid_query(postgres_driver):
    """Test validation of an invalid SQL query for a non-existent table."""
    postgres_driver.connect()

    # Call the validate_query function with an invalid table
    is_valid = postgres_driver.validate_query("SELECT * FROM nonexistent_table")

    # Assert that the query is invalid since the table doesn't exist
    assert is_valid is False


def test_fetch_data_from_database_successful(postgres_driver):
    """Test successful data fetching from the database."""
    postgres_driver.connect()
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["John", "Jane"], "value": [10.5, 20.0]})
    with patch("pandas.read_sql_query", return_value=mock_df) as mock_read_sql:
        df = postgres_driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.equals(mock_df)


def test_fetch_data_from_database_with_error(postgres_driver):
    """Test data fetching from the database with psycopg2.Error."""
    postgres_driver.connect()
    with patch("pandas.read_sql_query", side_effect=psycopg2.Error("Test error")) as mock_read_sql:
        df = postgres_driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_primary_key(postgres_driver, setup_postgres_database):
    """Test getting the primary key of a table."""
    postgres_driver.connect()
    expected_result = "id"  # List of tuples
    primary_key = postgres_driver.get_primary_key(DEFAULT_TABLE_NAME)
    assert primary_key == expected_result  # Assert on the column name


def test_get_primary_key_no_primary_key(postgres_driver):
    """Test getting the primary key when the table has no primary key."""
    postgres_driver.connect()
    # mock_connection.cursor.return_value.fetchall.return_value = []  # No primary key
    with pytest.raises(ValueError) as context:
        postgres_driver.get_primary_key("test_table")
    assert "No primary key found" in str(context.value)


def test_get_primary_key_multiple_primary_keys(postgres_driver):
    """Test getting the primary key when the table has multiple primary keys."""
    postgres_driver.connect()
    # mock_connection.cursor.return_value.fetchall.return_value = [
    #     ("id1",),
    #     ("id2",),
    # ]  # Mock multiple primary keys
    with pytest.raises(ValueError) as context:
        postgres_driver.get_primary_key("test_table_multiple_pk")
    assert "Multiple primary keys found" in str(context.value)
