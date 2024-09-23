from unittest.mock import patch

import pandas as pd
import psycopg2
import pytest
from psycopg2 import sql


def test_connect_successful(driver, mock_connection):
    """Test successful connection to the database."""
    driver.connect()
    assert driver.db_connection == mock_connection
    assert driver.cursor == mock_connection.cursor.return_value


def test_connect_with_error(driver, mock_connection):
    """Test connection failure with psycopg2.Error."""
    mock_connection.cursor.side_effect = psycopg2.Error("Test error")
    with pytest.raises(psycopg2.Error):
        driver.connect()


def test_disconnect_successful(driver, mock_connection):
    """Test successful disconnection from the database."""
    driver.connect()
    driver.disconnect()
    mock_connection.close.assert_called_once()


def test_execute_query_successful(driver, mock_connection):
    """Test successful execution of a SQL query."""
    driver.connect()
    expected_result = [("Jane",), ("John",)]
    mock_connection.cursor.return_value.fetchall.return_value = expected_result
    result = driver.execute_query("SELECT name FROM test_table WHERE value > 15")
    assert result == expected_result


def test_execute_query_with_error(driver, mock_connection):
    """Test handling of errors during query execution."""
    driver.connect()
    mock_connection.cursor.return_value.execute.side_effect = psycopg2.Error("Test error")
    with pytest.raises(psycopg2.Error):
        driver.execute_query("SELECT * FROM nonexistent_table")


def test_retrieve_descriptions_and_types_from_db_successful(driver, mock_connection):
    """Test successful retrieval of descriptions and types from the database."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.side_effect = [
        [
            ("id", "Unique identifier"),
            ("name", "Name of the person"),
            ("value", "Some value"),
        ],
        [("id", "numerical"), ("name", "categorical"), ("value", "numerical")],
    ]

    expected_descriptions = {
        "id": "Unique identifier",
        "name": "Name of the person",
        "value": "Some value",
    }
    expected_numerical_columns = ["id", "value"]
    expected_categorical_columns = ["name"]

    descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == expected_descriptions
    assert numerical_columns == expected_numerical_columns
    assert categorical_columns == expected_categorical_columns


def test_retrieve_descriptions_and_types_from_db_with_error(driver, mock_connection):
    """Test handling of errors during descriptions and types retrieval."""
    driver.connect()
    mock_connection.cursor.return_value.execute.side_effect = psycopg2.Error("Test error")

    descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == {}
    assert numerical_columns == []
    assert categorical_columns == []


def test_get_database_columns_successful(driver, mock_connection):
    """Test retrieval of database columns in order."""
    driver.connect()
    expected_columns = [("id",), ("name",), ("value",)]
    mock_connection.cursor.return_value.fetchall.return_value = expected_columns
    columns = driver.get_database_columns("test_table")
    driver.cursor.execute.assert_called_once_with(
        sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"), ["test_table"]
    )
    assert columns == ["id", "name", "value"]


def test_get_database_columns_with_error(driver, mock_connection):
    """Test handling of errors during column retrieval."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.side_effect = Exception("Test error")
    columns = driver.get_database_columns("test_table")
    assert columns == []


def test_validate_query_valid_query(driver, mock_connection):
    """Test validation of a valid SQL query."""
    driver.connect()
    mock_connection.cursor.return_value.fetchone.return_value = True
    mock_connection.cursor.return_value.fetchall.return_value = [("id",), ("name",), ("value",)]
    is_valid = driver.validate_query("SELECT name, value FROM test_table WHERE id = 1")
    assert is_valid is True


def test_validate_query_invalid_query(driver, mock_connection):
    """Test validation of an invalid SQL query for a non-existent table."""
    driver.connect()

    # Simulate no results for the table (table does not exist).
    mock_connection.cursor.return_value.fetchone.return_value = None  # No table found
    mock_connection.cursor.return_value.fetchall.return_value = []  # No columns found

    # Call the validate_query function with an invalid table
    is_valid = driver.validate_query("SELECT * FROM nonexistent_table")

    # Assert that the query is invalid since the table doesn't exist
    assert is_valid is False


def test_fetch_data_from_database_successful(driver, mock_connection):
    """Test successful data fetching from the database."""
    driver.connect()
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["John", "Jane"], "value": [10.5, 20.0]})
    with patch("pandas.read_sql_query", return_value=mock_df) as mock_read_sql:
        df = driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.equals(mock_df)


def test_fetch_data_from_database_with_error(driver, mock_connection):
    """Test data fetching from the database with psycopg2.Error."""
    driver.connect()
    with patch("pandas.read_sql_query", side_effect=psycopg2.Error("Test error")) as mock_read_sql:
        df = driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_primary_key(driver, mock_connection):
    """Test getting the primary key of a table."""
    driver.connect()
    expected_result = [("id",)]  # List of tuples
    mock_connection.cursor.return_value.fetchall.return_value = expected_result
    primary_key = driver.get_primary_key("test_table")
    assert primary_key == expected_result[0][0]  # Assert on the column name


def test_get_primary_key_no_primary_key(driver, mock_connection):
    """Test getting the primary key when the table has no primary key."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = []  # No primary key
    with pytest.raises(ValueError) as context:
        driver.get_primary_key("test_table")
    assert "No primary key found" in str(context.value)


def test_get_primary_key_multiple_primary_keys(driver, mock_connection):
    """Test getting the primary key when the table has multiple primary keys."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = [
        ("id1",),
        ("id2",),
    ]  # Mock multiple primary keys
    with pytest.raises(ValueError) as context:
        driver.get_primary_key("test_table_multiple_pk")
    assert "Multiple primary keys found" in str(context.value)
