from pathlib import Path
import sqlite3
import pytest

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver


@pytest.fixture
def sqlite_driver():
    sqlite_config = SQLiteConnectionConfig(
    db_file=Path("aegion.db"),  
    dataset_table_name="new_dataset"
    )

    db_driver = SQLiteDriver(sqlite_config=sqlite_config)

    return db_driver

@pytest.fixture(scope="function")
def setup_database():
    """Setup method to create a test database and driver instance."""
    test_db_file = Path("test_database.db")
    sqlite_config = SQLiteConnectionConfig(db_file=test_db_file, dataset_table_name="test_table")
    driver = SQLiteDriver(sqlite_config)

    # Create a test database and table
    with sqlite3.connect(test_db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
            """
        )
        cursor.execute("INSERT INTO test_table (name, value) VALUES ('John', 10.5)")
        cursor.execute("INSERT INTO test_table (name, value) VALUES ('Jane', 20.0)")

        # Create a table for column descriptions
        cursor.execute(
            """
            CREATE TABLE column_descriptions (
                column_name TEXT PRIMARY KEY,
                description TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("id", "Unique identifier"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("name", "Name"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("value", "Value"),
        )

        # Create a table for column types
        cursor.execute(
            """
            CREATE TABLE column_types (
                column_name TEXT PRIMARY KEY,
                column_type TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("id", "numerical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("name", "categorical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("value", "numerical"),
        )

        # Create a table with multiple primary keys
        cursor.execute(
            """
            CREATE TABLE test_table_multiple_pk (
                id1 INTEGER,
                id2 INTEGER,
                name TEXT,
                PRIMARY KEY (id1, id2)
            )
            """
        )

    yield driver

    # Cleanup method to remove the test database file
    test_db_file.unlink(missing_ok=True)
