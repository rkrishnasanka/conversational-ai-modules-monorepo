from pathlib import Path
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