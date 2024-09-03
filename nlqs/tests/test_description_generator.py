from pathlib import Path
import chromadb
from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.description_generator import get_chroma_collection


def test_get_chroma_collection(sqlite_driver):

    chroma_client = chromadb.PersistentClient()

    db_driver = sqlite_driver

    db_driver.connect()

    primary_key = db_driver.get_primary_key(db_driver.db_config.dataset_table_name)

    ret = get_chroma_collection(
        collection_name="test",
        client=chroma_client,
        db_driver=db_driver,
        primary_key=primary_key,
    )
