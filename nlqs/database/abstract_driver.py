from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class AbstractDriver(ABC):

    def __init__(self, config):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def connect(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def execute_query(self, query):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def retrieve_descriptions_and_types_from_db(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def fetch_data_from_database(self, table_name: str) -> pd.DataFrame:
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def validate_query(self, query):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def get_database_columns(self, table_name: str) -> List[str]:
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def get_primary_key(self, table_name: str) -> str:
        raise NotImplementedError("This method must be implemented by the subclass")