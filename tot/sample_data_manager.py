import json
import io
import csv
from typing import List, Dict, Any

class SampleDataManager:
    """
    Manages sample CSV data by loading and providing it in JSON format.
    """

    def __init__(self, csv_data: str):
        """
        Initialize the SampleDataManager with CSV data.

        Args:
            csv_data (str): CSV formatted data as a string.
        """
        self.sample_data = self._load_sample_data(csv_data)

    def _load_sample_data(self, csv_data: str) -> List[Dict[str, Any]]:
        """
        Load CSV data into a list of dictionaries.

        Args:
            csv_data (str): CSV formatted data as a string.

        Returns:
            List[Dict[str, Any]]: List of dictionaries representing CSV rows.
        """
        try:
            # Create a file-like object from the CSV string
            csv_file = io.StringIO(csv_data.strip())
            # Create a CSV reader object
            reader = csv.DictReader(csv_file)
            # Convert CSV rows to a list of dictionaries
            return [row for row in reader]
        except csv.Error as e:
            print(f"Error loading CSV data: {e}")
            return []

    def get_sample_data(self) -> str:
        """
        Get the sample data as a JSON string.

        Returns:
            str: JSON formatted string of the sample data.
        """
        # Convert the sample data to a JSON string with indentation
        return json.dumps(self.sample_data, indent=2)