import logging
from zenml import step
import pandas as pd

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Load data from the specified path."""
        try:
            data = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file."""
    ingestor = IngestData(data_path)
    data = ingestor.load_data()
    return data    