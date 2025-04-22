import logging
from zenml import step
import pandas as pd

class IngestData:

    def load_data(self) -> pd.DataFrame:
        """Load data from the specified path."""
        try:
            data = pd.read_csv("data/olist_customers_dataset.csv")
            logging.info(f"Data loaded successfully")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e

@step(enable_cache=False)
def ingest_data() -> pd.DataFrame:
    """Ingest data from a CSV file."""
    ingestor = IngestData()
    data = ingestor.load_data()
    return data    