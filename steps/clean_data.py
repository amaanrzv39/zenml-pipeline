import logging
from zenml import step
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataPreProcessingStrategy, DataDivideStrategy

@step
def clean_data(data: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
    try:
        data_cleaning = DataCleaning(data, DataPreProcessingStrategy())
        cleaned_data = data_cleaning.handle_data()

        data_divide = DataCleaning(cleaned_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_divide.handle_data()
        logging.info("Data cleaning and division completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning and splitting: {e}")
        raise e