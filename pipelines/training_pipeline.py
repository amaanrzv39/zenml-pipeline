from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    """Define the training pipeline."""
    # Ingest data
    data = ingest_data(data_path=data_path)
    
    # Clean data
    clean_data(data)
    train_model(data)
    evaluate_model(data)
