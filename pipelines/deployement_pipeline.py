import numpy as np
import pandas as pd
import json
from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pydantic import BaseModel, Field
from typing import Dict, Any
from .utils import get_data_for_test

from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name: str,
        running: bool = True,
        model_name: str = "model",
    ) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

class DeployementTriggerConfig(BaseModel):
    """Configuration for deployment trigger."""
    min_accuracy: float = Field(0.92, description="Minimum accuracy for deployment.")
   
@step(enable_cache=False)
def deployement_trigger(
    score: float,
    config: DeployementTriggerConfig,
) -> bool:
    """Trigger deployment based on model accuracy."""
    if score > config.min_accuracy:
        return True
    return False

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    ):
    """Pipeline for continuous deployment of ML models."""
    data = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(data)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
    decision = deployement_trigger(r2, DeployementTriggerConfig(min_accuracy=min_accuracy))
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=decision,
        workers=workers,
        timeout=timeout,
        )
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)