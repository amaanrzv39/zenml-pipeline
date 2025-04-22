from pipelines.deployement_pipeline import continuous_deployment_pipeline, inference_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import click
from rich import print
from typing import cast

@click.command()
@click.option(
    "--mode",
    "-c",
    type=click.Choice(["DEPLOY", "PREDICT", "DEPLOY_AND_PREDICT"], case_sensitive=False),
    default="DEPLOY_AND_PREDICT",
    help="Pipeline to train and deploy model or only get predictions.",
)
@click.option(
    "--min-accuracy",
    default = 0.92,
    help="Minimum accuracy for the model to be deployed.",
)

def run_deployement(mode: str, min_accuracy: float):
    """Run the deployment pipeline."""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = mode == "DEPLOY" or mode == "DEPLOY_AND_PREDICT"
    predict = mode == "PREDICT" or mode == "DEPLOY_AND_PREDICT"
    if deploy:
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=2,
            timeout=60,
        )
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )

    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        service.start(timeout=60)
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )

if __name__ == "__main__":
    run_deployement()