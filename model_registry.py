import mlflow
from mlflow import MlflowClient
from src.utils import *


if __name__ == "__main__":
    experiment_id = create_mlflow_experiment(
        experiment_name="Model registry",
        artifact_location="artifacts",
        tags={"status": "dev", "purpose": "lerning"},
    )
    previous_experiment_id=get_mlflow_experiment(experiment_name = "Random Forest Classifier experiment")

    with mlflow.start_run(run_name="model_registry_RF",experiment_id=experiment_id) as run:
        runs = mlflow.search_runs(experiment_ids=[previous_experiment_id.experiment_id], filter_string="tags.mlflow.runName = 'Random Forest re-training after Feature Engineering'")
        model_uri = f'runs:/{runs.iloc[0]["run_id"]}/random_forest_classifier'
        rfc = mlflow.sklearn.load_model(model_uri=model_uri)
        mlflow.sklearn.log_model(artifact_path="models", sk_model=rfc, registered_model_name="RandomForestClassifier")
    
    client= MlflowClient()
    model_name="RandomForestClassifier"

    # Description of registired model
    client.update_registered_model(name=model_name, description="This is the first test model")

    # Tags of registired model
    # One of the most important tages is the stage of the model which can be: "Production", "Staging", "Archived", or "None".
    client.set_registered_model_tag(name=model_name, key="Stage", value="Staging")

    # Adding description to model version.
    client.update_model_version(name=model_name, version=1, description="This is a test model version")

    # adding tags to model version.
    client.set_model_version_tag(name=model_name, version=1, key="Stage", value="Staging")
    