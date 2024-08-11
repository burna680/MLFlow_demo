import matplotlib.pyplot as plt 

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

from src.data.data_gathering import gather_data
from src.data.data_analysis import analyze_data
from src.data.data_versioning import version_data
from src.data.data_preparation import prepare_data
from src.model.model_training import train_model
from src.model.model_validation import validate_model
from src.utils import *

def main():
    # Define the experiment
    experiment_id = create_mlflow_experiment(
        experiment_name="Random Forest Classifier experiment",
        artifact_location="artifacts",
        tags={"status": "dev", "version": "1.0.1"},
    )
    experiment = get_mlflow_experiment(experiment_name="Random Forest Classifier experiment")
    with mlflow.start_run(run_name="Random Forest re-training after Feature Engineering",experiment_id=experiment.experiment_id) as run:
        # Enable autologging
        mlflow.tensorflow.autolog()
        
        ## To have more control over the parameters logged by MLflow, you can use the `log_param` and `log_params` methods
        ## One at a time
        # mlflow.log_param("learning_rate", 0.01)
         ## All at once
        # parameters = {
        #     "learning_rate": 0.01,
        #     "epochs": 100,
        #     "loss_function": "binary_crossentroy",
        #     "optimizer": "sgd"
        # }
        # mlflow.log_params(parameters)
        
        ## The same goes for the artifacts logged by MLflow
        ## One at a time
        # mlflow.log_artifact(local_path="sample_file.txt", artifact_path="text_files")
        ## All at once
        #mlflow.log_artifacts(local_dir="./run_artifacts",artifact_path="run_artifacts")

        # EDA
        X, y = gather_data()
        analyze_data(X, y)
        # version_data(X, y)

        # Data Preparation
        X_train, X_test, y_train, y_test = prepare_data(X, y)

        # Retraining the model with only important features obtained from EDA
        X_train = X_train.drop(['doors'], axis=1)
        X_test = X_test.drop(['doors'], axis=1)
        # Training
        clf = train_model(X_train, y_train)

        # Validation
        accuracy, y_pred = validate_model(clf, X_test, y_test)

        # Results
        # Confusion matrix
        fig_cm = plt.figure()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
        plt.title("Confusion Matrix")
        plt.legend()

        mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")
        # Serving
        
        # Monitoring

        # Logging 
        model_signature = infer_signature(X_train, y_train, params={"model_name": "clf1"})
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="random_forest_classifier",signature=model_signature,)

if __name__ == "__main__":
    main()
