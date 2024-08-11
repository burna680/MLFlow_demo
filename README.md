# MLFlow_demo

# MLOps Pipeline Project

## Overview

This project implements a complete MLOps pipeline for a machine learning scikit-learn model using Python. The pipeline includes the following steps:
1. **Data Gathering**: Fetches data from the UCI ML repository.
2. **Data Analysis**: Performs exploratory data analysis (EDA) to understand the dataset.
3. **Data Versioning**: Saves the dataset for version control. (TODO wiwth DVC)
4. **Data Preparation**: Prepares the data for modeling through feature engineering and data splitting.
5. **Model Training & Development**: Trains a RandomForestClassifier on the prepared data.
6. **Model Validation**: Validates the model using accuracy metrics and other evaluation tools.
7. **Model Serving**: Saves the trained model for deployment. (TODO)
8. **Model Monitoring**: Logs predictions and tracks model performance over time. (TODO)

## Project Structure

```plaintext
ml_pipeline/
├── data/
│   ├── raw/
│   └── prepared/
├── src/
│   ├── data/
│   └── model/
├── artifacts/
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/burna680/MLFlow_demo.git
   cd MLFlow_demo
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the MLflow server** (if you haven't already):
   ```bash
   mlflow ui
   ```

## Usage

### Running the Pipeline

Run the main script to execute the entire pipeline:

```bash
python main.py
```

This will perform all steps of the pipeline, from data gathering to model training. Outputs, logs, and saved models will be stored in the appropriate directories under `data/`, `model/`, and `artifacts/`.

### Exploring the Project

- **Data Modules**: Located under `src/data/`, these modules handle everything from gathering and preparing data to versioning it using MLflow.
- **Model Modules**: Located under `src/model/`, these modules are responsible for training, validating, serving, monitoring, and retraining the model.
- **Utilities**: Common and useful functions for the MLFlow project can be placed in `src/utils.py` to keep the code DRY.

## Future Work

- **CI/CD Integration**: Add continuous integration and continuous deployment pipelines.
- **Model Deployment**: Implement model deployment using tools like Flask or FastAPI.
- **Advanced Monitoring**: Incorporate advanced monitoring and alerting mechanisms.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
