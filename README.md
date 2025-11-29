# MLOps Assignment: End-to-End Machine Learning Pipeline

## Project Overview
This project implements a complete Machine Learning Operations (MLOps) pipeline for predicting housing prices using the California Housing dataset. It demonstrates data versioning, pipeline orchestration, model training, and continuous integration using industry-standard tools.

**Key Technologies:**
-   **DVC**: For data versioning and management.
-   **MLflow**: For pipeline orchestration, experiment tracking, and model management.
-   **GitHub Actions**: For Continuous Integration (CI) to automate pipeline testing.
-   **Jenkins**: For local CI/CD pipeline execution.
-   **Python/Scikit-learn**: For the machine learning model (Random Forest).

## Project Structure
```
mlops-kubeflow-assignment/
├── .github/workflows/    # GitHub Actions CI configurations
├── components/           # (Legacy) Component definitions
├── data/                 # Data directory (tracked by DVC)
├── src/                  # Source code and auxiliary scripts
├── pipeline.py           # Main MLflow pipeline script
├── requirements.txt      # Project dependencies
├── Dockerfile            # Docker configuration
├── Jenkinsfile           # Jenkins pipeline configuration
└── README.md             # Project documentation
```

## Setup Instructions

### Prerequisites
-   Python 3.9+
-   Git
-   (Optional) Docker for running Jenkins locally

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/AmarRam90/mlops-kubeflow-assignment.git
    cd mlops-kubeflow-assignment
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup DVC (Data Version Control)**:
    *   Initialize DVC (already done in this repo):
        ```bash
        dvc init
        ```
    *   Pull the data from the remote storage (if configured):
        ```bash
        dvc pull
        ```

## Pipeline Walkthrough

The pipeline is orchestrated using **MLflow** and consists of four stages:
1.  **Data Extraction**: Loads the California Housing dataset.
2.  **Data Preprocessing**: Cleans, splits (80/20), and scales the data.
3.  **Model Training**: Trains a Random Forest Regressor.
4.  **Model Evaluation**: Evaluates the model and logs metrics (R2, RMSE).

### Running the Pipeline Locally
To run the entire pipeline and track experiments:

```bash
python pipeline.py
```

### Viewing Results
To view the experiment runs, parameters, and artifacts:

1.  Start the MLflow UI:
    ```bash
    mlflow ui
    ```
2.  Open [http://localhost:5000](http://localhost:5000) in your browser.
3.  Navigate to the experiment **"California_Housing_Pipeline"** to see your runs.

## Continuous Integration (CI)

This project uses both **GitHub Actions** and **Jenkins** for Continuous Integration.

### GitHub Actions
-   **Workflow File**: `.github/workflows/mlflow-ci.yml`
-   **Triggers**: Pushes and Pull Requests to the `main` branch.
-   **Process**:
    1.  Sets up a Python environment.
    2.  Installs dependencies.
    3.  Lints the code with `flake8`.
    4.  Runs `pipeline.py` to verify pipeline integrity.

### Jenkins
-   **Pipeline File**: `Jenkinsfile`
-   **Setup**:
    -   The pipeline is defined to run in a Docker container or local agent.
    -   It creates a virtual environment, installs dependencies, runs linting, and executes the pipeline.
-   **Execution**:
    -   The job is triggered via polling or manual build.
    -   It ensures that the code builds and runs successfully in an isolated environment.