"""
Kubeflow Pipeline Components for MLOps Assignment
This module contains reusable components for data extraction, preprocessing, 
model training, and evaluation.

IMPORTANT: This file defines components using the @component decorator (KFP v2 style).
To compile to YAML using create_component_from_func as requested in the assignment,
run: python compile_components.py
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


# ==============================================================================
# Component 1: Data Extraction
# ==============================================================================
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "numpy>=1.23.0", "scikit-learn>=1.2.0"]
)
def data_extraction(
    dataset_output: Output[Dataset]
):
    """
    Extract the California Housing dataset.
    
    Note: This component loads data directly from scikit-learn for demonstration.
    In production, this would fetch from a DVC remote storage using cloud storage
    (S3, GCS, Azure) accessible from containers.
    The DVC versioning workflow is demonstrated in Task 1.
    
    Args:
        dataset_output: Output path where the extracted dataset will be saved
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    
    # Load California Housing dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Save to CSV
    df.to_csv(dataset_output.path, index=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(data.feature_names)}")
    print(f"Data successfully extracted to {dataset_output.path}")


# ==============================================================================
# Component 2: Data Preprocessing
# ==============================================================================
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "numpy>=1.23.0", "scikit-learn>=1.2.0"]
)
def data_preprocessing(
    dataset_input: Input[Dataset],
    train_data_output: Output[Dataset],
    test_data_output: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('PreprocessingOutputs', [('n_train_samples', int), ('n_test_samples', int)]):
    """
    Preprocess the dataset: clean, scale, and split into train/test sets.
    
    Args:
        dataset_input: Input dataset from extraction step
        train_data_output: Output path for training data
        test_data_output: Output path for test data
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing number of training and test samples
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    # Load the dataset
    df = pd.read_csv(dataset_input.path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    print(f"After dropping NA: {df.shape}")
    
    # Separate features and target
    # Assuming last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save train data
    train_df = pd.DataFrame(X_train_scaled)
    train_df['target'] = y_train
    train_df.to_csv(train_data_output.path, index=False)
    
    # Save test data
    test_df = pd.DataFrame(X_test_scaled)
    test_df['target'] = y_test
    test_df.to_csv(test_data_output.path, index=False)
    
    # Save scaler for future use
    scaler_path = train_data_output.path.replace('.csv', '_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    from collections import namedtuple
    output = namedtuple('PreprocessingOutputs', ['n_train_samples', 'n_test_samples'])
    return output(len(X_train), len(X_test))


# ==============================================================================
# Component 3: Model Training
# ==============================================================================
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "scikit-learn>=1.2.0", "joblib>=1.2.0"]
)
def model_training(
    train_data_input: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> NamedTuple('TrainingOutputs', [('train_score', float)]):
    """
    Train a Random Forest Regressor on the training data.
    
    This component trains a Random Forest model for regression tasks.
    The model predicts continuous target values (housing prices).
    
    Args:
        train_data_input: Input training dataset (preprocessed and scaled)
        model_output: Output path where the trained model will be saved as .pkl file
        n_estimators: Number of trees in the random forest (default: 100)
        max_depth: Maximum depth of each tree to prevent overfitting (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        TrainingOutputs: Named tuple containing:
            - train_score (float): R² score on training data (measures goodness of fit)
    
    Inputs Explained:
        - train_data_input: CSV file with scaled features and target column
          Passed automatically from the preprocessing component's output
        - Hyperparameters: Exposed as function arguments for easy tuning
    
    Outputs Explained:
        - model_output: Serialized model file (.pkl) using joblib
          This artifact is passed to the evaluation component
          Contains the trained RandomForestRegressor object
        - train_score: Scalar metric returned for monitoring
          Used to detect overfitting when compared with test score
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    # Load training data
    train_df = pd.read_csv(train_data_input.path)
    X_train = train_df.iloc[:, :-1].values  # All columns except last
    y_train = train_df.iloc[:, -1].values   # Last column (target)
    
    print(f"Training with {len(X_train)} samples")
    print(f"Features shape: {X_train.shape}")
    print(f"Model parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Calculate training score (R² for regression)
    train_score = model.score(X_train, y_train)
    print(f"Training R² score: {train_score:.4f}")
    
    # Save the model using joblib (efficient for sklearn models)
    joblib.dump(model, model_output.path)
    print(f"Model saved to {model_output.path}")
    
    from collections import namedtuple
    output = namedtuple('TrainingOutputs', ['train_score'])
    return output(float(train_score))


# ==============================================================================
# Component 4: Model Evaluation
# ==============================================================================
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "scikit-learn>=1.2.0", "joblib>=1.2.0"]
)
def model_evaluation(
    test_data_input: Input[Dataset],
    model_input: Input[Model],
    metrics_output: Output[Metrics]
) -> NamedTuple('EvaluationOutputs', [('r2_score', float), ('mse', float), ('rmse', float)]):
    """
    Evaluate the trained model on test data and save metrics.
    
    Args:
        test_data_input: Input test dataset
        model_input: Input trained model
        metrics_output: Output path for metrics file
        
    Returns:
        EvaluationOutputs: Named tuple containing evaluation metrics:
            - r2_score (float): R² score (coefficient of determination)
            - mse (float): Mean Squared Error
            - rmse (float): Root Mean Squared Error
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import r2_score, mean_squared_error
    import json
    import math
    
    # Load test data
    test_df = pd.read_csv(test_data_input.path)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    # Load the trained model
    model = joblib.load(model_input.path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    
    print(f"Evaluation Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Save metrics to JSON file
    metrics_dict = {
        "r2_score": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "n_test_samples": len(y_test)
    }
    
    # Write to metrics output
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Metrics saved to {metrics_output.path}")
    
    from collections import namedtuple
    output = namedtuple('EvaluationOutputs', ['r2_score', 'mse', 'rmse'])
    return output(float(r2), float(mse), float(rmse))


# ==============================================================================
# NOTE: Component Compilation
# ==============================================================================
# These components are defined with the @component decorator (KFP v2 style).
# To compile them to YAML files as required by the assignment, run:
#     python compile_components.py
# 
# This script uses kfp.components.create_component_from_func to generate
# the YAML specifications in the components/ directory.