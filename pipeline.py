import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math
import os
import shutil
# Ensure data directory exists
os.makedirs("data", exist_ok=True)
def data_extraction(output_path):
    print("Step 1: Data Extraction")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv(output_path, index=False)
    print(f"  Data saved to {output_path}")
def data_preprocessing(input_path, train_path, test_path, test_size, random_state):
    print("Step 2: Data Preprocessing")
    df = pd.read_csv(input_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save train
    train_df = pd.DataFrame(X_train_scaled)
    train_df['target'] = y_train
    train_df.to_csv(train_path, index=False)
    
    # Save test
    test_df = pd.DataFrame(X_test_scaled)
    test_df['target'] = y_test
    test_df.to_csv(test_path, index=False)
    
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
def model_training(train_path, n_estimators, max_depth, random_state):
    print("Step 3: Model Training")
    train_df = pd.read_csv(train_path)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    print(f"  Training R2: {train_score:.4f}")
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
    
    return model
def model_evaluation(model, test_path):
    print("Step 4: Model Evaluation")
    test_df = pd.read_csv(test_path)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    
    print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Log metrics to MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
def main():
    # Define parameters
    params = {
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10
    }
    
    # Set experiment name
    mlflow.set_experiment("California_Housing_Pipeline")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Define paths
        raw_data_path = "data/raw_data.csv"
        train_data_path = "data/train_data.csv"
        test_data_path = "data/test_data.csv"
        
        # Execute Pipeline Steps
        data_extraction(raw_data_path)
        
        data_preprocessing(
            raw_data_path, 
            train_data_path, 
            test_data_path, 
            params["test_size"], 
            params["random_state"]
        )
        
        model = model_training(
            train_data_path, 
            params["n_estimators"], 
            params["max_depth"], 
            params["random_state"]
        )
        
        model_evaluation(model, test_data_path)
        
        print("Pipeline run completed successfully.")
if __name__ == "__main__":
    main()