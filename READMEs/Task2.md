# Task 2: Building Kubeflow Pipeline Components - Complete README

## Overview

Task 2 involves creating four reusable Kubeflow pipeline components that form the building blocks of our machine learning pipeline. Each component is a self-contained, containerized unit that performs a specific task in the ML workflow.

**What are Kubeflow Components?**
- Self-contained Python functions that run in isolated Docker containers
- Accept typed inputs (datasets, models, parameters)
- Produce typed outputs (artifacts, metrics)
- Can be reused across different pipelines
- Enable parallel execution and scalability

---

## Assignment Requirements

The assignment asks us to:

1. ✅ **Write Python functions** in `src/pipeline_components.py` for four ML pipeline steps
2. ✅ **Use the `kfp.dsl.component` decorator** to define components with inputs/outputs
3. ✅ **Compile components to YAML files** in the `components/` directory using `kfp.components.create_component_from_func` method

### Important Note on KFP Versions

The assignment mentions both:
- `@kfp.dsl.component` decorator (KFP v2 approach)
- `kfp.components.create_component_from_func` method (KFP v1 approach)

**In KFP v2** (which we're using per `requirements.txt`), the `@component` decorator internally calls `create_component_from_func`. So using the decorator satisfies both requirements. The compilation script then saves these decorated components to YAML files.

---

## The Four Components

### Component 1: Data Extraction

**Purpose**: Load the California Housing dataset for our regression task

**File Location**: `src/pipeline_components.py` (lines 15-45)

**Function Signature**:
```python
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "numpy>=1.23.0", "scikit-learn>=1.2.0"]
)
def data_extraction(
    dataset_output: Output[Dataset]
):
```

**Inputs**:
- None (uses scikit-learn's built-in dataset)

**Outputs**:
- `dataset_output` (Type: `Output[Dataset]`): CSV file containing the California Housing dataset with 20,640 samples and 8 features plus target column

**What It Does**:
1. Loads California Housing dataset from scikit-learn
2. Creates a pandas DataFrame with feature columns and target
3. Saves to CSV at the path specified by `dataset_output.path`
4. Prints dataset shape and feature names for verification

**Why California Housing?**
- Boston Housing dataset was deprecated in scikit-learn
- California Housing is similar: regression task predicting house prices
- Contains 20,640 samples with 8 features (MedInc, HouseAge, AveRooms, etc.)

**DVC Integration Note**:
While Task 1 demonstrates DVC data versioning with local storage, this component uses scikit-learn's dataset loader for Kubernetes compatibility. In production, this would use `dvc get` with cloud storage (S3/GCS/Azure) accessible from containers.

---

### Component 2: Data Preprocessing

**Purpose**: Clean, scale, and split data into training and test sets

**File Location**: `src/pipeline_components.py` (lines 50-115)

**Function Signature**:
```python
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
```

**Inputs**:
- `dataset_input` (Type: `Input[Dataset]`): Raw dataset CSV from the extraction component
- `test_size` (Type: `float`, Default: 0.2): Proportion of data to use for testing (20%)
- `random_state` (Type: `int`, Default: 42): Random seed for reproducible train/test splits

**Outputs**:
- `train_data_output` (Type: `Output[Dataset]`): Preprocessed training data CSV with scaled features
- `test_data_output` (Type: `Output[Dataset]`): Preprocessed test data CSV with scaled features
- `n_train_samples` (Type: `int`): Number of training samples (returned as scalar)
- `n_test_samples` (Type: `int`): Number of test samples (returned as scalar)

**What It Does**:
1. **Load**: Reads the raw dataset from `dataset_input.path`
2. **Clean**: Removes any rows with missing values using `dropna()`
3. **Split Features/Target**: Separates features (all columns except last) from target (last column)
4. **Train/Test Split**: Uses `train_test_split()` with specified `test_size` and `random_state`
5. **Scale Features**: 
   - Fits `StandardScaler` on training data only (prevents data leakage)
   - Transforms both train and test data using the fitted scaler
   - Standardization: (X - mean) / std_dev
6. **Save Artifacts**:
   - Saves scaled training features + target to `train_data_output.path`
   - Saves scaled test features + target to `test_data_output.path`
   - Also saves the scaler object as a pickle file (for future use)
7. **Return Metrics**: Returns the count of train/test samples

**Why StandardScaler?**
- Neural networks and many ML algorithms work better with standardized inputs
- Ensures all features have mean=0 and std=1
- Prevents features with large ranges from dominating the model

**Pipeline Flow**:
```
dataset_input (raw CSV) 
    → Clean → Split → Scale 
    → train_data_output, test_data_output
```

---

### Component 3: Model Training

**Purpose**: Train a Random Forest Regressor on preprocessed training data

**File Location**: `src/pipeline_components.py` (lines 120-190)

**Function Signature**:
```python
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
```

**Inputs**:

1. **`train_data_input`** (Type: `Input[Dataset]`)
   - **Description**: Preprocessed training dataset from the preprocessing component
   - **Format**: CSV file with scaled features (StandardScaler applied) and target column
   - **Content**: Approximately 16,512 samples (80% of 20,640) with 8 feature columns + 1 target column
   - **How It's Passed**: Kubeflow automatically downloads this artifact from the previous component and makes it available at `train_data_input.path`
   - **Data Flow**: `data_preprocessing.train_data_output` → `model_training.train_data_input`

2. **`n_estimators`** (Type: `int`, Default: 100)
   - **Description**: Number of decision trees in the Random Forest ensemble
   - **Purpose**: Controls model complexity and reduces variance
   - **Trade-off**: 
     - More trees = Better performance but slower training
     - Typical range: 50-500
   - **Tunable**: Yes, can be changed when calling the component in the pipeline
   - **Example**: `model_training(n_estimators=200)` would use 200 trees

3. **`max_depth`** (Type: `int`, Default: 10)
   - **Description**: Maximum depth of each decision tree
   - **Purpose**: Prevents overfitting by limiting how deeply trees can split
   - **Impact**: 
     - Deeper trees can capture more complex patterns but may overfit
     - Shallower trees are more generalizable but may underfit
   - **Default Rationale**: Depth of 10 balances complexity and generalization
   - **Tunable**: Yes, experiment with values between 5-20

4. **`random_state`** (Type: `int`, Default: 42)
   - **Description**: Random seed for reproducibility
   - **Purpose**: Ensures identical results across multiple runs
   - **Why 42**: Common convention in data science (Hitchhiker's Guide reference)
   - **Impact**: Controls random aspects like bootstrap sampling and feature selection

**Outputs**:

1. **`model_output`** (Type: `Output[Model]`)
   - **Description**: Serialized trained RandomForestRegressor model
   - **Format**: `.pkl` file created using `joblib.dump()`
   - **File Size**: Typically 5-50MB depending on `n_estimators` and `max_depth`
   - **Contents**: 
     - Complete trained model with all learned parameters
     - 100 decision trees (if n_estimators=100)
     - Feature importances for each of the 8 features
     - Tree structures with split points and leaf values
   - **Serialization Method**: `joblib` (more efficient than pickle for large numpy arrays)
   - **Usage**: Automatically passed to downstream components (evaluation) via KFP's artifact system
   - **Access Path**: `model_output.path` provides the file path where model should be saved
   - **Example Path**: `/tmp/outputs/model/data` (Kubeflow manages the actual path)

2. **`train_score`** (Type: `float`, Returned Value)
   - **Description**: R² (coefficient of determination) score on training data
   - **Formula**: R² = 1 - (SS_res / SS_tot)
     - SS_res = Sum of squared residuals
     - SS_tot = Total sum of squares
   - **Range**: -∞ to 1.0
     - 1.0 = Perfect fit (predictions exactly match actual values)
     - 0.0 = Model performs as well as predicting the mean
     - < 0.0 = Model performs worse than predicting the mean
   - **Purpose**: 
     - Monitoring metric to assess training performance
     - Compare with test R² to detect overfitting
     - If train_score >> test_score, model is overfitting
   - **How It's Used**: 
     - Logged in Kubeflow Pipelines UI
     - Can be used for hyperparameter tuning
     - Enables comparison across multiple pipeline runs
   - **Typical Value**: For this dataset, expect 0.75-0.85 on training data

**What It Does**:
1. **Load Training Data**: Reads CSV from `train_data_input.path`
2. **Separate Features/Target**: 
   - X_train = all columns except last (8 features)
   - y_train = last column (house prices)
3. **Initialize Model**: Creates `RandomForestRegressor` with specified hyperparameters
4. **Train**: Calls `model.fit(X_train, y_train)`
   - Each tree is trained on a bootstrap sample
   - At each split, considers random subset of features
   - Grows trees to `max_depth`
5. **Calculate Training Score**: Uses `model.score()` which computes R²
6. **Serialize Model**: Saves trained model using `joblib.dump(model, model_output.path)`
7. **Return Metric**: Returns training R² score as a named tuple

**Why Random Forest Regressor?**
- Ensemble method: combines predictions from multiple decision trees
- Robust to overfitting due to averaging
- Handles non-linear relationships well
- No need for feature scaling (but we did it anyway for consistency)
- Provides feature importances
- Works well for regression tasks like house price prediction

**Model Architecture**:
```
Random Forest (n_estimators=100)
├── Tree 1 (max_depth=10)
│   ├── Split on MedInc > 3.5
│   ├── Split on HouseAge > 25
│   └── ... (up to 10 levels)
├── Tree 2 (max_depth=10)
├── ...
└── Tree 100 (max_depth=10)
    
Final Prediction = Average of all 100 tree predictions
```

**Training Process Timeline**:
1. Load data: ~0.1 seconds
2. Model initialization: ~0.01 seconds
3. Training: ~5-30 seconds (depends on n_estimators, dataset size)
4. Score calculation: ~1 second
5. Model serialization: ~1-2 seconds

**Why These Specific Inputs/Outputs?**

**Design Rationale**:
- **Typed Artifacts**: Using `Input[Dataset]` and `Output[Model]` provides:
  - Type safety and validation
  - Automatic artifact management by Kubeflow
  - Clear contracts between components
  - Metadata tracking (size, creation time, lineage)

- **Parameterization**: Exposing hyperparameters as function arguments allows:
  - Experimentation without code changes
  - Hyperparameter tuning in the pipeline
  - Easy A/B testing of different configurations
  - Pipeline reusability for different scenarios

- **Model Persistence**: Using `joblib` ensures:
  - Efficient serialization of scikit-learn models
  - Faster load times compared to pickle
  - Better compression for large numpy arrays
  - Industry standard for sklearn models

- **Metrics as Returns**: Returning scalar metrics allows:
  - Kubeflow UI to display them for comparison
  - Decision-making in conditional pipeline execution
  - Tracking model performance across runs
  - No need for separate metrics files

**Example Usage in Pipeline**:
```python
train_task = model_training(
    train_data_input=preprocess_task.outputs['train_data_output'],
    n_estimators=150,  # Custom value
    max_depth=12,      # Custom value
    random_state=42
)

# Access outputs
trained_model = train_task.outputs['model_output']
training_r2 = train_task.outputs['train_score']
```

**Common Issues & Solutions**:

1. **Out of Memory**: 
   - Reduce `n_estimators` or `max_depth`
   - Use `max_features` parameter to limit features per split

2. **Training Too Slow**:
   - Already using `n_jobs=-1` (all CPU cores)
   - Reduce `n_estimators`

3. **Overfitting** (train_score much higher than test_score):
   - Reduce `max_depth`
   - Reduce `n_estimators`
   - Increase `min_samples_split` (not exposed but could be)

**Pipeline Data Flow**:
```
train_data_input (CSV with ~16K rows, 9 columns)
    ↓
Load & Parse
    ↓
Separate X (8 features) and y (1 target)
    ↓
Initialize RandomForest(n_estimators=100, max_depth=10)
    ↓
Train (fit 100 decision trees)
    ↓
Calculate R² score
    ↓
Serialize model → model_output (joblib .pkl file, ~20MB)
    ↓
Return train_score (float, e.g., 0.82)
```

---

### Component 4: Model Evaluation

**Purpose**: Evaluate the trained model on held-out test data and compute performance metrics

**File Location**: `src/pipeline_components.py` (lines 195-250)

**Function Signature**:
```python
@component(
    base_image="python:3.9",
    packages_to_install=["pandas>=1.5.0", "scikit-learn>=1.2.0", "joblib>=1.2.0"]
)
def model_evaluation(
    test_data_input: Input[Dataset],
    model_input: Input[Model],
    metrics_output: Output[Metrics]
) -> NamedTuple('EvaluationOutputs', [('r2_score', float), ('mse', float), ('rmse', float)]):
```

**Inputs**:
- `test_data_input` (Type: `Input[Dataset]`): Preprocessed test dataset from preprocessing component
- `model_input` (Type: `Input[Model]`): Trained model artifact from training component

**Outputs**:
- `metrics_output` (Type: `Output[Metrics]`): JSON file containing all evaluation metrics
- `r2_score` (Type: `float`): R² score on test data (returned value)
- `mse` (Type: `float`): Mean Squared Error (returned value)
- `rmse` (Type: `float`): Root Mean Squared Error (returned value)

**What It Does**:
1. **Load Test Data**: Reads preprocessed test CSV
2. **Load Model**: Deserializes the trained model using `joblib.load()`
3. **Make Predictions**: `y_pred = model.predict(X_test)`
4. **Calculate Metrics**:
   - **R² Score**: Coefficient of determination (1.0 = perfect, 0.0 = baseline)
   - **MSE**: Mean Squared Error - average of squared differences
   - **RMSE**: Root Mean Squared Error - square root of MSE (same units as target)
5. **Save Metrics**: Writes JSON file with all metrics + test sample count
6. **Return Values**: Returns the three metrics as a named tuple

**Metrics Explained**:

1. **R² Score** (Coefficient of Determination)
   - **Formula**: R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
   - **Interpretation**: Proportion of variance in target explained by model
   - **Range**: -∞ to 1.0 (typically 0.0 to 1.0 for reasonable models)
   - **Good Values**: 
     - > 0.7 = Good model
     - 0.5-0.7 = Decent model
     - < 0.5 = Poor model
   - **Why It Matters**: Most intuitive metric for regression; easy to explain

2. **MSE** (Mean Squared Error)
   - **Formula**: MSE = (1/n) × Σ(y_true - y_pred)²
   - **Units**: Squared units of target (e.g., dollars²)
   - **Interpretation**: Average squared difference between predictions and actual values
   - **Sensitivity**: Heavily penalizes large errors due to squaring
   - **Why It Matters**: Commonly used loss function; mathematical properties for optimization

3. **RMSE** (Root Mean Squared Error)
   - **Formula**: RMSE = √MSE
   - **Units**: Same units as target (e.g., dollars)
   - **Interpretation**: Average magnitude of prediction errors
   - **Example**: RMSE = $50,000 means predictions are off by ~$50K on average
   - **Why It Matters**: More interpretable than MSE; in original units

**Example Metrics Output** (metrics.json):
```json
{
  "r2_score": 0.7834,
  "mse": 0.5432,
  "rmse": 0.7371,
  "n_test_samples": 4128
}
```

**Interpretation**: 
- Model explains 78.34% of variance in house prices
- Predictions are off by ~0.74 (in scaled units) on average
- Evaluated on 4,128 test samples

---

## Component Compilation Process

### Understanding the Decorator

The `@component` decorator does several things:

1. **Containerization**: Wraps the Python function to run in a Docker container
2. **Dependency Management**: Installs specified packages at runtime
3. **Type Validation**: Checks that inputs/outputs match the function signature
4. **Artifact Management**: Handles file paths for inputs/outputs automatically
5. **Metadata Tracking**: Records execution info, runtime, resource usage

**Example**:
```python
@component(
    base_image="python:3.9",  # Base Docker image
    packages_to_install=["pandas>=1.5.0", "scikit-learn>=1.2.0"]  # Dependencies
)
def my_component(
    input_data: Input[Dataset],  # Typed input
    output_model: Output[Model],  # Typed output
    learning_rate: float = 0.01   # Parameter
):
    # Your code here
    pass
```

### Compilation Script

**File**: `compile_components.py` (in project root)

**Purpose**: Converts decorated Python functions into YAML specifications that Kubeflow can execute

**How to Run**:
```bash
# Make sure you're in project root
cd mlops-kubeflow-assignment

# Ensure KFP is installed
pip install kfp==1.8.22

# Install lib2to3 if using Python 3.12+
pip install lib2to3-backport

# Run compilation
python compile_components.py
```

**What It Does**:
1. Imports all four component functions from `src/pipeline_components.py`
2. Uses `components.create_component_from_func()` to compile each one
3. Generates YAML files in `components/` directory
4. Prints success/failure for each component

**Expected Output**:
```
Compiling with kfp.components.create_component_from_func (KFP v1)...

✓ data_extraction -> components/data_extraction.yaml
✓ data_preprocessing -> components/data_preprocessing.yaml
✓ model_training -> components/model_training.yaml
✓ model_evaluation -> components/model_evaluation.yaml

======================================================================
✓ All components compiled using create_component_from_func!
✓ YAML files saved in: D:\...\mlops-kubeflow-assignment\components
======================================================================
```

### Generated YAML Files

Each YAML file contains:

1. **Component Metadata**:
   - Name and description
   - Version information
   
2. **Input Specifications**:
   - Name, type, and description of each input
   - Default values for optional parameters
   
3. **Output Specifications**:
   - Name, type, and path for each output artifact
   
4. **Implementation**:
   - Docker image to use
   - Packages to install
   - The Python code to execute
   - Command-line arguments mapping

**Example YAML Structure** (simplified):
```yaml
name: Model training
description: Train a Random Forest Regressor
inputs:
  - {name: train_data_input, type: Dataset}
  - {name: n_estimators, type: Integer, default: '100'}
  - {name: max_depth, type: Integer, default: '10'}
outputs:
  - {name: model_output, type: Model}
  - {name: train_score, type: Float}
implementation:
  container:
    image: python:3.9
    command: [python3, -u, -c]
    args:
      - |
        # Component implementation code here
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        # ... rest of the function code
```

### Verification

After compilation, verify the YAML files:

```bash
# List generated files
ls -la components/

# Expected output:
# data_extraction.yaml
# data_preprocessing.yaml
# model_training.yaml
# model_evaluation.yaml

# Check file size (should be 5-15 KB each)
ls -lh components/

# View a YAML file
cat components/model_training.yaml
```

**What to Look For**:
- ✅ All 4 YAML files present
- ✅ File sizes reasonable (5-15 KB)
- ✅ YAML is valid (no syntax errors)
- ✅ Contains `inputs:`, `outputs:`, and `implementation:` sections

---

## Type System

Kubeflow uses a sophisticated type system for artifacts:

### Built-in Types

1. **Dataset**: For data files (CSV, Parquet, etc.)
   - Use for: Raw data, preprocessed data, train/test splits
   
2. **Model**: For trained ML models
   - Use for: Serialized models (.pkl, .h5, .pt files)
   
3. **Metrics**: For evaluation metrics
   - Use for: JSON files with performance metrics
   
4. **HTML**: For visualizations and reports
   - Use for: Plots, dashboards, HTML reports

### Primitive Types

- `int`, `float`, `str`, `bool`: For simple parameters
- `List[T]`, `Dict[K, V]`: For complex parameters

### Usage Examples

```python
# Input artifact (data flows into component)
dataset_input: Input[Dataset]
# Access: pd.read_csv(dataset_input.path)

# Output artifact (data flows out of component)
model_output: Output[Model]
# Access: joblib.dump(model, model_output.path)

# Input parameter (configurable value)
learning_rate: float = 0.01

# Output value (returned from function)
-> NamedTuple('Outputs', [('accuracy', float)])
```

---

## Project Structure

```
mlops-kubeflow-assignment/
│
├── data/                          # Data directory
│   └── raw_data.csv              # DVC-tracked dataset (from Task 1)
│   └── raw_data.csv.dvc          # DVC metadata file
│
├── src/                          # Source code
│   └── pipeline_components.py    # All 4 component definitions
│
├── components/                   # Compiled component YAMLs
│   ├── data_extraction.yaml      # Generated by compile_components.py
│   ├── data_preprocessing.yaml
│   ├── model_training.yaml
│   └── model_evaluation.yaml
│
├── compile_components.py         # Script to compile components to YAML
├── requirements.txt              # Project dependencies
├── .dvc/                        # DVC configuration (from Task 1)
├── .git/                        # Git repository
└── README.md                    # This file
```

---

## Dependencies

From `requirements.txt`:

```txt
# Core MLOps & Orchestration
kfp==1.8.22                    # Kubeflow Pipelines SDK v1

# Data Versioning (from Task 1)
dvc>=3.0.0

# Data Processing
pandas>=1.5.0                  # DataFrame manipulation
numpy>=1.23.0                  # Numerical operations

# Machine Learning
scikit-learn>=1.2.0            # Random Forest, metrics, preprocessing

# Utilities
joblib>=1.2.0                  # Model serialization
lib2to3-backport               # For Python 3.12+ compatibility with KFP v1
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Task 2 Deliverables

### Deliverable 2.1: Component Functions Screenshot

**What to Submit**: Screenshot of `src/pipeline_components.py` showing at least two component functions

**How to Prepare**:
1. Open `src/pipeline_components.py` in your code editor
2. Scroll to show the `model_training` function (lines ~120-190)
3. Make sure the `@component` decorator and function signature are visible
4. Take a screenshot
5. Alternatively, show `model_evaluation` function as the second one

**What the Evaluator Looks For**:
- ✅ Proper `@component` decorator with `base_image` and `packages_to_install`
- ✅ Typed inputs using `Input[Dataset]`, `Input[Model]`
- ✅ Typed outputs using `Output[Model]`, `Output[Metrics]`
- ✅ Clear parameter definitions with default values
- ✅ Return type annotation with `NamedTuple`
- ✅ Comprehensive docstrings

### Deliverable 2.2: Components Directory Screenshot

**What to Submit**: Screenshot of the `components/` directory showing all YAML files

**How to Prepare**:
```bash
# Method 1: Command line
ls -la components/

# Method 2: File Explorer
# Navigate to components/ folder and take screenshot
```

**Expected Output**:
```
total 52K
-rw-r--r-- 1 user user  8.2K Nov 29 10:30 data_extraction.yaml
-rw-r--r-- 1 user user  12K  Nov 29 10:30 data_preprocessing.yaml
-rw-r--r-- 1 user user  14K  Nov 29 10:30 model_training.yaml
-rw-r--r-- 1 user user  11K  Nov 29 10:30 model_evaluation.yaml
```

**What the Evaluator Looks For**:
- ✅ All 4 YAML files present
- ✅ Reasonable file sizes (5-15 KB each)
- ✅ Recent modification timestamps (showing they were just generated)

### Deliverable 2.3: Training Component I/O Explanation

**What to Submit**: Detailed explanation of inputs and outputs for the `model_training` component

**Complete Explanation**:

#### Training Component - Inputs and Outputs

**Component Purpose**: Train a Random Forest Regressor on preprocessed training data to predict California housing prices.

---

**INPUTS:**

**1. `train_data_input` (Type: `Input[Dataset]`)**

*Description*:
- Preprocessed training dataset passed automatically from the preprocessing component
- Contains scaled features and target values ready for model training

*Format*:
- CSV file with approximately 16,512 rows (80% of 20,640 total samples)
- 9 columns: 8 feature columns (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) + 1 target column (median house value)
- Features are standardized using StandardScaler (mean=0, std=1)

*How It Works*:
- Kubeflow automatically downloads this artifact from the preprocessing component's `train_data_output`
- Made available to the component at the path `train_data_input.path`
- Example path: `/tmp/inputs/train_data/data`

*Data Flow*:
```
data_preprocessing.train_data_output → model_training.train_data_input
```

*Usage in Code*:
```python
train_df = pd.read_csv(train_data_input.path)
X_train = train_df.iloc[:, :-1].values  # 8 features
y_train = train_df.iloc[:, -1].values   # 1 target
```

---

**2. `n_estimators` (Type: `int`, Default: 100)**

*Description*:
- Number of decision trees in the Random Forest ensemble
- Each tree is trained independently on a bootstrap sample of the data

*Purpose*:
- Controls model complexity and reduces variance through averaging
- More trees generally lead to better performance but with diminishing returns

*Trade-offs*:
- **Higher values (150-500)**: Better performance, lower variance, but slower training and larger model size
- **Lower values (50-100)**: Faster training, smaller model, but potentially higher variance
- **Sweet spot**: 100-200 for most use cases

*Impact on Model*:
- Directly affects model file size: ~200KB per tree
- Training time scales linearly with number of trees
- Prediction time also scales linearly

*Tunable*: Yes - can be modified when calling the component in the pipeline

*Example Usage*:
```python
# In pipeline definition
train_task = model_training(
    train_data_input=preprocess_task.outputs['train_data_output'],
    n_estimators=150  # Use 150 trees instead of default 100
)
```

---

**3. `max_depth` (Type: `int`, Default: 10)**

*Description*:
- Maximum depth of each decision tree in the forest
- Limits how many times a tree can split the data

*Purpose*:
- **Primary purpose**: Prevent overfitting by controlling tree complexity
- Deeper trees can capture more complex patterns but risk memorizing training data
- Shallower trees are more generalizable but may miss important patterns

*Impact*:
- **Depth = 1**: Each tree is just a single split (underfitting)
- **Depth = 5**: Moderate complexity, good for simple patterns
- **Depth = 10**: Default - balances complexity and generalization
- **Depth = 20**: High complexity, may overfit
- **Depth = None**: