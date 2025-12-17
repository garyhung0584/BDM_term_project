# Diabetes Prediction System - Technical Documentation

## Overview
This project implements a machine learning pipeline to predict diabetes status based on the text CDC BRFSS2015 dataset. It features multiple classification strategies including standard Random Forest, XGBoost with GPU acceleration, and a Hierarchical (Two-Stage) Classifier designed to handle class imbalance.

## Project Structure
```text
.
├── data/                   # Dataset directory (auto-extracted)
├── models/                 # Saved models (.pkl files)
├── notebooks/              # Jupyter notebooks for EDA and Analysis
├── src/                    # Source code modules
│   ├── extract_data.py     # Unzips the raw dataset
│   ├── preprocessing.py    # Cleaning, Feature Engineering, and Balancing (SMOTE/ADASYN)
│   ├── train.py            # Single-stage training logic (RF, XGBoost, BalancedRF)
│   ├── train_hierarchical.py # Two-stage training logic
│   ├── predict.py          # Inference and SHAP explanations
│   ├── create_eda_notebook.py # Generator for EDA notebook
│   └── create_performance_notebook.py # Generator for Analysis notebook
├── main.py                 # Main orchestration script
└── requirements.txt        # Python dependencies
```

## Installation
1.  **Environment**
    Ensure Python 3.8+ is installed.
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # source .venv/bin/activate # Linux/Mac
    ```

2.  **Dependencies**
    Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: XGBoost with GPU support requires appropriate CUDA drivers installed.*

## Usage

### 1. Run the Full Pipeline
The `main.py` script orchestrates the entire process:
- Checks/Extracts Data
- Loads & Cleans Data
- Preprocesses (Adasyn/Scaling)
- Trains Model (Hierarchical by default)
- Evaluates & Saves Model

```bash
python main.py
```

### 2. Generate Analysis Notebooks
To create Jupyter notebooks for interactive analysis:

**Exploratory Data Analysis (EDA):**
```bash
python src/create_eda_notebook.py
```
This creates `notebooks/Diabetes_EDA.ipynb`.

**Model Performance:**
```bash
python src/create_performance_notebook.py
```
This creates `notebooks/Model_Performance.ipynb`.

## Key Modules

### `src.preprocessing`
- `handle_imbalance(X, y)`: Applies SMOTETomek.
- `handle_imbalance_adasyn(X, y)`: Applies ADASYN (Adaptive Synthetic Sampling).
- `clean_data(df)`: Removes duplicates and nulls.

### `src.train_hierarchical`
Implements the `HierarchicalClassifier` class which chains two models:
1.  **Stage 1**: Binary classification for `Healthy (0)` vs `At-Risk (1, 2)`.
2.  **Stage 2**: Binary classification for `Pre-diabetic (1)` vs `Diabetic (2)`, run only on samples predicted as At-Risk.

## Model Output
Models are saved to the `models/` directory using `joblib`.
- `diabetes_model.pkl`: Single-stage models (if trained).
- `diabetes_hierarchical_model.pkl`: Two-stage model (default).

## Troubleshooting
- **Memory Errors**: ADASYN and SMOTE are memory intensive (~250k rows -> ~400k rows). If you crash, reduce n_jobs in GridSearchCV or use `BalancedRandomForest` (undersampling) instead.
- **XGBoost Warning**: "Falling back to prediction using DMatrix" - benign warning about CPU/GPU memory transfer.
