# Analysis of `lgbm-predicts-diabetes.ipynb`

## 1. Objective
The notebook aims to predict the likelihood of diabetes using a LightGBM classification model. The project utilizes the "playground-series-s5e12" dataset and an external "diabetes_dataset.csv".

## 2. Data Preprocessing

### 2.1 Feature Engineering
The notebook explicitly engineers new features by interacting existing numerical features. Examples of these engineered features include:
*   `pysicaL_activity_*_sleep_hours🧮`: Product of physical activity and sleep hours.
*   `sleep_hours_per_day_*_sleep_hours🧮`
*   `bmi_*_diet_score🧮`
*   `diastolic_*_sistolic🧮`
*   `bmi_*_diastolic_bp`
*   `diastolic_bp-systolic_bp_*_bmi`

### 2.2 Categorical Mapping
*   Mappings for features like `gender`, `education_level`, `smoking_status`, etc., are defined.
*   **Note**: There is a flag `use_cat_mapping = False` which currently disables manual integer mapping, favoring the pipeline encoder instead.

### 2.3 Outlier Handling
*   **Method**: Interquartile Range (IQR).
*   **Logic**: Values outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ are clipped to the boundary values rather than being removed.
*   **Scope**: Applied to numerical features in both training (`tr_00`) and test (`ts_00`) sets.

## 3. Model Preparation

### 3.1 Data Splitting
*   The training data `tr_01` is split into training and validation sets:
    *   `test_size`: 0.2 (20% validation)
    *   `random_state`: Defined by seed.

### 3.2 Preprocessing Pipeline
*   **Categorical Features**: Processed using `category_encoders.HashingEncoder`.
*   **Numerical Features**: Passed through using `remainder='passthrough'`.
*   **ColumnTransformer**: Combines these steps into a `preprocessor` object.

## 4. Hyperparameter Tuning & Training

### 4.1 Optuna
*   An `objective` function is defined to optimize `LGBMClassifier` hyperparameters.
*   **Search Space**: Includes `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `max_bin`.
*   **Current State**: `n_trials` is set to 0. The notebook currently uses a hardcoded set of "Best Params":
    *   `n_estimators`: 960
    *   `learning_rate`: ~0.502
    *   `max_depth`: 2
    *   `num_leaves`: 52
    *   etc.

### 4.2 Model Pipeline
*   A `sklearn.pipeline.Pipeline` named `lgb_pipe` is constructed containing steps: `['preprocessor', 'estimator']`.
*   The model is fitted on the full training dataset (`X`, `y`) for the final feature importance listing.

## 5. Feature Importance
*   **Metrics**: The notebook extracts both **Split** importances (times a feature is used to split) and **Gain** importances (total gain of splits which use the feature).
*   **Output**: A bar chart visualizing feature importances sorted by Gain.

## 6. Missing Components / Next Steps
Based on the current state of the notebook (up to line 4071), the following sections appear to be missing or were not reached:
1.  **Explicit Model Evaluation**: Calculating metrics (ROC-AUC, Accuracy) on the validation set [`X_valid`, `y_valid`] to confirm performance before submission.
2.  **Prediction**: Generating predictions on the test set (`ts_01`).
3.  **Submission**: Creating the `submission.csv` file formatted according to `sb_00` (sample submission).
