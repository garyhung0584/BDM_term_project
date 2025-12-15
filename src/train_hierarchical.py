import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


def prepare_stage1_data(X, y):
    """
    Stage 1 Goal: Classify Healthy (0) vs At-Risk (1: Pre-diabetic, 2: Diabetic).
    New Target: 0 -> 0, 1 -> 1, 2 -> 1.
    """
    y_stage1 = y.copy()
    y_stage1 = y_stage1.replace({2: 1})
    print(f"Stage 1 Data Prepared. Classes: {y_stage1.unique()}")
    return X, y_stage1


def prepare_stage2_data(X, y):
    """
    Stage 2 Goal: Classify Pre-diabetic (1) vs Diabetic (2).
    Filter: Keep only rows where label is 1 or 2.
    """
    mask = y.isin([1, 2])
    X_stage2 = X[mask]
    y_stage2 = y[mask]
    print(
        f"Stage 2 Data Prepared. Shape: {X_stage2.shape}. Classes: {y_stage2.unique()}"
    )
    return X_stage2, y_stage2


def train_stage_model(X, y, stage_name="Stage"):
    """
    Trains a BalancedRandomForestClassifier with GridSearchCV.
    """
    print(f"\n--- Training {stage_name} Model ---")

    clf = BalancedRandomForestClassifier(
        random_state=42, sampling_strategy="all", replacement=True
    )

    # Extensive Hyperparameter Grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    # Use StratifiedKFold for robust tuning
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Optimize for Recall (Sensitivity) - usually most important in medical screening
    # using f1_macro to balance both classes in the binary split
    grid_search = GridSearchCV(
        clf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
    )

    grid_search.fit(X, y)

    print(f"Best Parameters for {stage_name}: {grid_search.best_params_}")
    print(f"Best Score for {stage_name}: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


class HierarchicalClassifier:
    def __init__(self, stage1_model, stage2_model):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model

    def predict(self, X):
        # Stage 1 Prediction: 0 (Healthy) vs 1 (At Risk)
        # We need to handle X as numpy array or dataframe
        pred_stage1 = self.stage1_model.predict(X)

        # Initialize final predictions with Stage 1 results
        # If Stage 1 says 0 (Healthy), final is 0.
        # If Stage 1 says 1 (At Risk), we need Stage 2 to decide if it's 1 (Pre) or 2 (Diabetic).
        # NOTE: Our Stage 2 model treats classes as [1, 2] directly if trained that way.

        final_preds = pred_stage1.copy()

        # Identify indices where Stage 1 predicts 'At Risk' (1)
        at_risk_mask = pred_stage1 == 1

        if np.any(at_risk_mask):
            X_at_risk = (
                X[at_risk_mask] if isinstance(X, pd.DataFrame) else X[at_risk_mask]
            )
            if len(X_at_risk) > 0:
                pred_stage2 = self.stage2_model.predict(X_at_risk)
                # Assign Stage 2 predictions to correct indices
                final_preds[at_risk_mask] = pred_stage2

        return final_preds


def save_hierarchical_model(model, scaler, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(
        {"stage1": model.stage1_model, "stage2": model.stage2_model, "scaler": scaler},
        filepath,
    )
    print(f"Hierarchical model saved to {filepath}")


def load_hierarchical_model(filepath):
    data = joblib.load(filepath)
    return HierarchicalClassifier(data["stage1"], data["stage2"]), data["scaler"]
