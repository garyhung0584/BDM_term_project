import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import os
from imblearn.ensemble import BalancedRandomForestClassifier

def train_and_tune(X_train, y_train, model_type='rf'):
    """
    Trains a model with hyperparameter tuning using GridSearchCV.
    """
    if model_type == 'rf':
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'brf':
        # Balanced Random Forest: Undersamples each bootstrap to balance classes
        clf = BalancedRandomForestClassifier(random_state=42, sampling_strategy="all", replacement=True)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'xgb':
        # Check for GPU
        try:
            import xgboost as xgb
            # Attempt to use GPU. XGBoost 3.0+ uses 'device' parameter.
            clf = xgb.XGBClassifier(random_state=42, device='cuda', eval_metric='mlogloss')
            print("XGBoost configured to use GPU.")
        except Exception as e:
            print(f"Warning: Could not configure GPU for XGBoost: {e}. Falling back to CPU.")
            clf = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 8]
        }
    elif model_type == 'lgbm':
        import lightgbm as lgb
        print("Using LightGBM with class_weight='balanced'...")
        # 'balanced' automatically adjusts weights inversely proportional to class frequencies
        clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced', objective='multiclass', n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50],  # Controls complexity
            'min_child_samples': [20, 50] # Helps prevent overfitting
        }
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        print("Using MLPClassifier (Neural Network)...")
        # MLP doesn't support class_weight directly in sklearn < 1.6 in a simple way for all solvers, 
        # but adam/sgd works. However, sklearn's MLP doesn't have a class_weight param. 
        # We handle imbalance via preprocessing (oversampling/undersampling) or we rely on the network learning it.
        # Ideally we use the balanced dataset for this.
        clf = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01], # L2 penalty (regularization term) parameter
            'learning_rate_init': [0.001, 0.01]
        }
    else:
        raise ValueError("Unknown model type. Use 'rf', 'brf', 'xgb', or 'lgbm'.")

    print(f"Starting Grid Search for {model_type}...")
    # Using f1_macro to prioritize minority classes (Pre-diabetic)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for {model_type}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints metrics.
    """
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    return y_pred

def save_model(model, scaler, filepath):
    """
    Saves the trained model and scaler to a file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, filepath)
    print(f"Model and scaler saved to {filepath}")
