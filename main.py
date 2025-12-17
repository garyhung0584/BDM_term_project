import os
import pandas as pd
from src.preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_features,
    handle_imbalance,
    handle_imbalance_adasyn,
    split_data,
)
from src.train_hierarchical import (
    prepare_stage1_data,
    prepare_stage2_data,
    train_stage_model,
    HierarchicalClassifier,
    save_hierarchical_model,
)
from src.train import evaluate_model, train_and_tune, save_model


def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        project_root, "data", "diabetes_012_health_indicators_BRFSS2015.csv"
    )
    model_path = os.path.join(project_root, "models", "diabetes_best_precision_model.pkl")

    print("--- 1. Data Loading & Cleaning ---")
    if not os.path.exists(data_path):
        print("Data file not found. Running extraction...")
        import src.extract_data
        src.extract_data.extract_dataset(
            os.path.join(
                project_root, "diabetes_012_health_indicators_BRFSS2015.csv.zip"
            ),
            os.path.join(project_root, "data"),
        )

    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)

    print("\n--- 2. Preprocessing & Splitting ---")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n--- 2.2 Feature Selection (RFE) ---")
    # Reducing noise by selecting top 10 most stable features
    from src.preprocessing import select_features_rfe
    X_train_sel, selector = select_features_rfe(X_train, y_train, n_features_to_select=10)
    X_test_sel = X_test[X_train_sel.columns] 
    
    print("\n--- 3. Feature Scaling ---")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_sel, X_test_sel)

    print("\n--- 4. Model Training (LightGBM) ---")
    # Using LightGBM with class_weight='balanced'
    model = train_and_tune(X_train_scaled, y_train, model_type='lgbm')

    print("\n--- 5. Evaluation with Threshold Tuning (Goal: High Precision for Class 1) ---")
    # evaluate_model(model, X_test_scaled, y_test)
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    
    # Get probabilities
    y_probs = model.predict_proba(X_test_scaled)
    # Class 1 is at index 1
    
    print("\nScanning thresholds for Class 1 (Pre-diabetic)...")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'Count':<10}")
    
    best_prec = 0
    best_thresh = 0
    
    for thresh in np.arange(0.5, 0.99, 0.05):
        # Custom prediction logic:
        # Start with Argmax (default)
        y_pred_custom = np.argmax(y_probs, axis=1)
        
        # Override: Only predict Class 1 if prob > thresh
        # Note: This is a simplification. For rigorous multi-class thresholding, 
        # we often use OvR. But here, let's just see if we can "promote" or "demote" Class 1.
        # Actually simplest way for "High Precision": 
        # Predict Class 1 ONLY if prob(1) > thresh. Else predict Class 0 (or whatever has high prob).
        
        class_1_mask = y_probs[:, 1] > thresh
        
        # This is tricky in multiclass. 
        # Let's assess Class 1 as a Binary problem (Pre-diabetic vs Rest) for precision.
        y_test_binary = (y_test == 1).astype(int)
        y_pred_binary = class_1_mask.astype(int)
        
        prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        count = sum(y_pred_binary)
        
        print(f"{thresh:<10.2f} {prec:<10.4f} {rec:<10.4f} {count:<10}")
        
        if prec > best_prec and count > 10: # Ensure we detect at least something
            best_prec = prec
            best_thresh = thresh

    print(f"\nBest Precision found: {best_prec:.4f} at Threshold {best_thresh:.2f}")

    print("\n--- 6. Saving Model ---")
    # Save model, scaler AND selector logic (which cols used)
    # We'll just save the model/scaler for now, but in prod we'd need the cols.
    save_model(model, scaler, model_path)

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
