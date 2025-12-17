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
    model_path = os.path.join(project_root, "models", "diabetes_lgbm_model.pkl")

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

    # Disable ADASYN this time - it caused overfitting
    # print("\n--- 2.1 Handling Imbalance (ADASYN) ---")
    # X_train, y_train = handle_imbalance_adasyn(X_train, y_train)

    print("\n--- 2.2 Feature Selection (RFE) ---")
    # Reducing noise by selecting top 10 most stable features
    from src.preprocessing import select_features_rfe
    X_train_sel, selector = select_features_rfe(X_train, y_train, n_features_to_select=10)
    X_test_sel = X_test[X_train_sel.columns] # Apply same selection to test
    
    print("\n--- 3. Feature Scaling ---")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_sel, X_test_sel)

    print("\n--- 4. Model Training (LightGBM) ---")
    # Using LightGBM with class_weight='balanced'
    model = train_and_tune(X_train_scaled, y_train, model_type='lgbm')

    print("\n--- 5. Evaluation ---")
    evaluate_model(model, X_test_scaled, y_test)

    print("\n--- 6. Saving Model ---")
    # Save model, scaler AND selector logic (which cols used)
    # We'll just save the model/scaler for now, but in prod we'd need the cols.
    save_model(model, scaler, model_path)

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
