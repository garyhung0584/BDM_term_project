import os
import pandas as pd
from src.preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_features,
    handle_imbalance,
    split_data,
)
from src.train_hierarchical import (
    prepare_stage1_data,
    prepare_stage2_data,
    train_stage_model,
    HierarchicalClassifier,
    save_hierarchical_model,
)
from src.train import evaluate_model


def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        project_root, "data", "diabetes_012_health_indicators_BRFSS2015.csv"
    )
    model_path = os.path.join(project_root, "models", "diabetes_hierarchical_model.pkl")

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

    # Optional sampling for faster debugging during tuning
    # df = df.sample(frac=0.1, random_state=42)
    # print("Warning: Using 10% sample for speed.")

    print("\n--- 2. Preprocessing & Splitting ---")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n--- 3. Feature Scaling ---")
    # Scale first, then split for stages
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    # Convert back to DataFrame for easier indexing if needed, or keep as array
    # preprocessing returns arrays.

    print("\n--- 4. Hierarchical Training Pipeline ---")

    # Stage 1: Healthy (0) vs At Risk (1/2 becomes 1)
    print("\n[Stage 1] Preparing Data (Healthy vs At-Risk)...")
    # We need to map y_train, which is a Series.
    # Helper functions expect X, y.
    # Note: X_train_scaled is an array.

    # Remap y_train for Stage 1
    y_train_s1 = y_train.replace({2: 1})

    # Train Stage 1
    print("[Stage 1] Training Classifier...")
    model_s1 = train_stage_model(
        X_train_scaled, y_train_s1, stage_name="Stage 1 (Screening)"
    )

    # Stage 2: Pre-diabetic (1) vs Diabetic (2)
    print("\n[Stage 2] Preparing Data (Pre-diabetic vs Diabetic)...")
    mask = y_train.isin([1, 2])
    X_train_s2 = X_train_scaled[mask]
    y_train_s2 = y_train[mask]

    # Train Stage 2
    print("[Stage 2] Training Classifier...")
    model_s2 = train_stage_model(
        X_train_s2, y_train_s2, stage_name="Stage 2 (Diagnostic)"
    )

    # Combine
    hierarchical_model = HierarchicalClassifier(model_s1, model_s2)

    print("\n--- 5. Evaluation ---")
    # Evaluate using the combined predict method
    evaluate_model(hierarchical_model, X_test_scaled, y_test)

    print("\n--- 6. Saving Model ---")
    save_hierarchical_model(hierarchical_model, scaler, model_path)

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
