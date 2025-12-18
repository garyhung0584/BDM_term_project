import os
import pandas as pd
from src.preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_features,
    split_data,
)
from src.train import evaluate_model, train_and_tune, save_model


def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        project_root, "input", "diabetes_012_health_indicators_BRFSS2015.csv"
    )
    model_path = os.path.join(project_root, "models", "diabetes_brf_champion_model.pkl")

    print("--- 1. Data Loading & Cleaning ---")
    if not os.path.exists(data_path):
        print("Data file not found. Running extraction...")
        import src.extract_data
        src.extract_data.extract_dataset(
            os.path.join(
                project_root, "diabetes_012_health_indicators_BRFSS2015.csv.zip"
            ),
            os.path.join(project_root, "input"),
        )

    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)

    print("\n--- 2. Preprocessing & Splitting ---")
    X_train, X_test, y_train, y_test = split_data(df)

    # Champion Model (Experiment 3) uses BalancedRandomForest to handle imbalance internally.
    # We use all features (no RFE) and no synthetic oversampling (no ADASYN) to avoid overfitting.
    
    print("\n--- 3. Feature Scaling ---")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)

    print("\n--- 4. Model Training (Balanced Random Forest) ---")
    # Using Balanced Random Forest (Champion Model)
    model = train_and_tune(X_train_scaled, y_train, model_type='brf')

    print("\n--- 5. Evaluation ---")
    evaluate_model(model, X_test_scaled, y_test)

    print("\n--- 6. Saving Model ---")
    save_model(model, scaler, model_path)

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
