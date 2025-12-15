import os
import pandas as pd
from src.preprocessing import load_data, clean_data, feature_engineering, preprocess_features, handle_imbalance, split_data
from src.train import train_and_tune, evaluate_model, save_model
from src.predict import explain_prediction

def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', 'diabetes_012_health_indicators_BRFSS2015.csv')
    model_path = os.path.join(project_root, 'models', 'diabetes_model.pkl')
    
    print("--- 1. Data Loading & Cleaning ---")
    if not os.path.exists(data_path):
        print("Data file not found. Running extraction...")
        # Fallback if manual run skipped
        import src.extract_data
        src.extract_data.extract_dataset(
            os.path.join(project_root, "diabetes_012_health_indicators_BRFSS2015.csv.zip"),
            os.path.join(project_root, "data")
        )
        
    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    
    print("\n--- 2. Preprocessing & Splitting ---")
    # For speed in this demo, maybe sample?
    # df = df.sample(frac=0.1, random_state=42) # Text to self: Uncomment for debugging speed
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("\n--- 3. Handling Imbalance ---")
    print("Using BalancedRandomForestClassifier. Skipping SMOTE data expansion to save time and reduce noise.")
    # For BRF, we pass the original imbalanced data. The model handles balancing internally during bootstrapping.
    X_train_res, y_train_res = X_train, y_train
    
    print("\n--- 4. Scaling Features ---")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_res, X_test)
    
    print("\n--- 5. Model Training (GridSearch) ---")
    # Choosing 'brf' (Balanced Random Forest)
    model = train_and_tune(X_train_scaled, y_train_res, model_type='brf')
    
    print("\n--- 6. Evaluation ---")
    evaluate_model(model, X_test_scaled, y_test)
    
    print("\n--- 7. Saving Model ---")
    save_model(model, scaler, model_path)
    
    print("\n--- 8. Explanation Demo (SHAP) ---")
    # Explain on a small subset of test data
    X_view = X_test_scaled[:5]
    explain_prediction(model, X_view, feature_names=df.drop(columns=['Diabetes_012']).columns)
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
