import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_model(filepath):
    """
    Loads the trained model and scaler.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    data = joblib.load(filepath)
    return data['model'], data['scaler']

def predict_sample(model, scaler, input_data):
    """
    Predicts the class for the input data.
    Input data should be a DataFrame or 2D array.
    """
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    return prediction, probabilities

def explain_prediction(model, X_sample, feature_names=None):
    """
    Generates SHAP values for a sample and plots the summary.
    Using TreeExplainer as we expect tree-based models (RF, XGB).
    """
    # Create object that can calculate shap values
    # Fix for XGBoost/SHAP multiclass base_score issue: pass the booster directly
    try:
        explainer = shap.TreeExplainer(model.get_booster())
    except AttributeError:
        # Fallback if model doesn't have get_booster (e.g. Random Forest)
        explainer = shap.TreeExplainer(model)
    
    # Calculate shap values. This is complex for multiclass, so taking the first class or aggregate might be needed
    # For multiclass, shap_values is a list of arrays (one for each class)
    shap_values = explainer.shap_values(X_sample)
    
    # Plotting summary (for Class 1: Pre-diabetic or Class 2: Diabetic often most interesting)
    # If binary, shap_values[1] is usually the positive class.
    # If 3 classes, we can inspect specific class of interest.
    
    # Just printing shape for now to confirm it works in headless env
    print(f"SHAP values calculated. Shape: {len(shap_values)} classes.")
    
    # In a real GUI/Notebook, we would show plots. Here we might save them.
    # shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    # plt.savefig('shap_summary.png')
    
    return shap_values
