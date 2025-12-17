import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

def load_data(filepath):
    """
    Loads the diabetes dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(df):
    """
    Performs basic data cleaning: removing duplicates and missing values.
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna()
    final_shape = df.shape
    print(f"Data cleaning complete. Removed {initial_shape[0] - final_shape[0]} rows (duplicates/missing). Final shape: {final_shape}")
    return df

def feature_engineering(df):
    """
    Performs feature engineering.
    - Ensures BMIs are within reasonable ranges (or bins them if needed).
    - Can add interaction terms here if desired.
    For this dataset, features are mostly pre-encoded, so we might just ensure types are correct.
    """
    # Example: Ensure Diabetes_012 is integer for classification
    if 'Diabetes_012' in df.columns:
        df['Diabetes_012'] = df['Diabetes_012'].astype(int)
    return df

def preprocess_features(X_train, X_test, method='standard'):
    """
    Scales features using StandardScaler or MinMaxScaler.
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def handle_imbalance(X, y):
    """
    Applies SMOTETomek to balance the class distribution.
    """
    print("Original class distribution:", np.bincount(y))
    # sampling_strategy='auto' resamples all classes but the majority class
    smt = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    X_res, y_res = smt.fit_resample(X, y)
    print("Resampled class distribution:", np.bincount(y_res))
    return X_res, y_res

def handle_imbalance_adasyn(X, y):
    """
    Applies ADASYN to balance the class distribution.
    ADASYN (Adaptive Synthetic) generates more synthetic data for harder-to-learn examples.
    """
    from imblearn.over_sampling import ADASYN
    print(f"Original class distribution: {np.bincount(y)}")
    
    # sampling_strategy='auto' resamples all not majority
    adasyn = ADASYN(random_state=42, sampling_strategy='auto')
    X_res, y_res = adasyn.fit_resample(X, y)
    
    print(f"Resampled (ADASYN) class distribution: {np.bincount(y_res)}")
    return X_res, y_res

def select_features_rfe(X, y, n_features_to_select=10):
    """
    Selects the most important features using Recursive Feature Elimination (RFE)
    with a Random Forest Estimator.
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"\n--- Feature Selection (RFE) ---")
    print(f"Selecting top {n_features_to_select} features...")
    
    # Use a lightweight RF for selection
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(X, y)
    
    selected_mask = selector.support_
    # Get column names if X is DataFrame, else just return transformed array
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selected_mask]
        print(f"Selected Features: {selected_features.tolist()}")
        X_selected = X.iloc[:, selected_mask]
    else:
        print(f"Selected {sum(selected_mask)} features (indices).")
        X_selected = X[:, selected_mask]
        
    return X_selected, selector

def split_data(df, target_col='Diabetes_012', test_size=0.2, random_state=42):
    """
    Splits data into features (X) and target (y), then into train and test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split into Train ({X_train.shape}) and Test ({X_test.shape}) sets.")
    return X_train, X_test, y_train, y_test
