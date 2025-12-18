# Diabetes Prediction System - Final Report

This document summarizes the development, experimentation, and results of the diabetes prediction system.

## 1. Project Goal
Develop a machine learning pipeline to classify individuals into three categories based on health indicators:
- **0: Healthy**
- **1: Pre-diabetic** (The hardest class to predict)
- **2: Diabetic**

## 2. Methodology & Experiments

We implemented and tested four distinct approaches to handle the significant class imbalance.

### Experiment 1: Baseline Random Forest
- **Technique**: Standard Random Forest, no special balancing.
- **Result**: High accuracy (82%), but **failed completely** as a screening tool.
- **Recall (Sensitivity)**:
    - Pre-diabetic: **0%** (Missed everyone)
    - Diabetic: **21%** (Missed 4 out of 5 diabetics)

### Experiment 2: XGBoost + GPU + SMOTETomek
- **Technique**: Gradient Boosting (XGBoost) with SMOTETomek oversampling to artificially balance the training data.
- **Result**: Slight accuracy boost (83%), but still failed on minority classes.
- **Recall**:
    - Pre-diabetic: **0%**
    - Diabetic: **24%**

### Experiment 3: Balanced Random Forest (Champion Model) 🏆
- **Technique**: `BalancedRandomForestClassifier` which undersamples the majority class inside every tree's bootstrap sample. Optimized for Recall.
- **Result**: Accuracy dropped to 67%, but **Recall soared**. This is the best model for a medical screening context where "missing a sick person" (False Negative) is the worst outcome.
- **Recall**:
    - Pre-diabetic: **14%** (Best achieved)
    - Diabetic: **70%** (Major improvement)

### Experiment 4: Hierarchical (Two-Stage) + ADASYN
- **Technique**: Used ADASYN to generate synthetic samples (expanding training set to ~450k rows) + Two-Stage Classification.
- **Result**: High training scores (>92%) but **poor generalization** on real test data. The model overfitted to the synthetic minority samples and reverted to predicting "Healthy" for almost everyone in the real test set.
- **Recall**:
    - Pre-diabetic: **0%** (1 / 926 detected)
    - Diabetic: **10%** (703 / 7019 detected)

### Experiment 5: LightGBM + RFE (Top 10 Features)
- **Technique**: Used `LightGBM` (balanced class weights) and removed noise by keeping only the Top 10 features via Recursive Feature Elimination. (No synthetic sampling).
- **Result**: Accuracy (65%) is similar to Balanced RF, but it achieved the **highest Pre-diabetic Recall** seen so far.
- **Recall**:
    - Pre-diabetic: **15%** (Best detected count: 137/926)
    - Diabetic: **65%**

### Experiment 6: MLP Classifier + ADASYN
- **Technique**: Multi-Layer Perceptron (Neural Network) with ADASYN oversampling.
- **Result**: High Accuracy (81%), but **failed to learn minority classes** effectively despite oversampling. It behaved similarly to the baseline Random Forest, favoring the majority class.
- **Recall**:
    - Pre-diabetic: **2%** (20 / 926 detected)
    - Diabetic: **28%**

### Experiment 8: MLP Classifier (No Balancing)
- **Technique**: MLP on raw, imbalanced data.
- **Result**: High Accuracy (84%), but **0% Recall** for Pre-diabetics. It learned to ignore the minority class entirely to maximize overall accuracy.
- **Recall**:
    - Pre-diabetic: **0%** (0 / 926 detected)
    - Diabetic: **17%**

### Experiment 9: High Precision Tuning (Threshold Moving)
- **Technique**: Used the best signal model (LightGBM) and increased the decision threshold (>50% to >95%) to filter out false positives for Pre-diabetics.
- **Result**: **Failed to improve Precision**.
- **Max Precision**: **3.3%** at Threshold 0.60. Above this threshold, the model stopped detecting anything.
- **Conclusion**: The model is never "confidently" wrong; the signal for Pre-diabetes is just too weak or essentially identical to "Healthy" in this feature space.

## 3. Comparative Results Table

| Model | Overall Accuracy | Healthy Recall | Pre-diabetic Recall | Diabetic Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline RF** | 82.11% | **96%** | 0% | 21% | Unsafe for screening |
| **XGBoost** | **83.36%** | 96% | 0% | 24% | Unsafe for screening |
| **Balanced RF** | 67.00% | 68% | 14% | **70%** | **Best for Diabetics** |
| **LightGBM + RFE** | 65.00% | 66% | **15%** | 65% | **Best for Pre-diabetics** |
| **MLP + ADASYN** | 81.25% | 93% | 2% | 28% | Unsafe for screening |
| **MLP (No Bal)** | **83.75%** | 98% | 0% | 17% | **Useless for Screening** |
| **High Precision Tuning** | N/A | N/A | <3% | N/A | **High Precision Impossible** |

## 4. Critical Insight: The "Glass Ceiling"
Despite using advanced techniques (SMOTE, ADASYN, Tomek Links, Hierarchical Ensemble), the detection of the **Pre-diabetic** class remained low (max 14%).

**The Reason:**
Dataset documentation from the CDC states:
> *"The CDC estimates that ... **8 in 10 prediabetics are unaware of their risk**."*

This implies that ~80% of "Pre-diabetic" individuals in the survey likely labeled themselves as **"Healthy"** (Class 0). The model effectively has no reliable "ground truth" for this class, as the "Healthy" class is heavily contaminated with undiagnosed pre-diabetics.
