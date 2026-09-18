import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from src.feature_engineering import PayerDenialRateTransformer, CptAvgBilledRatioTransformer
from generate_data import generate_claims_dataset

def build_and_train_pipeline(data_path='data/claims_data.csv', model_dir='model'):
    # Ensure data exists
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df = generate_claims_dataset(num_rows=400, seed=42)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # Features and Target
    X = df[['payer', 'provider', 'cpt_code', 'diagnosis_code', 'billed_amount', 'patient_age']]
    y = df['status']

    # Step 1: Stratified train-test split (80/20) to maintain 70/20/10 target ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Dataset split complete: {len(X_train)} train rows, {len(X_test)} test rows.")
    print("Train status distribution:\n", y_train.value_counts(normalize=True).round(3).to_dict())
    print("Test status distribution:\n", y_test.value_counts(normalize=True).round(3).to_dict())

    # Preprocessing pipelines
    cat_cols = ['payer', 'provider', 'cpt_code', 'diagnosis_code']
    num_cols = ['billed_amount', 'patient_age', 'payer_denial_rate', 'billed_to_cpt_ratio']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    # Bundle feature engineering, preprocessor, and classifier into a single leak-free pipeline
    full_pipeline = Pipeline([
        ('payer_denial_feat', PayerDenialRateTransformer()),
        ('cpt_ratio_feat', CptAvgBilledRatioTransformer()),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # Fit pipeline ONLY on training data
    full_pipeline.fit(X_train, y_train)

    # Step 3: Evaluate on test set
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)
    classes = full_pipeline.classes_

    report = classification_report(y_test, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\n--- Model Evaluation Results ---")
    print(report)
    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"Classes: {classes}")
    print(cm)

    # Save model pipeline artifact
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'claim_pipeline.joblib')
    joblib.dump(full_pipeline, model_path)
    print(f"\nSaved trained pipeline artifact to '{model_path}'.")

    # Save metrics summary text
    metrics_path = os.path.join(model_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== ML Model Evaluation Report ===\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(f"Classes: {list(classes)}\n")
        f.write(str(cm))
        f.write("\n")

    return full_pipeline, report, cm

if __name__ == '__main__':
    build_and_train_pipeline()
