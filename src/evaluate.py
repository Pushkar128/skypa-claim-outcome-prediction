import os
import sys
import joblib
import pandas as pd

# Ensure root directory is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def evaluate_saved_model(model_path='model/claim_pipeline.joblib', data_path='data/claims_data.csv'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_data.py first.")
        
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df[['payer', 'provider', 'cpt_code', 'diagnosis_code', 'billed_amount', 'patient_age']]
    y = df['status']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    y_pred = pipeline.predict(X_test)
    classes = pipeline.classes_
    
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    print("\n==========================================")
    print("      SAVED MODEL EVALUATION REPORT       ")
    print("==========================================")
    print(report)
    print("\nConfusion Matrix:")
    print("Classes:", list(classes))
    print(cm)
    
    return report, cm

if __name__ == '__main__':
    evaluate_saved_model()
