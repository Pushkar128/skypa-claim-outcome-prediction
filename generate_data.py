import os
import pandas as pd
import numpy as np

def generate_claims_dataset(num_rows=400, seed=42):
    np.random.seed(seed)
    
    payers = ['Aetna', 'Cigna', 'UnitedHC', 'Humana', 'BCBS']
    providers = ['Dr. Rao', 'Dr. Shah', 'Dr. Lee', 'Dr. Patel', 'Dr. Gupta']
    cpt_codes = ['99213', '99214', '99215', '99203', '99204']
    diagnosis_codes = ['M54.5', 'J06.9', 'E11.9', 'I10', 'R07.9']
    
    # Target status distribution: ~70% Paid, ~20% Partially_Paid, ~10% Denied
    statuses = ['Paid', 'Partially_Paid', 'Denied']
    base_probs = [0.70, 0.20, 0.10]
    
    data = []
    
    for i in range(1, num_rows + 1):
        payer = np.random.choice(payers, p=[0.25, 0.20, 0.25, 0.15, 0.15])
        provider = np.random.choice(providers)
        cpt = np.random.choice(cpt_codes, p=[0.35, 0.30, 0.15, 0.10, 0.10])
        diag = np.random.choice(diagnosis_codes)
        patient_age = int(np.random.normal(loc=48, scale=16))
        patient_age = max(18, min(85, patient_age))
        
        # Billed amount based on CPT code complexity
        cpt_base_price = {
            '99213': 150.0,
            '99214': 250.0,
            '99215': 380.0,
            '99203': 180.0,
            '99204': 290.0
        }
        base_price = cpt_base_price[cpt]
        billed_amount = round(float(np.random.normal(loc=base_price, scale=35.0)), 2)
        billed_amount = max(50.0, billed_amount)
        
        # Determine status with realistic conditional probabilities
        # Higher billed amounts and certain CPT/payers increase denial/partial risk
        p_paid = 0.70
        p_partial = 0.20
        p_denied = 0.10
        
        if billed_amount > 320.0:
            p_paid -= 0.15
            p_partial += 0.10
            p_denied += 0.05
        elif payer in ['Cigna', 'Humana']:
            p_paid -= 0.08
            p_denied += 0.08
            
        if cpt == '99215':
            p_paid -= 0.10
            p_partial += 0.05
            p_denied += 0.05
            
        # Normalize probabilities
        prob_sum = p_paid + p_partial + p_denied
        final_probs = [p_paid / prob_sum, p_partial / prob_sum, p_denied / prob_sum]
        
        status = np.random.choice(statuses, p=final_probs)
        
        data.append({
            'id': i,
            'payer': payer,
            'provider': provider,
            'cpt_code': cpt,
            'diagnosis_code': diag,
            'billed_amount': billed_amount,
            'patient_age': patient_age,
            'status': status
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    claims_df = generate_claims_dataset(num_rows=400, seed=42)
    output_path = os.path.join('data', 'claims_data.csv')
    claims_df.to_csv(output_path, index=False)
    
    print(f"Generated {len(claims_df)} rows of synthetic claims data at '{output_path}'.")
    print("Class distribution:")
    print(claims_df['status'].value_counts(normalize=True).round(3))
