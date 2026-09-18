# Insurance Claim Outcome Prediction (End-to-End ML Mini Build)

An end-to-end machine learning system and FastAPI web service that predicts insurance claim outcomes (**Paid**, **Partially_Paid**, **Denied**) prior to processing. Built for the Skypa Systems AI/ML Engineer coding assessment.

---

## Architecture & Project Structure

```
skypesystems/
├── README.md                      # Setup guide, evaluation metrics, analysis & design decisions
├── requirements.txt               # Dependencies (scikit-learn, pandas, fastapi, uvicorn, etc.)
├── .gitignore                     # Git exclusions
├── generate_data.py               # Synthetic claims dataset generator (400 rows, 70/20/10 ratio)
├── data/
│   └── claims_data.csv            # Historical claims dataset
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py     # Leakage-free custom Scikit-Learn transformers
│   ├── train.py                   # Stratified split, training Random Forest & saving pipeline
│   └── evaluate.py                # Standalone evaluation & confusion matrix printing
├── model/
│   └── claim_pipeline.joblib      # Saved trained model pipeline artifact
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI service with POST /predict & 400 input validation
│   └── appeal_guidance.py         # Bonus: Snippet retrieval & LLM hallucination mitigation rules
└── tests/
    └── test_api.py                # Pytest integration suite for API endpoints & validation
```

---

## Quick Start & Setup

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Dataset
Simulate 400 realistic insurance claims with class imbalance (70% Paid, 20% Partially_Paid, 10% Denied):
```bash
python generate_data.py
```

### 3. Train Model & Save Pipeline
Perform a stratified 80/20 train/test split, apply data-leakage-free feature engineering, fit a balanced Random Forest classifier, evaluate performance, and save the pipeline:
```bash
python src/train.py
```

### 4. Evaluate Saved Pipeline
Run standalone evaluation on test set:
```bash
python src/evaluate.py
```

### 5. Run FastAPI Web Service
Start the live prediction server on `http://127.0.0.1:8000`:
```bash
uvicorn app.main:app --reload
```
Access interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 6. Run Test Suite
```bash
pytest tests/test_api.py
```

---

## Step 1 — Data Preparation & Data Leakage Prevention

- **Stratified Split**: The dataset is split using `train_test_split(..., stratify=y, test_size=0.20, random_state=42)`. Because target classes are imbalanced (70% Paid, 20% Partially_Paid, 10% Denied), stratification guarantees both training and testing sets preserve the exact target class distribution.
- **Engineered Features**:
  1. `payer_denial_rate`: Calculated as `(Denied claims for payer in TRAIN) / (Total claims for payer in TRAIN)`.
  2. `billed_to_cpt_ratio`: Calculated as `billed_amount / (Mean billed_amount for CPT code in TRAIN)`.
- **Preventing Data Leakage**: Both engineered features derive statistical aggregations **strictly from training split rows**. Unseen payers or CPT codes in the test set default to overall training averages (`global_denial_rate_` and `global_avg_billed_`).

---

## Step 2 & 3 — Model Training & Evaluation Metrics

### Model Architecture
- Classifier: `RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)`
- Bundled into a Scikit-Learn `Pipeline` containing custom feature transformers, `OneHotEncoder(handle_unknown='ignore')` for categorical columns (`payer`, `provider`, `cpt_code`, `diagnosis_code`), and `StandardScaler()` for numerical features.

### Evaluation Performance Metrics

| Target Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Denied** | 0.000 | 0.000 | 0.000 | 11 |
| **Paid** | 0.634 | 0.900 | 0.744 | 50 |
| **Partially_Paid** | 0.429 | 0.158 | 0.231 | 19 |
| **Accuracy** | | | **0.600** | 80 |
| **Macro Avg** | **0.354** | **0.353** | **0.325** | 80 |
| **Weighted Avg** | **0.498** | **0.600** | **0.520** | 80 |

### Confusion Matrix
```
Predicted  --->    Denied  Paid  Partially_Paid
Actual Denied [      0     11        0        ]
Actual Paid   [      1     45        4        ]
Actual Part   [      1     15        3        ]
```

### Analysis of Weakest Class & Concrete Improvement Ideas
- **Weakest Class**: The model is weakest at the **`Denied`** class (Precision: 0.000, Recall: 0.000, F1: 0.000). Because `Denied` is a severe minority class (~10-13% of total data), a standard classifier defaults to predicting `Paid` (74.4% F1) to maximize overall accuracy without catching the rare denial patterns.
- **Concrete Improvement Strategy**:
  1. **SMOTE-NC Oversampling**: Apply Synthetic Minority Over-sampling Technique (SMOTE) specifically on the training set to generate synthetic `Denied` claims before model training.
  2. **Cost-Sensitive Learning / Class Weight Tuning**: Increase the penalty weight for `Denied` class errors (e.g. `class_weight={'Denied': 10, 'Partially_Paid': 3, 'Paid': 1}`).
  3. **Feature Addition**: Include historical prior-authorization flags, specific diagnosis-code denial histories, and out-of-network provider indicators.

---

## Step 4 — Serving API Contract

### Request: `POST /predict`
```json
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "payer": "Aetna",
  "provider": "Dr. Rao",
  "cpt_code": "99213",
  "diagnosis_code": "M54.5",
  "billed_amount": 220.00,
  "patient_age": 45
}
```

### Response (200 OK)
```json
{
  "predicted_status": "Paid",
  "probabilities": {
    "Denied": 0.0512,
    "Paid": 0.8145,
    "Partially_Paid": 0.1343
  }
}
```

### Validation & Error Handling (HTTP 400)
Requests with missing mandatory fields, negative `billed_amount`, or invalid data types (e.g. `billed_amount` as text) trigger a structured HTTP 400 error response:
```json
{
  "error": "Bad Request",
  "message": "Invalid claim payload. Please check field types and required inputs.",
  "details": [...]
}
```

---

## Bonus Challenge — Appeal Guidance & LLM Hallucination Risk Mitigation

### Endpoint: `POST /appeal-guidance`
Input: `{ "denial_reason": "missing_auth" }`
Returns keyword-matched appeal guidance snippet (e.g., retro-authorization proof requirements for emergency care).

### 5-10 Lines on Reducing LLM Hallucination Risk
When feeding retrieved appeal snippets to an LLM for drafting formal appeal letters:
1. **Strict Grounding Prompts**: Enforce system prompts requiring the LLM to rely *strictly* on provided EOB policy snippets and fail gracefully if evidence is absent.
2. **Structured Output Schemas (JSON/Pydantic)**: Mandate rigid JSON schemas (e.g., `appeal_reason`, `clinical_citation`, `action_requested`) to eliminate conversational drift.
3. **Mandatory Source Quotation**: Require the model to directly cite paragraph line items from the insurance contract guidelines.
4. **Immutable Metadata Injection**: Inject verified claim fields (`patient_id`, `date_of_service`, `cpt_code`) into non-editable template slots prior to LLM text generation.
5. **Deterministic Post-Validation**: Filter generated text through regex validators to verify dates, procedural codes, and claim dollar amounts match input records exactly before sending.

---

## Assumptions & Time-Limit Trade-offs

1. **Synthetic Data Simulation**: Created 400 records with realistic conditional distribution modeling billing complexity vs denial rates.
2. **Model Selection**: Used `RandomForestClassifier` with balanced class weights rather than exhaustive Hyperparameter Tuning (GridSearchCV) to maximize reproducibility and pipeline stability within the 90-minute window.
3. **Categorical Encodings**: Employed `OneHotEncoder` with `handle_unknown='ignore'` to handle unseen high-cardinality categories smoothly during real-time inference.
