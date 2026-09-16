# Customer Churn Prediction (Scratch Model Example)

This example demonstrates how to build and deploy a real-world tabular ML model from scratch using ModelKit and scikit-learn.

## Dataset
Download the Telecom Churn CSV dataset from Kaggle:
- **Kaggle Dataset URL**: [https://www.kaggle.com/datasets/barun2104/telecom-churn](https://www.kaggle.com/datasets/barun2104/telecom-churn)
- Save the downloaded file as `data/telecom_churn.csv`.

## Dependencies
Install the required packages using ModelKit or pip:
```bash
modelkit install scikit-learn pandas
# or
pip install scikit-learn pandas
```

## Quickstart

### 1. Initialize Project
```bash
# Option A: In a new folder
modelkit init churn_model
cd churn_model

# Option B: In the current directory
modelkit init .
```

### 2. Copy Code & Data
Place `data.py`, `model.py`, `trainer.py`, `evaluator.py`, and `inference.py` into your project package directory (`churn_model/`), and place `telecom_churn.csv` into `data/`.

### 3. Validate Dataset
```bash
modelkit data validate
```

### 4. Train Model
```bash
modelkit train
```

### 5. Evaluate Performance
```bash
modelkit evaluate
```

### 6. Serve API
```bash
modelkit serve --port 8000
```

### 7. Test Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AccountWeeks": 128,
    "ContractRenewal": 1,
    "DataPlan": 1,
    "DataUsage": 2.7,
    "CustServCalls": 1,
    "DayMins": 265.1,
    "DayCalls": 110,
    "MonthlyCharge": 89.0,
    "OverageFee": 9.87,
    "RoamMins": 10.0
  }'
```
Response:
```json
{
  "churn_prediction": 0,
  "churn_risk": 0.12,
  "decision": "Retain (Low Risk)",
  "status": "success"
}
```
