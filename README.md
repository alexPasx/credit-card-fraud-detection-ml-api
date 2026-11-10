# Credit Card Fraud Detection — End-to-End ML Engineering

An end-to-end machine learning project for detecting potentially fraudulent credit card transactions, covering model development, evaluation, API serving, automated testing, and Docker containerization.

The objective is to demonstrate a reproducible ML workflow and deploy a trained classification model as a locally running REST API.

## 1. Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem. Fraudulent transactions represent a very small fraction of all transactions, making accuracy alone an insufficient evaluation metric.

This project focuses on:

- Handling imbalanced classification data.
- Comparing model training strategies.
- Evaluating precision, recall, F1-score, and Average Precision.
- Selecting an appropriate classification threshold.
- Serving predictions through FastAPI.
- Testing and containerizing the application.

## 2. Dataset

**Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | 0.173% |
| Input features | 30 |
| Target | Class |
| Task | Binary classification |

The dataset contains 28 PCA-transformed features (`V1`–`V28`), along with `Time` and `Amount`.

The target variable is `Class`, where `0` represents a legitimate transaction and `1` represents fraud.

## 3. Machine Learning pipeline

The selected baseline model uses:

- StandardScaler for feature standardization.
- Logistic Regression for binary classification.
- Stratified data splitting.
- Stratified cross-validation.
- Decision threshold tuning.

Alternative strategies, including SMOTE and class-weight balancing, were evaluated. The standard Logistic Regression pipeline was retained because it offered competitive performance with lower implementation complexity.

## 4. Model evaluation

The selected model uses a decision threshold of **0.05**, rather than the default 0.50.

### Test Set results

| Metric | Result |
|---|---|
| Precision — Fraud | 65.32% |
| Recall — Fraud | 82.65% |
| F1-score — Fraud | 72.97% |
| Average Precision | 0.7414 |
| Decision threshold | 0.05 |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,821 | 43 |
| Actual Fraud | 17 | 81 |

Lowering the threshold improves fraud recall while increasing false positives.

**Evaluation limitation:** The test set was inspected during model development, so the reported performance should not be interpreted as an estimate from a completely untouched final holdout. Independent validation would be required before any real-world use.

## 5. Project architecture

```text
Transaction JSON
       |
       v
   FastAPI
       |
       v
Input Validation
       |
       v
Scikit-learn Pipeline
(StandardScaler + LogisticRegression)
       |
       v
Fraud Probability
       |
       v
Threshold = 0.05
       |
       v
JSON Prediction Response
```

The trained model is serialized using Joblib and loaded when the API starts.

## 6. Technology stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| REST API | FastAPI |
| API server | Uvicorn |
| Automated testing | Pytest |
| Containerization | Docker |
| Development | Google Colab, VS Code |

## 7. Repository structure

```text
fraud-detection-api/
├── main.py
├── test_main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── models/
    └── fraud_model.joblib
```

## 8. Running locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn main:app --reload
```

Open the interactive API documentation:

http://localhost:8000/docs

### API endpoints

**GET `/health`**

Returns the API health status and model version.

Example response:

```json
{
  "status": "ok",
  "model_version": "1.0.0"
}
```

**POST `/predict`**

Accepts a JSON object containing exactly 30 numerical features: `Time`, `V1` through `V28`, and `Amount`.

Example request structure:

```json
{
  "features": {
    "Time": 0.0,
    "V1": 0.0,
    "V2": 0.0,
    "...": 0.0,
    "V28": 0.0,
    "Amount": 0.0
  }
}
```

The `"..."` entry is illustrative only and must be replaced with the remaining feature names in an actual request.

Example response fields:

```json
{
  "prediction": 0,
  "fraud_score": 0.00006665895,
  "threshold": 0.05,
  "model_version": "1.0.0"
}
```

The response above illustrates the output format using a previously tested transaction.

## 9. Docker deployment — local

Build the image:

```bash
docker build -t fraud-detection-api:1.1 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 fraud-detection-api:1.1
```

Access the API at:

http://localhost:8000/docs

Docker packages the application, trained model, and required dependencies into a reproducible runtime environment.

## 10. Automated testing

Run the test suite:

```bash
python -m pytest test_main.py -v
```

Four automated tests cover:

1. Health endpoint.
2. Valid prediction request.
3. Invalid request with missing features.
4. Unique request identifiers.

All four tests passed during local development.

## 11. Logging

The API includes request-level logging with:

- Unique request identifiers.
- HTTP method and endpoint.
- HTTP response status.
- Request latency in milliseconds.
- Exception logging for unexpected request failures.

The application does not intentionally log raw transaction feature values.

## 12. Proposed AWS architecture — not deployed

A potential AWS deployment could use:

- **Amazon ECR:** Docker image storage.
- **Amazon ECS with Fargate:** Container execution.
- **Application Load Balancer:** HTTP routing.
- **Amazon CloudWatch:** Centralized logs and monitoring.
- **AWS IAM:** Access and permission management.

```text
Client
  |
  v
Application Load Balancer
  |
  v
Amazon ECS / Fargate
  |
  v
FastAPI + ML Model
  |
  +----> CloudWatch Logs

Docker Image ---> Amazon ECR
```

This architecture is a conceptual proposal. No AWS infrastructure was provisioned or tested as part of this project.

## 13. Limitations

- The dataset is anonymized and does not represent a live banking environment.
- The model is trained on historical, static transaction data.
- Model performance may change under data distribution shifts.
- The classification threshold is not calibrated against real-world financial costs.
- The test set was not maintained as a fully untouched final holdout.
- The API has no authentication or rate limiting.
- No production infrastructure or continuous model monitoring has been implemented.

## 14. Future improvements

Potential extensions include independent model validation, probability calibration, cost-sensitive threshold selection, CI/CD, model drift detection, and secure cloud deployment.

These improvements are outside the scope of the current implementation.

## 15. Project status

**Completed:** Model development, evaluation, serialization, FastAPI serving, automated testing, request logging, and local Docker execution.

**Deployment:** Local Docker environment only.

**Purpose:** Educational and portfolio demonstration of an end-to-end ML Engineering workflow.