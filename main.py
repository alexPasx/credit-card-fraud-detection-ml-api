
from pathlib import Path
import math
import logging
import time
import uuid
from fastapi import Request
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load model once when the application starts
MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]
model_version = artifact["model_version"]

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed: id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response

class Transaction(BaseModel):
    features: dict[str, float] = Field(
        description="Time, V1-V28 and Amount"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model_version
    }


@app.post("/predict")
def predict(transaction: Transaction):

    if set(transaction.features) != set(feature_names):
        raise HTTPException(
            status_code=422,
            detail="Expected exactly 30 transaction features"
        )

    values = [
        transaction.features[name]
        for name in feature_names
    ]

    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail="Feature values must be finite"
        )

    data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    probability = float(
        model.predict_proba(data)[0, 1]
    )

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "fraud_score": probability,
        "threshold": threshold,
        "model_version": model_version
    }
