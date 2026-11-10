
from fastapi.testclient import TestClient
from main import app

# Create a test client for our FastAPI application
client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "1.0.0"


def test_predict_valid_transaction():
    features = {
        "Time": 160760.0,
        "V1": -0.674466064578314,
        "V2": 1.40810501967799,
        "V3": -1.11062205357093,
        "V4": -1.32836577843066,
        "V5": 1.38899603254837,
        "V6": -1.30843906707795,
        "V7": 1.88587890268717,
        "V8": -0.614232966299775,
        "V9": 0.311652212453101,
        "V10": 0.65075700363522,
        "V11": -0.857784661547805,
        "V12": -0.229961445775592,
        "V13": -0.19981700479103,
        "V14": 0.266371326329879,
        "V15": -0.0465441684754424,
        "V16": -0.741398089749789,
        "V17": -0.605616644106022,
        "V18": -0.39256818789208,
        "V19": -0.162648311024695,
        "V20": 0.394321820843914,
        "V21": 0.0800842396026648,
        "V22": 0.810033595602455,
        "V23": -0.224327230436412,
        "V24": 0.707899237446867,
        "V25": -0.13583702273753,
        "V26": 0.0451021964988772,
        "V27": 0.533837219064273,
        "V28": 0.291319252625364,
        "Amount": 23.0
    }

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert abs(data["fraud_score"] - 0.00006665895465620403) < 1e-6
    assert data["threshold"] == 0.05
    assert data["model_version"] == "1.0.0"


def test_predict_missing_features():
    response = client.post(
        "/predict",
        json={
            "features": {
                "Time": 160760.0,
                "V1": -0.67,
                "Amount": 23.0
            }
        }
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Expected exactly 30 transaction features"
    )
def test_request_id_is_unique():
    response_1 = client.get("/health")
    response_2 = client.get("/health")

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    request_id_1 = response_1.headers["X-Request-ID"]
    request_id_2 = response_2.headers["X-Request-ID"]

    assert request_id_1
    assert request_id_2
    assert request_id_1 != request_id_2