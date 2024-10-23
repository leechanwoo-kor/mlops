# examples/test_training.py
import requests
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# MLflow에 기록
training_data = {
    "experiment_name": "iris_classification",
    "model_name": "iris_logistic_regression",
    "params": {
        "solver": model.get_params()["solver"],
        "max_iter": model.get_params()["max_iter"]
    },
    "metrics": {
        "accuracy": float(accuracy)
    },
    "tags": {
        "dataset": "iris",
        "model_type": "logistic_regression"
    }
}

# API 호출
response = requests.post(
    "http://localhost:8000/api/v1/models/train",
    json=training_data
)

print(f"Training logged: {response.json()}")

# 실험 결과 조회
response = requests.get(
    "http://localhost:8000/api/v1/models/experiments/iris_classification/runs"
)

print(f"Experiment runs: {json.dumps(response.json(), indent=2)}")