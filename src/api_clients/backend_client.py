# src/api_clients/backend_client.py
import requests
import json
from typing import Optional

class BackendClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base = base_url.rstrip("/")

    def status(self):
        return requests.get(f"{self.base}/status").json()

    def upload_onnx(self, path):
        with open(path, "rb") as f:
            r = requests.post(f"{self.base}/upload_model/onnx", files={"file": f})
        return r.json()

    def set_backend(self, backend):
        return requests.post(f"{self.base}/set_backend", data={"backend": backend}).json()

    def start(self, backend=None):
        data = {}
        if backend:
            data["backend"] = backend
        return requests.post(f"{self.base}/start", data=data).json()

    def stop(self):
        return requests.post(f"{self.base}/stop").json()

    def start_train(self, dataset_yaml, epochs=50):
        return requests.post(f"{self.base}/train", data={"dataset_yaml": dataset_yaml, "epochs": epochs}).json()
