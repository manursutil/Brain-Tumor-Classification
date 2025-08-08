import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api as api


def make_jpeg_bytes(size=(32, 32), color=(123, 222, 111)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class DummyModel:
    pass


@pytest.fixture(autouse=True)
def setup_app(monkeypatch):
    """
    Autouse to isolate tests:
    - Provide a dummy model via dependency override
    - Reset monkeypatched functions after each test
    """
    dummy = DummyModel()

    api.model = dummy

    api.app.dependency_overrides[api.get_model] = lambda: dummy

    yield

    api.app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_error_when_model_none(client: TestClient):
    # Override to simulate missing model
    api.app.dependency_overrides[api.get_model] = lambda: None
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "error", "model_loaded": False}


def test_predict_success(client: TestClient, monkeypatch):
    # Stub predict_image to avoid running real inference
    expected: dict[str, Any] = {"prediction": "tumor", "confidence": 0.87}
    monkeypatch.setattr(api, "predict_image", lambda model, image: expected)

    jpeg_bytes = make_jpeg_bytes()
    files = {"file": ("test.jpg", jpeg_bytes, "image/jpeg")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    assert resp.json() == expected


def test_predict_too_large_returns_413(client: TestClient):
    too_big = b"0" * (api.MAX_IMAGE_SIZE + 1)
    files = {"file": ("huge.jpg", too_big, "image/jpeg")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 413
    assert resp.json()["detail"] == "Image too large"


def test_predict_invalid_image_returns_500(client: TestClient, monkeypatch):
    def boom(_):
        raise ValueError("bad image")

    monkeypatch.setattr(api.Image, "open", boom)

    bogus = b"not-a-real-image"
    files = {"file": ("bad.jpg", bogus, "image/jpeg")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 500
    assert "bad image" in resp.json()["detail"]


def test_metrics_success(client: TestClient, monkeypatch):
    monkeypatch.setattr(api, "evaluate_model", lambda model: {"f1": 0.91, "acc": 0.95})
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.json() == {"f1": 0.91, "acc": 0.95}


def test_metrics_failure_returns_500(client: TestClient, monkeypatch):
    def blow_up(_):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(api, "evaluate_model", blow_up)
    resp = client.get("/metrics")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "metrics failed"
