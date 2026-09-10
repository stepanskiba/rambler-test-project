import pytest
from fastapi.testclient import TestClient
import numpy as np

from src import app as app_module
from src.app import app

client = TestClient(app)


def moderate(text: str):
    return client.post("/moderate", json={"text": text})


def test_health():
    assert client.get("/health").status_code == 200


def test_response_contract():
    r = moderate("привет, отличная статья")
    assert r.status_code == 200

    body = r.json()
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.parametrize("text", ["", "   ", "a" * 5001])
def test_invalid_input(text):
    assert moderate(text).status_code == 422


def test_missing_field():
    assert client.post("/moderate", json={}).status_code == 422


def test_neutral_text_is_not_blocked():
    assert moderate("спасибо, было интересно почитать").json()["decision"] != "BLOCK"

@pytest.mark.parametrize("proba, expected", [(0.01, "ALLOW"), (0.5, "REVIEW"), (0.99, "BLOCK")])
def test_zones(monkeypatch, proba, expected):
    """Границы зон проверяем на заглушке, независимо от качества модели."""
    monkeypatch.setattr(app_module.model, "predict_proba", lambda texts: np.array([[1 - proba, proba]]))
    assert moderate("любой текст").json()["decision"] == expected