from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from mlops_server.main import app

@pytest.fixture
def client():
    """Client de test FastAPI simulant les requêtes HTTP."""
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    """Vérifie que la sonde de santé répond 200 et renvoie un accélérateur valide."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert isinstance(data["device"], str)
    assert "platform" in data
    assert "python_version" in data

def test_chat_validation_empty_prompt(client):
    """Vérifie que Pydantic rejette un prompt vide avec un code HTTP 422."""
    payload = {"prompt": ""}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422

def test_chat_validation_invalid_temperature(client):
    """Vérifie que Pydantic rejette une température hors bornes [0.0, 2.0]."""
    payload = {"prompt": "Bonjour", "temperature": 3.5}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422

def test_chat_validation_invalid_max_tokens(client):
    """Vérifie que Pydantic rejette un max_tokens supérieur à 4096."""
    payload = {"prompt": "Bonjour", "max_tokens": 5000}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422

@patch("mlops_server.main.llm_engine.generate")
def test_chat_generation_mocked(mock_generate, client):
    """Vérifie le formatage de la réponse de l'API avec un moteur d'inférence mocké."""
    mock_generate.return_value = ("Décision : Faute évidente.", 42.5)

    payload = {
        "prompt": "Analyse cette action de jeu : tacle par derrière dans la surface.",
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Décision : Faute évidente."
    assert data["latency_ms"] == 42.5
    assert "model" in data
    assert data["finish_reason"] == "stop"
    mock_generate.assert_called_once_with(
        prompt=payload["prompt"],
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
    )
