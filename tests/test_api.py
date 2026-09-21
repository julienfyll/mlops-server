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
    assert data["app_version"] == "1.1.0-auto-update-test"
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


def test_chat_validation_whitespace_prompt(client):
    """Vérifie que Pydantic rejette un prompt composé uniquement d'espaces blancs."""
    payload = {"prompt": "     "}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Vérifie que l'endpoint /metrics répond HTTP 200 et expose les métriques Prometheus du LLM."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "llm_time_to_first_token_seconds" in response.text
    assert "llm_tokens_generated_total" in response.text


def test_chat_stream_validation_errors(client):
    """Vérifie que /v1/chat/stream rejette les entrées invalides (vide, espaces, température hors bornes)."""
    assert client.post("/v1/chat/stream", json={"prompt": ""}).status_code == 422
    assert client.post("/v1/chat/stream", json={"prompt": "   "}).status_code == 422
    assert client.post("/v1/chat/stream", json={"prompt": "Test", "temperature": 5.0}).status_code == 422


@patch("mlops_server.main.llm_engine.generate_stream")
def test_chat_stream_headers_and_contract_mocked(mock_generate_stream, client):
    """Vérifie le contrat SSE, les en-têtes HTTP anti-buffering et le format JSON des chunks émis."""
    mock_generate_stream.return_value = iter(["Bonjour", " le", " monde"])

    payload = {"prompt": "Dis bonjour", "max_tokens": 16}
    response = client.post("/v1/chat/stream", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert response.headers.get("cache-control") == "no-cache"
    assert response.headers.get("connection") == "keep-alive"
    assert response.headers.get("x-accel-buffering") == "no"

    events = [e for e in response.text.strip().split("\n\n") if e]
    assert len(events) == 4  # 3 tokens émis + 1 marqueur [DONE]
    assert events[-1] == "data: [DONE]"

    # Vérification de la conformité du payload JSON des chunks de tokens
    import json
    first_chunk = json.loads(events[0].replace("data: ", ""))
    assert first_chunk["token"] == "Bonjour"
    assert first_chunk["index"] == 1


@patch("mlops_server.main.llm_engine.generate_stream")
def test_chat_stream_client_disconnection_mocked(mock_generate_stream, client):
    """Vérifie l'arrêt immédiat lors d'une déconnexion client en cours de streaming."""
    mock_generate_stream.return_value = iter(["Token1", "Token2", "Token3", "Token4"])

    # Simulation de déconnexion réseau dès le 2e token
    call_state = {"count": 0}

    async def fake_is_disconnected(self=None):
        call_state["count"] += 1
        return call_state["count"] >= 2

    with patch("fastapi.Request.is_disconnected", new=fake_is_disconnected):
        response = client.post("/v1/chat/stream", json={"prompt": "Génère", "max_tokens": 50})

    assert response.status_code == 200
    assert "data: [DONE]" not in response.text  # Non complété car le client a coupé la connexion
    events = [e for e in response.text.strip().split("\n\n") if e]
    assert len(events) < 4

    # Vérification que Prometheus a enregistré le statut client_disconnected
    metrics_resp = client.get("/metrics")
    assert 'status="client_disconnected"' in metrics_resp.text

