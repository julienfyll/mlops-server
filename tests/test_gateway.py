"""Tests unitaires et d'intégration pour l'application AI Gateway."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import httpx
import pytest

from ai_gateway.circuit_breaker import CircuitState
from ai_gateway.main import app, circuit_breaker


@pytest.fixture
def gateway_client():
    """Client de test FastAPI simulant les requêtes vers l'AI Gateway."""
    # Réinitialisation de l'état du disjoncteur avant chaque test
    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.failure_count = 0
    circuit_breaker.last_failure_time = None

    with TestClient(app) as client:
        yield client


def test_gateway_health(gateway_client):
    """Vérifie que la sonde de santé de la Gateway renvoie son rôle et l'état du disjoncteur."""
    response = gateway_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["gateway_role"] == "Control Plane (Mac M5)"
    assert data["circuit_breaker"]["state"] == "CLOSED"


@pytest.mark.anyio
async def test_gateway_proxy_chat_success(gateway_client):
    """Vérifie que la Gateway relaie avec succès la requête vers le worker amont."""
    mock_upstream_response = httpx.Response(
        status_code=200,
        json={"response": "Réponse du GPU AMD", "model": "Qwen2.5-0.5B", "latency_ms": 120.0},
        request=httpx.Request("POST", "http://test/v1/chat"),
    )

    with patch("ai_gateway.main.http_client.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_upstream_response

        response = gateway_client.post("/v1/chat", json={"prompt": "Bonjour"})
        assert response.status_code == 200
        assert response.json()["response"] == "Réponse du GPU AMD"
        assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.anyio
async def test_gateway_circuit_breaker_tripping_on_failures(gateway_client):
    """Vérifie que 3 pannes consécutives du worker GPU déclenchent le Circuit Breaker en HTTP 503."""
    # Simulation d'une panne réseau totale vers le worker GPU (ConnectError)
    with patch(
        "ai_gateway.main.http_client.post",
        side_effect=httpx.ConnectError("Connection refused by GPU worker"),
    ):
        # 1er échec : la Gateway tente l'appel et renvoie 502 Bad Gateway
        r1 = gateway_client.post("/v1/chat", json={"prompt": "Req 1"})
        assert r1.status_code == 502
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == CircuitState.CLOSED

        # 2e échec
        r2 = gateway_client.post("/v1/chat", json={"prompt": "Req 2"})
        assert r2.status_code == 502
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == CircuitState.CLOSED

        # 3e échec : seuil atteint, le disjoncteur saute !
        r3 = gateway_client.post("/v1/chat", json={"prompt": "Req 3"})
        assert r3.status_code == 502
        assert circuit_breaker.state == CircuitState.OPEN

        # 4e requête : Le disjoncteur est OUVERT.
        # La requête est rejetée IMMÉDIATEMENT en 503 SANS même tenter de contacter le réseau !
        r4 = gateway_client.post("/v1/chat", json={"prompt": "Req 4"})
        assert r4.status_code == 503
        data_503 = r4.json()
        assert data_503["error"] == "circuit_breaker_open"
        assert "Retry-After" in r4.headers
        assert r4.headers["Retry-After"] == "10"


def test_gateway_metrics_endpoint(gateway_client):
    """Vérifie que l'AI Gateway expose correctement ses métriques au format Prometheus."""
    response = gateway_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "gateway_circuit_breaker_state" in response.text
    assert "gateway_requests_total" in response.text
    assert "gateway_upstream_latency_seconds" in response.text

