"""AI Gateway : Contrôleur de trafic, relais de streaming et disjoncteur (Circuit Breaker)."""

from contextlib import asynccontextmanager
import logging
import os
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ai_gateway.circuit_breaker import CircuitBreaker, CircuitState
from ai_gateway.metrics import (
    GATEWAY_CIRCUIT_BREAKER_STATE,
    GATEWAY_REQUESTS_TOTAL,
    GATEWAY_UPSTREAM_LATENCY_SECONDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_gateway.main")

# Configuration de la Gateway et de l'amont
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://100.93.198.49:30080").rstrip("/")
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))

# Initialisation du Circuit Breaker (3 échecs consécutifs -> 10s de repos)
circuit_breaker = CircuitBreaker(
    failure_threshold=int(os.getenv("CB_FAILURE_THRESHOLD", "3")),
    recovery_timeout=float(os.getenv("CB_RECOVERY_TIMEOUT", "10.0")),
    name="AMD-RX6800-Worker",
)

# Client HTTP persistant partagé (pool de connexions)
http_client: httpx.AsyncClient = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestion du cycle de vie du pool de connexions HTTP de la Gateway."""
    global http_client
    logger.info(f"Démarrage de l'AI Gateway -> Cible amont configurée : {UPSTREAM_URL}")
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    yield
    logger.info("Arrêt de l'AI Gateway et fermeture du pool de connexions HTTP...")
    await http_client.aclose()


app = FastAPI(
    title="MLOps AI Gateway",
    version="1.0.0",
    description="Passerelle d'inférence LLM avec Circuit Breaker et relais de streaming",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Supervision"])
async def gateway_health() -> JSONResponse:
    """Vérifie l'état de la Gateway, du Circuit Breaker et teste la joignabilité de l'amont."""
    upstream_online = False
    upstream_device = "inconnu"

    try:
        r = await http_client.get(f"{UPSTREAM_URL}/health", timeout=3.0)
        if r.status_code == 200:
            upstream_online = True
            upstream_device = r.json().get("device", "inconnu")
    except Exception:
        upstream_online = False

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "gateway_role": "Control Plane (Mac M5)",
            "upstream_url": UPSTREAM_URL,
            "upstream_online": upstream_online,
            "upstream_device": upstream_device,
            "circuit_breaker": {
                "state": circuit_breaker.state.value,
                "failure_count": circuit_breaker.failure_count,
            },
        },
    )


@app.get("/metrics", tags=["Supervision"])
async def gateway_metrics() -> Response:
    """Expose les métriques Prometheus de l'AI Gateway."""
    state_mapping = {
        CircuitState.CLOSED: 0,
        CircuitState.HALF_OPEN: 1,
        CircuitState.OPEN: 2,
    }
    GATEWAY_CIRCUIT_BREAKER_STATE.labels(name=circuit_breaker.name).set(
        state_mapping[circuit_breaker.state]
    )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", tags=["Inférence Synchrone"])
async def proxy_chat(request: Request) -> JSONResponse:
    """Relais de la requête synchrone vers le serveur GPU protégé par le Circuit Breaker."""
    # 1. Vérification de l'état du disjoncteur
    if not await circuit_breaker.can_execute():
        logger.warning("Requête bloquée immédiatement : Circuit Breaker OUVERT.")
        GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat", status="503").inc()
        return JSONResponse(
            status_code=503,
            content={
                "error": "circuit_breaker_open",
                "message": "Le serveur GPU est temporairement indisponible (Circuit Breaker ouvert).",
                "retry_after_seconds": circuit_breaker.recovery_timeout,
            },
            headers={"Retry-After": str(int(circuit_breaker.recovery_timeout))},
        )

    # 2. Transmission de la requête vers le worker amont
    body = await request.json()
    start_time = time.perf_counter()
    try:
        response = await http_client.post(f"{UPSTREAM_URL}/v1/chat", json=body)
        duration = time.perf_counter() - start_time
        GATEWAY_UPSTREAM_LATENCY_SECONDS.labels(endpoint="/v1/chat").observe(duration)
        if response.status_code == 200:
            await circuit_breaker.record_success()
            GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat", status="200").inc()
            return JSONResponse(status_code=200, content=response.json())
        else:
            logger.error(f"Erreur HTTP {response.status_code} renvoyée par le worker GPU.")
            await circuit_breaker.record_failure()
            GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat", status=str(response.status_code)).inc()
            return JSONResponse(status_code=response.status_code, content=response.json())
    except Exception as exc:
        logger.error(f"Échec de communication réseau avec le worker GPU : {exc}")
        await circuit_breaker.record_failure()
        GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat", status="502").inc()
        return JSONResponse(
            status_code=502,
            content={"error": "bad_gateway", "message": f"Impossible de joindre le worker GPU : {str(exc)}"},
        )


@app.post("/v1/chat/stream", tags=["Inférence Streaming"])
async def proxy_chat_stream(request: Request):
    """Relais transparent du flux SSE vers le client avec protection Circuit Breaker."""
    # 1. Vérification de l'état du disjoncteur
    if not await circuit_breaker.can_execute():
        logger.warning("Requête de streaming bloquée : Circuit Breaker OUVERT.")
        GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat/stream", status="503").inc()
        return JSONResponse(
            status_code=503,
            content={
                "error": "circuit_breaker_open",
                "message": "Le serveur GPU est temporairement indisponible (Circuit Breaker ouvert).",
                "retry_after_seconds": circuit_breaker.recovery_timeout,
            },
            headers={"Retry-After": str(int(circuit_breaker.recovery_timeout))},
        )

    body = await request.json()

    async def stream_relay() -> AsyncGenerator[str, None]:
        success_recorded = False
        start_time = time.perf_counter()
        try:
            async with http_client.stream("POST", f"{UPSTREAM_URL}/v1/chat/stream", json=body) as upstream_resp:
                if upstream_resp.status_code != 200:
                    error_payload = await upstream_resp.aread()
                    await circuit_breaker.record_failure()
                    GATEWAY_REQUESTS_TOTAL.labels(
                        endpoint="/v1/chat/stream", status=str(upstream_resp.status_code)
                    ).inc()
                    yield f"data: {error_payload.decode()}\n\n"
                    return

                # Première réception réussie de l'amont
                await circuit_breaker.record_success()
                success_recorded = True
                duration = time.perf_counter() - start_time
                GATEWAY_UPSTREAM_LATENCY_SECONDS.labels(endpoint="/v1/chat/stream").observe(duration)
                GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat/stream", status="200").inc()

                async for line in upstream_resp.aiter_lines():
                    # Si le client final coupe sa connexion avec la Gateway
                    if await request.is_disconnected():
                        logger.warning("Client final déconnecté de la Gateway. Fermeture du stream amont.")
                        break

                    if line:
                        yield f"{line}\n\n"

        except Exception as exc:
            logger.error(f"Interruption ou échec du streaming avec le worker amont : {exc}")
            if not success_recorded:
                await circuit_breaker.record_failure()
                GATEWAY_REQUESTS_TOTAL.labels(endpoint="/v1/chat/stream", status="502").inc()
            yield f"data: {{\"error\": \"streaming_interrupted\", \"message\": \"{str(exc)}\"}}\n\n"

    return StreamingResponse(
        stream_relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def start() -> None:
    """Point d'entrée de démarrage de l'AI Gateway."""
    import uvicorn
    uvicorn.run("ai_gateway.main:app", host=GATEWAY_HOST, port=GATEWAY_PORT, reload=False)


if __name__ == "__main__":
    start()
