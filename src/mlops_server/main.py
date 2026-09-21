import asyncio
from contextlib import asynccontextmanager
import json
import logging
import threading
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mlops_server.config import settings, get_compute_device
from mlops_server.engine import llm_engine
from mlops_server.metrics import (
    LLM_TTFT_SECONDS,
    LLM_ITL_SECONDS,
    LLM_TOKENS_GENERATED_TOTAL,
    LLM_REQUESTS_TOTAL,
)
from mlops_server.schemas import (
    HealthResponse,
    GenerationRequest,
    GenerationResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_server.main")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Charge le modèle une fois au démarrage et libère les ressources à l'arrêt."""
    logger.info(f"Démarrage du service - Modèle cible : {settings.DEFAULT_MODEL}")
    llm_engine.load(settings.DEFAULT_MODEL)
    yield
    logger.info("Arrêt du service et libération de la mémoire...")
    llm_engine.unload()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API MLOps d'inférence LLM distribuée (Mac M5 Metal & Linux AMD ROCm)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verrou d'exclusion mutuelle pour protéger le GPU contre les accès concurrents
engine_lock = asyncio.Lock()

@app.get("/metrics", tags=["Observabilité"])
async def metrics() -> Response:
    """Endpoint Prometheus exposant les métriques de performance et de santé."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", response_model=HealthResponse, tags=["Observabilité"])
async def health_check() -> HealthResponse:
    """Sonde de santé vérifiant le statut du serveur et le GPU détecté."""
    return HealthResponse(
        status="ok",
        app_version="1.1.0-auto-update-test",
        device=get_compute_device(),
    )

@app.post("/v1/chat", response_model=GenerationResponse, tags=["Inférence"])
async def chat_generate(request: GenerationRequest) -> GenerationResponse:
    """Génération textuelle synchrone par le modèle LLM protégée par verrou GPU."""
    async with engine_lock:
        try:
            generated_text, latency_ms = llm_engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            # Approximation du nombre de tokens générés pour les métriques
            estimated_tokens = max(1, len(generated_text.split()))
            LLM_TOKENS_GENERATED_TOTAL.labels(
                model=settings.DEFAULT_MODEL,
                endpoint="/v1/chat",
            ).inc(estimated_tokens)
            LLM_REQUESTS_TOTAL.labels(
                model=settings.DEFAULT_MODEL,
                endpoint="/v1/chat",
                status="success",
            ).inc()

            return GenerationResponse(
                response=generated_text,
                model=settings.DEFAULT_MODEL,
                latency_ms=latency_ms,
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"Erreur lors de la génération synchrone : {e}")
            LLM_REQUESTS_TOTAL.labels(
                model=settings.DEFAULT_MODEL,
                endpoint="/v1/chat",
                status="error",
            ).inc()
            raise

@app.post("/v1/chat/stream", tags=["Inférence"])
async def chat_generate_stream(
    request_data: GenerationRequest,
    request: Request,
) -> StreamingResponse:
    """Génération textuelle en streaming (SSE) avec détection de déconnexion et métrologie TTFT/ITL."""
    stop_event = threading.Event()


    async def event_generator() -> AsyncGenerator[str, None]:
        async with engine_lock:
            token_generator = llm_engine.generate_stream(
                prompt=request_data.prompt,
                max_tokens=request_data.max_tokens,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                stop_event=stop_event,
            )
            start_time = time.perf_counter()
            first_token_received = False
            token_count = 0
            last_token_time = start_time

            try:
                for token in token_generator:
                    current_time = time.perf_counter()

                    # Détection active de déconnexion client
                    if await request.is_disconnected():
                        logger.warning("Client déconnecté en cours de streaming. Arrêt du GPU demandé.")
                        stop_event.set()
                        LLM_REQUESTS_TOTAL.labels(
                            model=settings.DEFAULT_MODEL,
                            endpoint="/v1/chat/stream",
                            status="client_disconnected",
                        ).inc()
                        break

                    # Mesure du TTFT sur le premier token émis
                    if not first_token_received:
                        ttft = current_time - start_time
                        LLM_TTFT_SECONDS.observe(ttft)
                        first_token_received = True
                    else:
                        # Mesure de l'ITL sur les tokens suivants
                        itl = current_time - last_token_time
                        LLM_ITL_SECONDS.observe(itl)

                    last_token_time = current_time
                    token_count += 1

                    chunk_data = json.dumps(
                        {
                            "token": token,
                            "index": token_count,
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {chunk_data}\n\n"
                    await asyncio.sleep(0)  # Céder la main à l'event loop pour flush I/O

                # Marqueur standard de fin de stream SSE si aucune annulation
                if not stop_event.is_set():
                    yield "data: [DONE]\n\n"
                    LLM_REQUESTS_TOTAL.labels(
                        model=settings.DEFAULT_MODEL,
                        endpoint="/v1/chat/stream",
                        status="success",
                    ).inc()
                    LLM_TOKENS_GENERATED_TOTAL.labels(
                        model=settings.DEFAULT_MODEL,
                        endpoint="/v1/chat/stream",
                    ).inc(token_count)

            except Exception as e:
                logger.error(f"Erreur durant le streaming de tokens : {e}")
                stop_event.set()
                LLM_REQUESTS_TOTAL.labels(
                    model=settings.DEFAULT_MODEL,
                    endpoint="/v1/chat/stream",
                    status="error",
                ).inc()
                raise


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def start() -> None:
    """Point d'entrée de démarrage."""
    import uvicorn
    uvicorn.run(
        "mlops_server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # reload à False pour éviter de recharger les poids à chaque save
    )

if __name__ == "__main__":
    start()
