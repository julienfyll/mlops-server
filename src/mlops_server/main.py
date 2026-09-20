from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mlops_server.config import settings, get_compute_device
from mlops_server.engine import llm_engine
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
    """Génération textuelle par le modèle LLM accéléré sur le GPU hôte."""
    generated_text, latency_ms = llm_engine.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    
    return GenerationResponse(
        response=generated_text,
        model=settings.DEFAULT_MODEL,
        latency_ms=latency_ms,
        finish_reason="stop",
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
