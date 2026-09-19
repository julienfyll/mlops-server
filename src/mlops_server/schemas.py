import platform
import sys
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="État général du service")
    device: str = Field(..., description="Accélérateur matériel détecté")
    platform: str = Field(default_factory=platform.platform, description="Plateforme hôte")
    python_version: str = Field(default_factory=lambda: sys.version.split()[0], description="Version Python")

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Texte envoyé au modèle")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Nombre maximum de tokens générés")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Créativité du modèle")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Échantillonnage par noyau")

class GenerationResponse(BaseModel):
    response: str = Field(..., description="Texte généré")
    model: str = Field(..., description="Nom du modèle utilisé")
    latency_ms: float = Field(..., description="Temps de traitement en millisecondes")
    finish_reason: str = Field(default="completed", description="Raison de fin de génération")
