import os
import platform
import sys

def get_compute_device() -> str:
    """Détecte dynamiquement le meilleur accélérateur matériel disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"cuda (GPU: {device_name})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon Metal)"
        return "cpu"
    except ImportError:
        arch = platform.machine()
        system = platform.system()
        return f"cpu (PyTorch non chargé - Système: {system} {arch})"

class Settings:
    PROJECT_NAME: str = "LLM Inference Service"
    VERSION: str = "0.1.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEFAULT_MODEL: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

settings = Settings()
