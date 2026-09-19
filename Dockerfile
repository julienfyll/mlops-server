# ==============================================================================
# ÉTAPE 1 : BUILDER (Installation isolée avec uv et compilation du bytecode)
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_COMPILE_BYTECODE=1 : pré-compile les fichiers .pyc pour accélérer le démarrage
# UV_LINK_MODE=copy : garantit que le venv est autonome et transportable
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 1. Mise en cache Docker des dépendances tierces (PyTorch, FastAPI, etc.)
# Si le code dans src/ change mais que uv.lock reste identique, cette étape est instantanée (cache hit).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2. Copie du code applicatif et installation du package mlops-server
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY README.md /app/README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==============================================================================
# ÉTAPE 2 : RUNNER (Image de production minimale sans outils de compilation)
# ==============================================================================
FROM python:3.12-slim-bookworm AS runner

# Sécurité : Création d'un utilisateur non-root pour l'exécution
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copie uniquement de l'environnement virtuel et du code depuis le builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md

# Configuration de l'environnement
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Exécution sous l'utilisateur non-root
USER appuser

EXPOSE 8000

# Point d'entrée de production
CMD ["mlops-server"]
