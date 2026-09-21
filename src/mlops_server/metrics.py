"""Définition des métriques Prometheus pour le serveur MLOps."""

from prometheus_client import Counter, Histogram

# Histogramme pour le Time To First Token (TTFT - Phase Prefill)
# Buckets de 50ms à 5s pour capturer la réactivité initiale
LLM_TTFT_SECONDS = Histogram(
    "llm_time_to_first_token_seconds",
    "Temps écoulé jusqu'à la réception du premier token (TTFT)",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)

# Histogramme pour la latence inter-tokens (ITL - Phase Decode)
# Buckets de 10ms à 200ms pour capturer la régularité du flux
LLM_ITL_SECONDS = Histogram(
    "llm_inter_token_latency_seconds",
    "Délai entre deux tokens consécutifs (ITL)",
    buckets=(0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2),
)

# Compteur total des tokens générés
LLM_TOKENS_GENERATED_TOTAL = Counter(
    "llm_tokens_generated_total",
    "Nombre total de tokens générés par le modèle",
    ["model", "endpoint"],
)

# Compteur total des requêtes par statut
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Nombre total de requêtes traitées par l'API",
    ["model", "endpoint", "status"],
)
