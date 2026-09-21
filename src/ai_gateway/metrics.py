"""Définition des métriques Prometheus pour l'AI Gateway."""

from prometheus_client import Counter, Gauge, Histogram

# Compteur total des requêtes transitant par la Gateway
GATEWAY_REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Nombre total de requêtes traitées par l'AI Gateway",
    ["endpoint", "status"],
)

# Jauge pour surveiller l'état du Circuit Breaker (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
GATEWAY_CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "État actuel du Circuit Breaker (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["name"],
)

# Histogramme de la latence de transit amont (Gateway vers Worker GPU)
GATEWAY_UPSTREAM_LATENCY_SECONDS = Histogram(
    "gateway_upstream_latency_seconds",
    "Latence réseau des requêtes relayées vers le worker GPU amont",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
