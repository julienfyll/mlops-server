"""Gestionnaire de disjoncteur (Circuit Breaker) pour protéger le serveur d'inférence GPU."""

import asyncio
from enum import Enum
import logging
import time
from typing import Optional

logger = logging.getLogger("ai_gateway.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal : Les requêtes transitent vers le worker GPU
    OPEN = "OPEN"            # Déclenché : Échecs répétés, rejet immédiat des requêtes (HTTP 503)
    HALF_OPEN = "HALF_OPEN"  # Période d'essai : Autorise une requête test après un délai de repos


class CircuitBreakerOpenException(Exception):
    """Exception levée lorsque le disjoncteur est ouvert pour protéger le GPU."""

    def __init__(
        self,
        message: str = "Circuit ouvert : le serveur GPU est temporairement indisponible.",
        retry_after: int = 10,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class CircuitBreaker:
    """Disjoncteur asynchrone protégeant le serveur d'inférence contre les pannes en cascade."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
        name: str = "GPU-Worker",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Vérifie si une requête peut être transmise au worker GPU.
        
        Si le disjoncteur est OPEN, vérifie si la période de repos est écoulée
        pour basculer en HALF_OPEN.
        """
        async with self._lock:
            now = time.monotonic()

            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                    logger.info(f"[{self.name}] Délai de repos écoulé -> bascule en mode HALF_OPEN.")
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # En mode semi-ouvert, on autorise la requête test
                return True

            return False

    async def record_success(self) -> None:
        """Enregistre un succès et referme le disjoncteur si on était en phase d'essai."""
        async with self._lock:
            if self.state != CircuitState.CLOSED:
                logger.info(f"[{self.name}] Requête test réussie ! Rétablissement du circuit en mode CLOSED.")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None

    async def record_failure(self) -> None:
        """Enregistre un échec et déclenche l'ouverture du circuit si le seuil est atteint."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            logger.warning(
                f"[{self.name}] Échec détecté ({self.failure_count}/{self.failure_threshold})."
            )

            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                logger.error(
                    f"[{self.name}] Seuil d'échecs atteint ! DÉCLENCHEMENT DU CIRCUIT BREAKER -> État OPEN."
                )
                self.state = CircuitState.OPEN
