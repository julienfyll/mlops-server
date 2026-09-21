"""Tests unitaires pour le composant Circuit Breaker."""

import asyncio
import pytest
from ai_gateway.circuit_breaker import CircuitBreaker, CircuitState


@pytest.mark.anyio
async def test_circuit_breaker_initial_state():
    """Vérifie que le circuit démarre dans l'état CLOSED et autorise l'exécution."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True


@pytest.mark.anyio
async def test_circuit_breaker_tripping():
    """Vérifie que le circuit bascule en OPEN après avoir atteint le seuil d'échecs."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

    # 1er et 2e échec : le circuit reste fermé
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True

    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True

    # 3e échec : le seuil est atteint, le circuit doit sauter (OPEN)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # Les requêtes suivantes sont immédiatement bloquées (Fail-Fast)
    assert await cb.can_execute() is False


@pytest.mark.anyio
async def test_circuit_breaker_recovery_cycle():
    """Vérifie le cycle complet de récupération : OPEN -> HALF_OPEN -> CLOSED."""
    # Période de repos très courte (0.05s) pour un test unitaire instantané
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

    # Déclenchement de l'ouverture
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert await cb.can_execute() is False

    # Attente de l'expiration du délai de repos
    await asyncio.sleep(0.06)

    # Le premier appel après le repos doit faire basculer en HALF_OPEN
    assert await cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Si la requête de test réussit, le circuit doit se refermer (CLOSED)
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert await cb.can_execute() is True


@pytest.mark.anyio
async def test_circuit_breaker_half_open_failure():
    """Vérifie qu'un échec en mode HALF_OPEN réouvre immédiatement le circuit."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.06)
    assert await cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Si la requête test échoue, le disjoncteur saute à nouveau immédiatement
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert await cb.can_execute() is False
