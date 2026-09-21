"""Tests unitaires pour le moteur d'évaluation sémantique et le Golden Dataset."""

import json
import os
import pytest
from mlops_server.evals import evaluate_response


def test_golden_dataset_structure():
    """Vérifie la conformité structurelle de chaque élément du Golden Dataset."""
    dataset_path = "tests/evals/golden_dataset.json"
    assert os.path.exists(dataset_path), "Le fichier golden_dataset.json doit exister."

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert isinstance(dataset, list), "Le Golden Dataset doit être une liste de cas de test."
    assert len(dataset) >= 5, "Le dataset doit contenir au moins 5 cas de test."

    for item in dataset:
        assert "id" in item, "Chaque cas doit avoir un 'id'."
        assert "name" in item, "Chaque cas doit avoir un 'name'."
        assert "prompt" in item, "Chaque cas doit avoir un 'prompt'."
        assert "criteria" in item, "Chaque cas doit avoir des 'criteria'."
        assert "type" in item["criteria"], "Les critères doivent spécifier un 'type'."


def test_evaluator_contains_any():
    """Vérifie l'évaluation par présence d'au moins un mot-clé."""
    criteria = {"type": "contains_any", "expected": ["Canberra", "canberra"]}
    passed, _ = evaluate_response("La capitale est Canberra.", criteria)
    assert passed is True

    failed, reason = evaluate_response("La capitale est Sydney.", criteria)
    assert failed is False
    assert "Aucun mot-clé attendu" in reason


def test_evaluator_valid_json_strict():
    """Vérifie la validation de structure JSON stricte."""
    criteria = {"type": "valid_json", "required_keys": ["nom", "age"]}

    # Cas JSON valide pur
    passed, _ = evaluate_response('{"nom": "Alice", "age": 28}', criteria)
    assert passed is True

    # Cas JSON entouré de markdown
    passed_md, _ = evaluate_response('```json\n{"nom": "Bob", "age": 35}\n```', criteria)
    assert passed_md is True

    # Cas clé manquante
    failed_keys, reason_keys = evaluate_response('{"nom": "Charlie"}', criteria)
    assert failed_keys is False
    assert "manquantes" in reason_keys

    # Cas JSON invalide
    failed_syntax, reason_syntax = evaluate_response("Ceci n'est pas un JSON.", criteria)
    assert failed_syntax is False
    assert "Impossible de parser" in reason_syntax


def test_evaluator_max_words_constraint():
    """Vérifie la contrainte de concision (longueur maximale en mots)."""
    criteria = {"type": "contains_any", "expected": ["Oui"], "max_words": 2}

    # Concis : validé
    passed, _ = evaluate_response("Oui.", criteria)
    assert passed is True

    # Verbeux : rejeté
    failed, reason = evaluate_response("Oui, absolument, je confirme totalement cette affirmation.", criteria)
    assert failed is False
    assert "Trop verbeux" in reason
