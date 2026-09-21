"""Module d'évaluation sémantique pour mesurer la qualité des réponses LLM."""

import json
import re
from typing import Any, Dict, Tuple


def evaluate_response(response_text: str, criteria: Dict[str, Any]) -> Tuple[bool, str]:
    """Évalue la conformité d'une réponse textuelle vis-à-vis des critères déclarés."""
    c_type = criteria.get("type")

    # 1. Vérification de concision (max_words)
    if "max_words" in criteria:
        word_count = len(response_text.strip().split())
        if word_count > criteria["max_words"]:
            return False, f"Trop verbeux ({word_count} mots > max {criteria['max_words']})"

    # 2. Vérification par présence de mots-clés (contains_any)
    if c_type == "contains_any":
        expected = criteria.get("expected", [])
        if not any(exp.lower() in response_text.lower() for exp in expected):
            return False, f"Aucun mot-clé attendu {expected} trouvé dans la réponse"
        return True, "Conforme (mot-clé trouvé)"

    # 3. Vérification par présence de tous les mots-clés (contains_all)
    elif c_type == "contains_all":
        expected = criteria.get("expected", [])
        missing = [exp for exp in expected if exp.lower() not in response_text.lower()]
        if missing:
            return False, f"Mots-clés manquants : {missing}"
        return True, "Conforme (tous les éléments présents)"

    # 4. Vérification de format JSON strict
    elif c_type == "valid_json":
        # Extraction du bloc JSON (gère les éventuelles balises ```json ... ```)
        cleaned = response_text.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        elif cleaned.startswith("{") and "}" in cleaned:
            cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return False, "Le JSON extrait n'est pas un objet racine dict"

            required_keys = criteria.get("required_keys", [])
            missing_keys = [k for k in required_keys if k not in parsed]
            if missing_keys:
                return False, f"Clés JSON obligatoires manquantes : {missing_keys}"

            return True, f"JSON strict valide ({len(parsed)} clés vérifiées)"
        except Exception as err:
            return False, f"Impossible de parser en JSON valide : {err}"

    return True, "Critère inconnu (validé par défaut)"
