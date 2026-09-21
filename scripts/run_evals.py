#!/usr/bin/env python3
"""
Moteur d'évaluations automatisées (LLM Evals Runner) pour mlops-server et ai-gateway.
Exécute le Golden Dataset contre une API d'inférence, évalue la conformité sémantique
et sert de barrière de qualité (Quality Gate) en CI/CD.

Usage:
    uv run python scripts/run_evals.py --url http://localhost:8080
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple
import httpx


from mlops_server.evals import evaluate_response


def run_evals(base_url: str, dataset_path: str, timeout: float = 30.0) -> bool:
    """Exécute la suite complète d'évaluations et affiche les métriques."""
    if not os.path.exists(dataset_path):
        print(f"❌ Fichier Golden Dataset introuvable : {dataset_path}")
        return False

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    print("=" * 75)
    print(f"  🧪 MLOPS EVALUATION RUNNER - GOLDEN DATASET ({len(dataset)} CAS)")
    print(f"  Cible : {base_url}")
    print("=" * 75)

    results = []
    passed_count = 0
    total_latency_ms = 0.0

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        # Vérification initiale de la cible
        try:
            health = client.get("/health")
            if health.status_code != 200:
                print(f"⚠️ Avertissement : /health a renvoyé HTTP {health.status_code}")
        except Exception as e:
            print(f"❌ Impossible de joindre l'API sur {base_url} : {e}")
            return False

        for i, item in enumerate(dataset, 1):
            eval_id = item["id"]
            name = item["name"]
            prompt = item["prompt"]
            max_tokens = item.get("max_tokens", 50)
            temperature = item.get("temperature", 0.0)
            criteria = item.get("criteria", {})

            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            t0 = time.perf_counter()
            try:
                resp = client.post("/v1/chat", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_latency_ms += elapsed_ms

                if resp.status_code == 200:
                    data = resp.json()
                    output_text = data.get("response", "")
                    passed, reason = evaluate_response(output_text, criteria)
                else:
                    output_text = f"HTTP {resp.status_code}"
                    passed, reason = False, f"Erreur HTTP {resp.status_code} ({resp.text[:60]})"

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                output_text = "ERREUR RÉSEAU"
                passed, reason = False, str(exc)

            if passed:
                passed_count += 1
                status_icon = "✅ PASS"
            else:
                status_icon = "❌ FAIL"

            preview = output_text.replace("\n", " ").strip()
            if len(preview) > 40:
                preview = preview[:37] + "..."

            results.append({
                "id": eval_id,
                "name": name,
                "status": status_icon,
                "latency_ms": elapsed_ms,
                "reason": reason,
                "preview": preview,
                "passed": passed,
            })

            print(f"[{i}/{len(dataset)}] {status_icon} | {name:<26} | {elapsed_ms:>6.1f}ms | {reason}")

    # Rapport final
    pass_rate = (passed_count / len(dataset)) * 100
    avg_latency = total_latency_ms / len(dataset) if dataset else 0.0

    print("-" * 75)
    print("📊 BILAN DE LA SUITE D'ÉVALUATIONS :")
    print(f"   • Taux de réussite (Pass Rate) : {passed_count} / {len(dataset)} ({pass_rate:.1f}%)")
    print(f"   • Latence moyenne par inférence : {avg_latency:.1f} ms")
    print("=" * 75)

    if pass_rate == 100.0:
        print("🎉 SUCCÈS : Toutes les évaluations sémantiques sont conformes ! Quality Gate VALIDÉ.")
        return True
    else:
        print(f"🚫 ÉCHEC : {len(dataset) - passed_count} test(s) ont échoué. Déploiement bloqué.")
        return False


def main():
    parser = argparse.ArgumentParser(description="LLM Semantic Evals Runner")
    parser.add_argument(
        "--url",
        default=os.getenv("EVAL_URL", "http://localhost:8080"),
        help="URL de base de l'API (défaut: http://localhost:8080)",
    )
    parser.add_argument(
        "--dataset",
        default="tests/evals/golden_dataset.json",
        help="Chemin vers le fichier JSON du Golden Dataset",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout en secondes par requête d'évaluation",
    )
    args = parser.parse_args()

    success = run_evals(base_url=args.url, dataset_path=args.dataset, timeout=args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
