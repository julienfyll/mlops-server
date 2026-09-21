#!/usr/bin/env python3
"""
Script de benchmark client pour mesurer la latence, le TTFT, l'ITL et le débit (tokens/s) de l'API mlops-server.
Usage:
    uv run python scripts/benchmark.py --url http://localhost:8000 --requests 3 --tokens 64
"""

import argparse
import asyncio
import json
import time
import httpx


async def send_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
):
    """Envoie une requête de streaming SSE et mesure précisément le TTFT et l'ITL."""
    endpoint = f"{url.rstrip('/')}/v1/chat/stream"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    generated_text_chunks = []

    try:
        async with client.stream("POST", endpoint, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                return {"id": req_id, "status": f"HTTP {response.status_code}", "error": error_body.decode()}

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break

                current_time = time.perf_counter()
                if first_token_time is None:
                    first_token_time = current_time

                try:
                    chunk = json.loads(data_str)
                    token = chunk.get("token", "")
                    generated_text_chunks.append(token)
                    token_count += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_latency_ms
        decode_time_sec = (end_time - first_token_time) if first_token_time else (total_latency_ms / 1000)
        tokens_per_sec = (token_count / decode_time_sec) if decode_time_sec > 0 else 0
        avg_itl_ms = (decode_time_sec * 1000 / max(1, token_count - 1)) if token_count > 1 else 0

        return {
            "id": req_id,
            "status": "success",
            "ttft_ms": ttft_ms,
            "avg_itl_ms": avg_itl_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "tokens": token_count,
            "preview": "".join(generated_text_chunks).strip()[:60],
        }
    except Exception as e:
        return {"id": req_id, "status": "error", "error": str(e)}


async def send_sync_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
):
    """Envoie une requête synchrone classique vers /v1/chat."""
    endpoint = f"{url.rstrip('/')}/v1/chat"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    start = time.perf_counter()
    try:
        response = await client.post(endpoint, json=payload, timeout=60.0)
        elapsed_total = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            model_latency = data.get("latency_ms", elapsed_total)
            generated_text = data.get("response", "")
            token_count = max(1, len(generated_text.split()))
            tokens_per_sec = (token_count / (model_latency / 1000)) if model_latency > 0 else 0

            return {
                "id": req_id,
                "status": "success",
                "model_latency_ms": model_latency,
                "total_latency_ms": elapsed_total,
                "tokens_per_sec": tokens_per_sec,
                "tokens": token_count,
                "preview": generated_text[:60].strip(),
            }
        else:
            return {"id": req_id, "status": f"HTTP {response.status_code}", "error": response.text}
    except Exception as e:
        return {"id": req_id, "status": "error", "error": str(e)}


async def check_health(client: httpx.AsyncClient, url: str):
    """Vérifie l'état de santé et l'accélérateur matériel."""
    try:
        r = await client.get(f"{url.rstrip('/')}/health", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"❌ Impossible de joindre {url}/health : {e}")
    return None


async def fetch_metrics(client: httpx.AsyncClient, url: str):
    """Récupère et affiche les métriques Prometheus récentes."""
    try:
        r = await client.get(f"{url.rstrip('/')}/metrics", timeout=5.0)
        if r.status_code == 200:
            llm_lines = [line for line in r.text.split("\n") if line.startswith("llm_")]
            return llm_lines
    except Exception:
        pass
    return []


async def run_benchmark(url: str, num_requests: int, max_tokens: int, prompt: str, stream: bool):
    mode_label = "STREAMING SSE (/v1/chat/stream)" if stream else "SYNCHRONE (/v1/chat)"
    print("=" * 70)
    print(f"  🚀 BENCHMARK MLOPS-SERVER [{mode_label}]")
    print("=" * 70)
    print(f"Cible : {url}")

    async with httpx.AsyncClient() as client:
        health = await check_health(client, url)
        if health:
            print("✅ Serveur en ligne !")
            print(f"   • Accélérateur : {health.get('device', 'inconnu')}")
            print(f"   • Plateforme   : {health.get('platform', 'inconnue')}")
            print(f"   • Version      : {health.get('app_version', 'inconnue')}")
        else:
            print("⚠️  Attention : Le serveur ne répond pas sur /health.")

        print("-" * 70)
        print(f"Exécution de {num_requests} requête(s) (max_tokens={max_tokens})...")

        bench_start = time.perf_counter()
        results = []

        for i in range(num_requests):
            req_id = i + 1
            if stream:
                res = await send_streaming_request(client, url, prompt, max_tokens, req_id)
            else:
                res = await send_sync_request(client, url, prompt, max_tokens, req_id)
            results.append(res)

        bench_elapsed = (time.perf_counter() - bench_start) * 1000
        successful = [r for r in results if r.get("status") == "success"]

        print("\n=== RÉSULTATS DÉTAILLÉS ===")
        for r in results:
            if r["status"] == "success":
                if stream:
                    print(
                        f"Req #{r['id']:02d} : TTFT={r['ttft_ms']:.1f}ms | ITL={r['avg_itl_ms']:.1f}ms | "
                        f"Débit={r['tokens_per_sec']:.1f} tok/s | Tokens={r['tokens']} | \"{r['preview']}...\""
                    )
                else:
                    print(
                        f"Req #{r['id']:02d} : Latence={r['model_latency_ms']:.1f}ms | "
                        f"Débit={r['tokens_per_sec']:.1f} tok/s | Tokens={r['tokens']} | \"{r['preview']}...\""
                    )
            else:
                print(f"Req #{r['id']:02d} : Échec ({r.get('status')}) - {r.get('error')}")

        if successful:
            print("-" * 70)
            print("📊 SYNTHÈSE DES PERFORMANCES :")
            print(f"   • Requêtes réussies   : {len(successful)} / {num_requests}")
            if stream:
                avg_ttft = sum(r["ttft_ms"] for r in successful) / len(successful)
                avg_itl = sum(r["avg_itl_ms"] for r in successful) / len(successful)
                avg_tps = sum(r["tokens_per_sec"] for r in successful) / len(successful)
                print(f"   • TTFT moyen          : {avg_ttft:.1f} ms  (Phase Prefill)")
                print(f"   • ITL moyen           : {avg_itl:.1f} ms/token (Phase Decode)")
                print(f"   • Débit moyen         : {avg_tps:.1f} tokens/seconde")
            else:
                avg_latency = sum(r["model_latency_ms"] for r in successful) / len(successful)
                avg_tps = sum(r["tokens_per_sec"] for r in successful) / len(successful)
                print(f"   • Latence moyenne     : {avg_latency:.1f} ms")
                print(f"   • Débit moyen         : {avg_tps:.1f} tokens/seconde")
            print(f"   • Temps total du banc : {bench_elapsed:.1f} ms")

        # Lecture des métriques Prometheus finales
        metrics_lines = await fetch_metrics(client, url)
        if metrics_lines:
            print("-" * 70)
            print("📈 APERÇU DES MÉTRIQUES PROMETHEUS (/metrics) :")
            for m in metrics_lines:
                if "_sum" in m or "_count" in m or "_total" in m:
                    print(f"   {m}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark client pour mlops-server")
    parser.add_argument("--url", default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--requests", type=int, default=3, help="Nombre de requêtes de test")
    parser.add_argument("--tokens", type=int, default=64, help="Nombre maximum de tokens")
    parser.add_argument("--prompt", default="Explique en trois points pourquoi le ciel est bleu.", help="Prompt de test")
    parser.add_argument("--sync", action="store_true", help="Forcer le mode synchrone /v1/chat au lieu du streaming")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            url=args.url,
            num_requests=args.requests,
            max_tokens=args.tokens,
            prompt=args.prompt,
            stream=not args.sync,
        )
    )


if __name__ == "__main__":
    main()
