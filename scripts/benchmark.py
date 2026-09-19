#!/usr/bin/env python3
"""
Script de benchmark client pour mesurer la latence et le débit (tokens/s) de l'API mlops-server.
Usage:
    uv run python scripts/benchmark.py --url http://localhost:8000 --requests 3 --tokens 128
"""

import argparse
import asyncio
import time
import httpx

async def send_inference_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int, req_id: int):
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
            # Approximation du nombre de tokens (1 token ≈ 0.75 mot)
            token_count = max(1, len(generated_text.split()))
            tokens_per_sec = (token_count / (model_latency / 1000)) if model_latency > 0 else 0
            
            return {
                "id": req_id,
                "status": "success",
                "model_latency_ms": model_latency,
                "total_latency_ms": elapsed_total,
                "tokens_per_sec": tokens_per_sec,
                "tokens": token_count,
            }
        else:
            return {"id": req_id, "status": f"HTTP {response.status_code}", "error": response.text}
    except Exception as e:
        return {"id": req_id, "status": "error", "error": str(e)}

async def check_health(client: httpx.AsyncClient, url: str):
    endpoint = f"{url.rstrip('/')}/health"
    try:
        r = await client.get(endpoint, timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"❌ Impossible de joindre {endpoint} : {e}")
    return None

async def run_benchmark(url: str, num_requests: int, max_tokens: int, prompt: str):
    print("=" * 60)
    print("  🚀 BENCHMARK MLOPS-SERVER (LATENCE & DÉBIT INFERENCE)")
    print("=" * 60)
    print(f"Cible : {url}")
    
    async with httpx.AsyncClient() as client:
        health = await check_health(client, url)
        if health:
            print(f"✅ Serveur en ligne !")
            print(f"   • Accélérateur : {health.get('device', 'inconnu')}")
            print(f"   • Plateforme   : {health.get('platform', 'inconnue')}")
        else:
            print("⚠️  Attention : Le serveur ne répond pas sur /health.")
        
        print("-" * 60)
        print(f"Envoi de {num_requests} requête(s) (max_tokens={max_tokens})...")
        
        tasks = [
            send_inference_request(client, url, prompt, max_tokens, i + 1)
            for i in range(num_requests)
        ]
        
        bench_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        bench_elapsed = (time.perf_counter() - bench_start) * 1000
        
        successful = [r for r in results if r.get("status") == "success"]
        
        print("\n=== RÉSULTATS DÉTAILLÉS ===")
        for r in results:
            if r["status"] == "success":
                print(f"Req #{r['id']:02d} : Latence={r['model_latency_ms']:.1f}ms | Débit={r['tokens_per_sec']:.1f} tok/s | Tokens={r['tokens']}")
            else:
                print(f"Req #{r['id']:02d} : Échec ({r.get('status')}) - {r.get('error')}")
                
        if successful:
            avg_latency = sum(r["model_latency_ms"] for r in successful) / len(successful)
            avg_throughput = sum(r["tokens_per_sec"] for r in successful) / len(successful)
            print("-" * 60)
            print(f"📊 SYNTHÈSE :")
            print(f"   • Requêtes réussies   : {len(successful)} / {num_requests}")
            print(f"   • Latence moyenne     : {avg_latency:.1f} ms")
            print(f"   • Débit moyen         : {avg_throughput:.1f} tokens/seconde")
            print(f"   • Temps total du test : {bench_elapsed:.1f} ms")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Benchmark client pour mlops-server")
    parser.add_argument("--url", default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--requests", type=int, default=3, help="Nombre de requêtes de test")
    parser.add_argument("--tokens", type=int, default=128, help="Nombre maximum de tokens")
    parser.add_argument("--prompt", default="Explique les principes du jeu en triangle au football.", help="Prompt")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.url, args.requests, args.tokens, args.prompt))

if __name__ == "__main__":
    main()
