# MLOps Server : Inférence LLM Distribuée

Serveur d'inférence LLM haute performance conçu pour une architecture MLOps distribuée :
- **Station de pilotage & développement** : MacBook Air M5 (Apple Silicon, accélération Metal MPS).
- **Nœud de calcul & inférence cible** : PC Ubuntu 22.04 LTS équipé d'un GPU **AMD Radeon RX 6800 (16 Go VRAM)** avec accélération native **ROCm**.

---

## 🛠 Stack Technique

- **Gestionnaire d'environnement & paquets** : [`uv`](https://docs.astral.sh/uv/) (verrouillage strict via `uv.lock`).
- **Framework API Web** : FastAPI + Pydantic v2 (validation de schéma stricte) + Uvicorn.
- **Moteur d'inférence** : PyTorch natif (`mps` sur Apple Silicon, `cuda/rocm` sur AMD Linux, fallback `cpu`) + Hugging Face `transformers`.
- **Modèle initial** : `Qwen/Qwen2.5-0.5B-Instruct` (léger, rapide, chat template standardisé).
- **Tests** : `pytest` + `httpx` (FastAPI `TestClient`).

---

## 🚀 Démarrage Rapide (Local Mac M5)

### 1. Installation des dépendances

```bash
# Synchronise l'environnement virtuel avec uv.lock
uv sync
```

### 2. Exécution des tests automatisés

```bash
# Lance la suite de tests unitaires et de validation
uv run pytest -v
```

### 3. Lancement du serveur

```bash
# Démarre l'API sur http://0.0.0.0:8000
uv run mlops-server
```

### 4. Lancement via Docker Compose (Conteneurisé)

```bash
# Démarre l'image conteneurisée linux/amd64 avec montage du cache Hugging Face
docker compose up -d

# Vérifier les logs
docker compose logs -f

# Arrêter le conteneur
docker compose down
```

---

## 📡 Endpoints de l'API

### `GET /health`
Sonde d'observabilité renvoyant l'état du serveur et l'accélérateur matériel détecté.

**Exemple de réponse :**
```json
{
  "status": "ok",
  "device": "mps (Apple Silicon Metal)",
  "platform": "macOS-...",
  "python_version": "3.12.14"
}
```

### `POST /v1/chat`
Génération de texte par le LLM.

**Exemple de requête :**
```json
{
  "prompt": "Analyse l'action suivante : tacle en retard dans la surface.",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Exemple de réponse :**
```json
{
  "response": "Décision : Faute évidente, penalty et carton jaune.",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "latency_ms": 142.3,
  "finish_reason": "stop"
}
```

---

## 🗺 Feuille de Route MLOps

1. [x] **Phase 1 : Socle Applicatif & Local Mac M5** (FastAPI, PyTorch MPS, tests pytest, repo GitHub).
2. [x] **Phase 2 : Conteneurisation Multi-Architecture** (Dockerfile multi-stage, OrbStack Rosetta x86_64, docker-compose).
3. [ ] **Phase 3 : Accélération Matérielle AMD RX 6800** (Ubuntu 22.04, ROCm, passthrough `/dev/kfd` et `/dev/dri`).
4. [ ] **Phase 4 : CI/CD & Orchestration** (GitHub Actions, GHCR, mini-cluster k3s, IaC Terraform).
