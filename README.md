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

### 4. Lancement via Docker Compose (CPU / Dev)

```bash
# Démarre l'image conteneurisée linux/amd64 avec montage du cache Hugging Face
docker compose up -d

# Vérifier les logs
docker compose logs -f

# Arrêter le conteneur
docker compose down
```

---

## ⚡ Déploiement Accéléré sur GPU AMD (ROCm & Docker)

> [!TIP]
> **Zéro installation ROCm sur l'hôte** : Grâce à notre conteneur Docker optimisé (`ghcr.io/julienfyll/mlops-server:rocm`), la machine hôte Linux ne nécessite aucun paquet ROCm lourd ou instable. Seul le pilote noyau standard (`amdgpu.ko`) et Docker sont requis.

### 1. Préparation de la machine hôte
Consultez le guide détaillé pas-à-pas : [`docs/LINUX_SETUP.md`](docs/LINUX_SETUP.md).

### 2. Démarrage Automatique & Dynamique
Le script [`scripts/run_rocm.sh`](scripts/run_rocm.sh) détecte automatiquement les permissions matérielles (`/dev/kfd`, `/dev/dri`, GID `render` et `video`) et monte le cache local Hugging Face (`~/.cache/huggingface`) :

```bash
# Lancement standard (Radeon RX 6800 - RDNA 2, port 8000 par défaut)
./scripts/run_rocm.sh

# Surcharge dynamique (autre architecture AMD GPU ou port personnalisé)
HSA_OVERRIDE_GFX_VERSION=11.0.0 PORT=9000 ./scripts/run_rocm.sh
```

### 3. Banc de Test & Benchmark de Charge
Pour évaluer les performances de génération en temps réel depuis une machine distante :

```bash
# Lance 10 requêtes concurrentes vers le serveur GPU
uv run python scripts/benchmark.py --url http://<IP_SERVEUR>:8000 --requests 10 --tokens 64
```

---

## 📡 Endpoints de l'API

### `GET /health`
Sonde d'observabilité renvoyant l'état du serveur et l'accélérateur matériel détecté.

**Exemple de réponse (GPU AMD Radeon RX 6800) :**
```json
{
  "status": "ok",
  "device": "cuda (GPU: AMD Radeon Graphics)",
  "platform": "Linux-6.8.0-138-generic-x86_64-with-glibc2.36",
  "python_version": "3.12.14"
}
```

### `POST /v1/chat`
Génération de texte par le LLM.

**Exemple de requête :**
```json
{
  "prompt": "Explique pourquoi le ciel est bleu en une phrase.",
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Exemple de réponse :**
```json
{
  "response": "Le ciel est bleu car les particules d'étoiles qui font l'air dans l'atmosphère terrestre, réfléchissent la lumière du soleil à travers leur surface et le reflètent dans notre vue.",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "latency_ms": 1296.0,
  "finish_reason": "stop"
}
```

---

## ☸️ Orchestration Kubernetes (k3s)

L'application est orchestrée sur un cluster Kubernetes **k3s** avec réservation matérielle exclusive du GPU AMD (`amd.com/gpu: 1`), stratégie `Recreate` et auto-guérison (*self-healing*) :

```bash
# 1. Déployer le plugin matériel AMD GPU
kubectl apply -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml

# 2. Déployer le serveur LLM et son service NodePort
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml

# 3. Vérifier le statut du Pod (1/1 Running sur GPU AMD)
kubectl get pods -l app=mlops-server -o wide

# 4. Interroger l'API via le port NodePort 30080
curl http://<IP_SERVEUR>:30080/health
```

---

## ⚡ Gestion Énergétique & Accès Distant (Tailscale & WoL)

Le projet intègre une gestion énergétique intelligente pour machine hybride permettant de réduire la consommation à **0,5 Watt** au repos tout en offrant un accès universel chiffré :

```bash
# 1. Vérifier si le serveur est en ligne
./scripts/power.sh status

# 2. Allumer le PC à distance depuis le Mac (Wake-on-LAN)
./scripts/power.sh on

# 3. Démarrer le serveur GPU sur k3s (alloue les 16 Go de VRAM)
./scripts/gpu.sh start

# 4. Arrêter le serveur GPU (libère 100% de la VRAM pour le gaming/bureautique)
./scripts/gpu.sh stop

# 5. Éteindre proprement le PC à distance (consommation 0,5W)
./scripts/power.sh off
```

> [!NOTE]
> Grâce au réseau maillé **Tailscale (WireGuard)**, le pilotage SSH et les appels d'inférence LLM (`http://100.93.198.49:30080`) fonctionnent depuis n'importe quelle connexion Internet (extérieur, 4G/5G) sans ouverture de port sur la box.

---

## 🗺 Feuille de Route MLOps

1. [x] **Phase 1 : Socle Applicatif & Local Mac M5** (FastAPI, PyTorch MPS, tests pytest, repo GitHub).
2. [x] **Phase 2 : Conteneurisation Multi-Architecture** (Dockerfile multi-stage, image CPU légère 369 Mo, docker-compose).
3. [x] **Phase 3 : CI/CD GitHub Actions & Publication GHCR** (Build & Push automatique des images CPU et ROCm 6.2).
4. [x] **Phase 4 : Accélération Matérielle AMD RX 6800** (Ubuntu 22.04, passthrough `/dev/kfd` & `/dev/dri`, benchmark 35 tok/s).
5. [x] **Phase 5 : Orchestration Kubernetes (k3s)** (k3s, AMD GPU Device Plugin, manifests Deployment & Service, auto-healing validé).
6. [x] **Phase 6 : Déploiement Continu Automatisé (Keel)** (Détection de nouveau SHA sur GHCR, polling @every 1m, stratégie Recreate mono-GPU).
7. [x] **Phase 7 : Réseau Zéro-Trust & Accès Universel (Tailscale)** (Tunnel chiffré WireGuard, IP universelle `100.93.198.49`).
8. [x] **Phase 8 : Gestion Énergétique & Wake-on-LAN (WoL)** (Allumage à distance par paquet magique et arrêt propre sans mot de passe).

---

## 🔭 Perspectives Futures (DevOps, Platform Engineering & AI Engineering)

- [ ] **Ingénierie de l'IA Générative (Contrôle du Non-Déterminisme)** : Décodage contraint (Outlines / JSON Schema strict), évaluations automatisées (Evals) et fine-tuning LoRA.
- [ ] **Infrastructure as Code (Terraform / OpenTofu)** : Définition déclarative de l'infrastructure et de l'environnement système.
- [ ] **Gestion des Secrets & DevSecOps (SOPS / Vault)** : Chiffrement des identifiants et clés d'accès sans rien exposer dans Git.
