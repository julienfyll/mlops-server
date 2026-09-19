# Directives du Projet : Déploiement Serveur & Pipeline MLOps

## 1. Objectif
Construire un pipeline MLOps distribué de bout en bout :
- **Environnement de développement** : MacBook Air M5 (Apple Silicon).
- **Environnement de calcul cible** : PC Linux AMD (GPU 16 Go VRAM, accélération matérielle ROCm).
- **Application** : API d'inférence LLM générique et performante.

## 2. Stack Technique Validée
- **Gestionnaire d'environnement & paquets** : `uv` (`uv add`, `uv run`, `uv sync`).
- **Framework API Web** : FastAPI + Pydantic (validation stricte) + Uvicorn (serveur ASGI).
- **Moteur d'inférence IA** : Inférence native avec PyTorch + Hugging Face `transformers`.
- **Méthodologie** : Approche modulaire et itérative.

## 3. Structure du Répertoire
- Répertoire racine : `/Users/julien_fyll/PROJET/DeploiementServeur`
- Code applicatif dans `src/`
- Configuration et locks gérés via `pyproject.toml` et `uv.lock`

## 4. Protocole de Pair-Programming Pédagogique
- **Explication ciblée des enjeux** : Pour chaque décision technique ou commande structurante, exposer de manière concise et équilibrée :
  1. Le rôle exact de l'outil.
  2. L'enjeu technique majeur (reproductibilité, performance, matériel).
  3. Le compromis (trade-off) clé.
- **Rythme pas-à-pas & Arrêt obligatoire** :
  - Ne JAMAIS enchaîner deux étapes consécutives sans marquer une pause.
  - À la fin de chaque étape exécutée, fournir un compte-rendu clair des actions réalisées et de leurs impacts.
  - Présenter systématiquement le détail de l'étape suivante (objectifs, fichiers concernés, logique du code ou commandes prévues).
  - Attendre impérativement le retour et l'accord explicite de l'utilisateur avant de passer à l'étape suivante.
