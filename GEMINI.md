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
- **Principe de Vérification Systématique (Zéro Déchet Non Testé)** :
  - Tout nouvel ajout ou fichier de configuration créé dans la base de code (`Dockerfile`, `docker-compose.yml`, scripts, manifests) doit être OBLIGATOIREMENT testé et validé par une commande d'exécution réelle avant de clore l'étape.
  - Ne jamais se fier à la seule validité syntaxique théorique : vérifier le cycle complet (démarrage, test d'appel/santé, arrêt propre).
- **Protocole de Transparence Systématique Avant Action** :
  - Avant de déclencher toute commande terminal ou modification de fichier dans le projet, afficher obligatoirement un encadré explicitant :
    1. **La commande exacte** ou le fichier ciblé.
    2. **Le but précis** de l'opération (pourquoi cette action est nécessaire).
    3. **Le résultat attendu en retour** (ce que la commande va produire ou afficher).
  - Aucune action ne doit être lancée sans cette visibilité préalable.


