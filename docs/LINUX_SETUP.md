# Guide de Déploiement & Configuration Hôte : AMD Radeon RX 6800 (Ubuntu 22.04 LTS)

Ce guide décrit la procédure pas-à-pas pour configurer une machine hôte Linux avec une carte graphique **AMD Radeon RX 6800** (architecture RDNA 2, silicium Navi 21, ISA `gfx1030`) afin de faire tourner l'API d'inférence LLM via notre conteneur Docker ROCm.

---

## 1. Architecture & Principe Fondateur

> [!IMPORTANT]
> **Zéro installation ROCm sur le système hôte.**
> Toute la suite logicielle complexe (PyTorch ROCm, bibliothèques mathématiques HIP, rocBLAS, MIOpen) est isolée dans l'image Docker `ghcr.io/julienfyll/mlops-server:rocm`.
> L'OS hôte a uniquement besoin de son pilote noyau standard open-source (`amdgpu.ko`) et de Docker Engine.

```
┌─────────────────────────────────────────────────────────────┐
│ CONTENEUR DOCKER (ghcr.io/julienfyll/mlops-server:rocm)      │
│  - PyTorch 2.6+ ROCm 6.2                                    │
│  - Bibliothèques HIP, rocBLAS, MIOpen                       │
│  - HSA_OVERRIDE_GFX_VERSION=10.3.0                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ Accès direct via /dev/kfd & /dev/dri
┌───────────────────────────▼─────────────────────────────────┐
│ SYSTÈME HÔTE LINUX (Ubuntu 22.04 LTS)                       │
│  - Pilote noyau open-source standard (amdgpu.ko)             │
│  - Docker Engine standard                                   │
│  - Utilisateur membre des groupes 'render' et 'video'       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Étape 1 : Diagnostic Matériel & Noyau (Lecture Seule)

Vérifier que le GPU est physiquement détecté et que le pilote noyau standard est actif :

```bash
lspci -nnk | grep -i -EA3 "vga|3d|display"
```
* **Résultat attendu** :
  ```text
  03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21 [Radeon RX 6800/6800 XT / 6900 XT] [1002:73bf]
          Subsystem: ...
          Kernel driver in use: amdgpu
          Kernel modules: amdgpu
  ```

Vérifier la présence des interfaces de calcul du noyau :
```bash
ls -l /dev/kfd /dev/dri
```
* **Résultat attendu** :
  - `/dev/kfd` : Nœud de calcul HSA (Heterogeneous System Architecture).
  - `/dev/dri/card*` et `/dev/dri/renderD*` : Accès à la mémoire vidéo (VRAM) et rendu.

---

## 3. Étape 2 : Configuration des Permissions Utilisateur

Par défaut, l'accès direct aux périphériques `/dev/kfd` et `/dev/dri` nécessite l'appartenance aux groupes `render` et `video`.

```bash
# Ajouter l'utilisateur courant aux groupes requis
sudo usermod -aG render,video $USER

# Appliquer immédiatement les groupes à la session en cours
newgrp render
newgrp video
```
*(Alternative : fermer et rouvrir la session utilisateur).*

---

## 4. Étape 3 : "Smoke Test" Docker ROCm

Avant de lancer le serveur complet, valider que Docker peut communiquer avec le GPU et que PyTorch détecte la RX 6800 :

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ghcr.io/julienfyll/mlops-server:rocm \
  python3 -c "import torch; print('CUDA/ROCm disponible ?', torch.cuda.is_available()); print('Nom du GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"
```

* **Résultat attendu** :
  ```text
  CUDA/ROCm disponible ? True
  Nom du GPU : AMD Radeon RX 6800
  ```

---

## 5. Étape 4 : Démarrage Opérationnel du Serveur

Cloner le projet et lancer le conteneur via le script automatisé :

```bash
# 1. Cloner le dépôt
git clone git@github.com:julienfyll/mlops-server.git
cd mlops-server

# 2. Lancer le conteneur
./scripts/run_rocm.sh
```

Vérifier la santé du serveur :
```bash
curl http://localhost:8000/health
```

Tester une génération de texte :
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explique la photosynthese en 2 phrases", "max_new_tokens": 50}'
```

---

## 6. Boîte à Outils de Débogage & Diagnostic

Contrairement à une idée reçue, un conteneur est **beaucoup plus simple à débugger** qu'une installation locale :
1. **L'environnement est déterministe** : aucun risque de conflit avec des paquets Python ou des bibliothèques système locales.
2. **L'hôte reste intact** : aucun risque de casser l'affichage graphique ou le système.

### A. Ouvrir un shell interactif dans le conteneur
Pour investiguer manuellement en direct :
```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ghcr.io/julienfyll/mlops-server:rocm \
  bash
```

### B. Outils d'inspection matérielle à l'intérieur du conteneur
Une fois dans le conteneur (ou via `docker run`) :
- **Vérifier les unités de calcul détectées par ROCm** :
  ```bash
  rocminfo | grep "Name:"
  ```
- **Vérifier l'utilisation de la VRAM, la température et la charge GPU** :
  ```bash
  rocm-smi
  ```

### C. Activer les logs de diagnostic avancés
En cas de comportement inattendu ou de crash PyTorch :
- `AMD_LOG_LEVEL=3` : Affiche chaque appel système HIP/ROCm.
- `TORCH_SHOW_CPP_STACKTRACES=1` : Affiche la pile d'appels C++ complète en cas d'exception.
- `HIP_VISIBLE_DEVICES=0` : Force l'utilisation exclusive du GPU principal.

Exemple :
```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e AMD_LOG_LEVEL=3 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  ghcr.io/julienfyll/mlops-server:rocm \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

### D. Consulter les logs du conteneur en cours d'exécution
```bash
docker logs -f mlops-server-rocm
```
