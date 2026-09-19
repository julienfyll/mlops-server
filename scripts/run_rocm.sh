#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Script de déploiement et de démarrage ROCm pour AMD Radeon RX 6800
# Machine cible : Ubuntu 22.04 LTS
# ==============================================================================

echo "=== [1/4] Vérification des prérequis matériels AMD ROCm ==="

# 1. Vérification de la présence de /dev/kfd (Kernel Fusion Driver)
if [ ! -e "/dev/kfd" ]; then
    echo "❌ ERREUR : Le périphérique /dev/kfd est introuvable."
    echo "   Vérifiez que le pilote amdgpu / rocm est bien installé sur Ubuntu 22.04."
    exit 1
fi
echo "✅ /dev/kfd détecté (canal de calcul ROCm opérationnel)."

# 2. Vérification de /dev/dri (Direct Rendering Infrastructure)
if [ ! -d "/dev/dri" ]; then
    echo "❌ ERREUR : Le dossier /dev/dri est introuvable."
    echo "   Vérifiez que votre carte graphique AMD est bien reconnue."
    exit 1
fi
echo "✅ /dev/dri détecté (accès à la mémoire VRAM opérationnel)."

# 3. Vérification des groupes render et video
CURRENT_GROUPS=$(groups)
if ! echo "$CURRENT_GROUPS" | grep -qw "render" || ! echo "$CURRENT_GROUPS" | grep -qw "video"; then
    echo "⚠️  ATTENTION : Votre utilisateur n'appartient pas aux groupes 'render' et/ou 'video'."
    echo "   Exécutez : sudo usermod -a -G render,video $USER"
    echo "   Puis redémarrez votre session avant de continuer."
fi

echo ""
echo "=== [2/4] Téléchargement de la dernière image officielle GHCR ==="
IMAGE_NAME="ghcr.io/julienfyll/mlops-server:rocm"
docker pull "$IMAGE_NAME" || {
    echo "⚠️  Impossible de tirer $IMAGE_NAME depuis GHCR (image en cours de build ou privée)."
    echo "   Tentative d'utilisation de l'image locale si elle existe..."
}

echo ""
echo "=== [3/4] Démarrage du conteneur avec accélération RX 6800 ==="
CONTAINER_NAME="mlops-server"

# Arrêt d'une éventuelle instance précédente
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Arrêt de l'ancien conteneur $CONTAINER_NAME..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Récupération dynamique des GID de l'hôte pour garantir l'accès direct aux périphériques
RENDER_GID=$(getent group render | cut -d: -f3 || echo "110")
VIDEO_GID=$(getent group video | cut -d: -f3 || echo "44")

# Paramètres configurables via variables d'environnement (avec valeurs par défaut saines)
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
PORT="${PORT:-8000}"

# Lancement du conteneur
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "$VIDEO_GID" \
  --group-add "$RENDER_GID" \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
  -e HOST=0.0.0.0 \
  -e PORT="$PORT" \
  -e HF_HUB_DISABLE_XET=1 \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/.cache/huggingface:/home/appuser/.cache/huggingface" \
  "$IMAGE_NAME"

echo ""
echo "=== [4/4] Vérification du statut ==="
sleep 3
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✅ Le conteneur $CONTAINER_NAME est démarré avec succès !"
    echo "   Accès à l'API : http://localhost:8000/health"
    echo "   Logs en direct : docker logs -f $CONTAINER_NAME"
else
    echo "❌ Le conteneur n'a pas pu démarrer. Consultez les logs avec :"
    echo "   docker logs $CONTAINER_NAME"
    exit 1
fi
