#!/usr/bin/env bash
# ==============================================================================
# Script de pilotage à distance du serveur GPU MLOps sur k3s
# Usage :
#   ./scripts/gpu.sh start   -> Alloue le GPU et démarre le serveur (en 2s)
#   ./scripts/gpu.sh stop    -> Éteint le serveur et libère 100% de la VRAM
#   ./scripts/gpu.sh status  -> Affiche l'état du Pod sur la machine Linux
# ==============================================================================

set -euo pipefail

REMOTE_HOST="gpu-server"
SERVER_URL="http://192.168.1.22:30080"

case "${1:-}" in
    start)
        echo "🚀 Démarrage du serveur GPU sur k3s..."
        ssh -o BatchMode=yes "$REMOTE_HOST" 'export KUBECONFIG=~/.kube/config && kubectl scale deployment mlops-server --replicas=1'
        echo "⏳ Attente de l'initialisation du modèle sur la Radeon RX 6800..."
        ssh -o BatchMode=yes "$REMOTE_HOST" 'export KUBECONFIG=~/.kube/config && kubectl rollout status deployment/mlops-server --timeout=60s'
        echo "✅ Serveur GPU prêt !"
        echo "   • Route de santé : ${SERVER_URL}/health"
        curl -s "${SERVER_URL}/health" || true
        echo ""
        ;;
    stop)
        echo "🛑 Arrêt du serveur GPU sur k3s..."
        ssh -o BatchMode=yes "$REMOTE_HOST" 'export KUBECONFIG=~/.kube/config && kubectl scale deployment mlops-server --replicas=0'
        echo "✅ Serveur arrêté. 100% de la VRAM et du GPU sont libérés pour tes jeux et applications !"
        ;;
    status)
        echo "📊 Statut du serveur GPU sur k3s :"
        ssh -o BatchMode=yes "$REMOTE_HOST" 'export KUBECONFIG=~/.kube/config && kubectl get pods -l app=mlops-server -o wide'
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
