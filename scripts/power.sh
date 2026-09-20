#!/usr/bin/env bash
# ==============================================================================
# Script de gestion énergétique à distance (Wake-on-LAN & Extinction propre)
# Usage :
#   ./scripts/power.sh on      -> Réveille le PC Linux via le Paquet Magique (WoL)
#   ./scripts/power.sh off     -> Éteint proprement le PC Linux à distance
#   ./scripts/power.sh status  -> Vérifie si le PC est allumé ou éteint
# ==============================================================================

set -euo pipefail

MAC_ADDRESS="d8:43:ae:25:1a:df"
TAILSCALE_IP="100.93.198.49"
LOCAL_IP="192.168.1.22"
REMOTE_HOST="gpu-server"

check_online() {
    # Teste d'abord le ping rapide sur l'IP Tailscale, puis locale
    if ping -c 1 -W 1 "$TAILSCALE_IP" >/dev/null 2>&1 || ping -c 1 -W 1 "$LOCAL_IP" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

case "${1:-}" in
    on)
        echo "⚡ Envoi du Paquet Magique Wake-on-LAN à la carte réseau..."
        echo "   • Adresse MAC cible : $MAC_ADDRESS"

        # Utilise Python standard (disponible nativement sur macOS sans installer d'outil tiers)
        python3 - <<EOF
import socket

mac = "$MAC_ADDRESS".replace(":", "").replace("-", "")
mac_bytes = bytes.fromhex(mac)
magic_packet = b"\xff" * 6 + mac_bytes * 16

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Diffusion sur le broadcast global et sur le sous-réseau local (port 9)
for target in ["255.255.255.255", "192.168.1.255"]:
    try:
        sock.sendto(magic_packet, (target, 9))
    except Exception as e:
        pass
sock.close()
EOF

        echo "📡 Paquet magique broadcasté sur le réseau local !"
        echo "⏳ Attente du démarrage de la machine (détection réseau)..."

        for i in {1..30}; do
            if check_online; then
                echo ""
                echo "🟢 Le PC Linux a démarré et répond sur le réseau !"
                echo "   • IP Tailscale : $TAILSCALE_IP"
                exit 0
            fi
            printf "."
            sleep 2
        done

        echo ""
        echo "⚠️ Le PC n'a pas encore répondu au bout de 60s."
        echo "   Note : Assure-toi que dans ton BIOS MSI, l'option suivante est active :"
        echo "   Settings -> Advanced -> Wake Up Event Setup -> 'Resume by PCI-E Device' = [Enabled]"
        ;;

    off)
        echo "🛑 Arrêt propre du PC Linux à distance..."
        if ! check_online; then
            echo "ℹ️ La machine semble déjà éteinte ou inaccessible."
            exit 0
        fi

        # Arrêt sécurisé via sudoers configuré sans mot de passe
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" "sudo poweroff" || true
        echo "✅ Ordre d'extinction envoyé. Le système se synchronise et s'éteint proprement."
        ;;

    status)
        echo "🔍 Vérification de l'état du PC Linux..."
        if check_online; then
            echo "🟢 Statut : EN LIGNE (Allumé)"
            echo "   • IP Tailscale : $TAILSCALE_IP"
            echo "   • IP LAN       : $LOCAL_IP"
            echo "   • Accès SSH    : ssh $REMOTE_HOST"
        else
            echo "🔴 Statut : HORS LIGNE (Éteint ou en veille)"
            echo "   • Pour l'allumer : ./scripts/power.sh on"
        fi
        ;;

    *)
        echo "Usage: $0 {on|off|status}"
        exit 1
        ;;
esac
