#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 24.04 Oracle Always Free (Ampere) server.
# Run on the server:  sudo bash setup.sh
set -euo pipefail

echo "== Packages"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 iptables-persistent
systemctl enable --now docker
usermod -aG docker ubuntu

echo "== 4 GB swap (headroom for model loading on 12 GB RAM)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== Open ports 80/443 in the OS firewall (Oracle images block them by default)"
for port in 80 443; do
  iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
    || iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
done
netfilter-persistent save

echo "== Done. Log out and back in so 'docker' works without sudo."
