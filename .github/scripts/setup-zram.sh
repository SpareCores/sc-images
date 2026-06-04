#!/usr/bin/env bash
# Configure zram swap via Ubuntu zram-tools (PERCENT = % of RAM, e.g. 125 → 1.25× RAM).
set -euo pipefail

PERCENT="${ZRAM_PERCENT:-125}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "setup-zram: no apt-get; skipping on non-Debian runner" >&2
  exit 0
fi

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zram-tools

if [ -f /etc/default/zramswap ]; then
  if grep -q '^PERCENT=' /etc/default/zramswap; then
    sudo sed -i "s/^PERCENT=.*/PERCENT=${PERCENT}/" /etc/default/zramswap
  else
    echo "PERCENT=${PERCENT}" | sudo tee -a /etc/default/zramswap >/dev/null
  fi
  if ! grep -q '^ALGO=' /etc/default/zramswap; then
    echo 'ALGO=zstd' | sudo tee -a /etc/default/zramswap >/dev/null
  fi
else
  sudo tee /etc/default/zramswap >/dev/null <<EOF
ALGO=zstd
PERCENT=${PERCENT}
PRIORITY=100
EOF
fi

# Remove file-backed swap from older workflow revisions if present.
if swapon --show 2>/dev/null | grep -q '/swapfile'; then
  sudo swapoff /swapfile 2>/dev/null || true
fi

sudo systemctl enable zramswap
sudo systemctl restart zramswap

echo "zram configured (PERCENT=${PERCENT}):"
swapon --show
free -h
