#!/usr/bin/env bash
# Configure zram swap via Ubuntu zram-tools (PERCENT = % of RAM, e.g. 125 → 1.25× RAM).
set -euo pipefail

PERCENT="${ZRAM_PERCENT:-125}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "setup-zram: no apt-get; skipping on non-Debian runner" >&2
  exit 0
fi

zram_kernel_available() {
  if [ -d /sys/module/zram ] || [ -b /dev/zram0 ]; then
    return 0
  fi
  if modinfo zram &>/dev/null 2>&1; then
    sudo modprobe zram 2>/dev/null && return 0
  fi
  return 1
}

# Azure / GHA kernels build CONFIG_ZRAM=m but ship zram.ko in linux-modules-extra-*.
# https://bugs.launchpad.net/ubuntu/bionic/+source/linux-azure/+bug/1762756
install_zram_kernel_modules() {
  local kver pkg
  kver=$(uname -r)
  pkg="linux-modules-extra-${kver}"
  if dpkg -s "$pkg" &>/dev/null 2>&1; then
    return 0
  fi
  echo "setup-zram: installing ${pkg} for zram.ko"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg"
}

ensure_zram_kernel() {
  if zram_kernel_available; then
    return 0
  fi
  sudo apt-get update -qq
  if install_zram_kernel_modules && zram_kernel_available; then
    return 0
  fi
  return 1
}

if ! ensure_zram_kernel; then
  echo "setup-zram: zram unavailable on kernel $(uname -r); keeping existing swap" >&2
  echo "  (install linux-modules-extra-$(uname -r) manually if the package exists in apt)" >&2
  swapon --show 2>/dev/null || echo "  (no swap active)"
  exit 0
fi

# Avoid deb-systemd-invoke failures during package configure on GHA runners.
sudo SYSTEMD_OFFLINE=1 DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zram-tools

if [ -f /etc/default/zramswap ]; then
  if grep -q '^PERCENT=' /etc/default/zramswap; then
    sudo sed -i "s/^PERCENT=.*/PERCENT=${PERCENT}/" /etc/default/zramswap
  else
    echo "PERCENT=${PERCENT}" | sudo tee -a /etc/default/zramswap >/dev/null
  fi
  if grep -q '^ENABLED=' /etc/default/zramswap; then
    sudo sed -i 's/^ENABLED=.*/ENABLED=true/' /etc/default/zramswap
  else
    echo 'ENABLED=true' | sudo tee -a /etc/default/zramswap >/dev/null
  fi
  if ! grep -q '^ALGO=' /etc/default/zramswap; then
    echo 'ALGO=zstd' | sudo tee -a /etc/default/zramswap >/dev/null
  fi
else
  sudo tee /etc/default/zramswap >/dev/null <<EOF
ALGO=zstd
PERCENT=${PERCENT}
PRIORITY=100
ENABLED=true
EOF
fi

swapoff_existing() {
  # GHA runners ship with file-backed swap; zramswap cannot start until it is off.
  if swapon --show --noheadings 2>/dev/null | grep -q .; then
    echo "Disabling existing swap before zram:"
    swapon --show
    sudo swapoff --all 2>/dev/null || true
  fi
}

start_zram_via_service() {
  swapoff_existing
  if [ -x /usr/sbin/zramswap ]; then
    sudo /usr/sbin/zramswap stop 2>/dev/null || true
    echo "Starting zram via /usr/sbin/zramswap start"
    sudo /usr/sbin/zramswap start
    return 0
  fi
  echo "Starting zram via systemctl"
  sudo systemctl start zramswap
}

start_zram_manual() {
  echo "Starting zram via zramctl (fallback)"
  swapoff_existing
  sudo modprobe zram
  local mem_kb size_mb
  mem_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  size_mb=$(( mem_kb * PERCENT / 100 / 1024 ))
  [ "$size_mb" -gt 0 ] || size_mb=1
  for z in /dev/zram*; do
    [ -e "$z" ] || continue
    sudo zramctl --reset "$z" 2>/dev/null || true
  done
  local zdev
  zdev=$(sudo zramctl --find --size "${size_mb}M" --algo zstd)
  sudo mkswap "$zdev"
  sudo swapon -p 100 "$zdev"
}

if start_zram_via_service; then
  :
elif start_zram_manual; then
  :
else
  echo "setup-zram: failed to enable zram" >&2
  systemctl status zramswap --no-pager 2>/dev/null || true
  journalctl -u zramswap -n 40 --no-pager 2>/dev/null || true
  exit 1
fi

if ! swapon --show 2>/dev/null | grep -qE 'zram|/dev/zram'; then
  echo "setup-zram: no zram device in swapon output" >&2
  swapon --show || true
  exit 1
fi

echo "zram configured (PERCENT=${PERCENT}):"
swapon --show
free -h
