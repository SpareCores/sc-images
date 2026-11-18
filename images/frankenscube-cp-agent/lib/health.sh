#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# Health pings (only for URLs present in health.pings), fsync/steal check, SDN pings.

# percentile <p> <file-with-numbers>: nearest-rank percentile (one number per line).
percentile() {
  local p=$1 file=$2 n idx
  n=$(wc -l <"$file")
  [ "$n" -gt 0 ] || return 1
  idx=$(((p * n + 99) / 100))
  [ "$idx" -ge 1 ] || idx=1
  sort -n "$file" | sed -n "${idx}p"
}

# steal_pct <stat-before> <stat-after>: CPU steal in percent between two /proc/stat samples.
steal_pct() {
  awk 'FNR == 1 && NR == 1 { for (i = 2; i <= NF; i++) { b[i] = $i; bt += $i }; bs = $9 }
       FNR == 1 && NR > 1 { for (i = 2; i <= NF; i++) { at += $i }; as = $9 }
       END { dt = at - bt; if (dt <= 0) { print 0 } else { printf "%.1f\n", (as - bs) * 100 / dt } }' \
    <(grep -m1 '^cpu ' "$1") <(grep -m1 '^cpu ' "$2")
}

# fsync_samples <dir> <count>: wall-clock milliseconds of <count> synchronous 4 KiB writes.
fsync_samples() {
  local dir=$1 count=$2 i t0 t1 f
  f=$(mktemp -p "$dir")
  for ((i = 0; i < count; i++)); do
    t0=$EPOCHREALTIME
    dd if=/dev/zero of="$f" bs=4096 count=1 oflag=dsync conv=notrunc status=none
    t1=$EPOCHREALTIME
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f\n", (b - a) * 1000 }'
  done
  rm -f "$f"
}

# The check runs every 5 min (every 10th full cycle); bad for 15 min in a row raises the ALERT
# that triggers the GP fallback.
FSYNC_P99_LIMIT_MS=25
STEAL_LIMIT_PCT=10
HEALTH_BAD_SECONDS=900

_bad_since() {
  local name=$1 bad=$2 f="$AGENT_RUN/bad-$1" now since
  now=$(fc_now_epoch)
  if [ "$bad" != 1 ]; then
    rm -f "$f"
    return 0
  fi
  [ -e "$f" ] || echo "$now" >"$f"
  since=$(cat "$f")
  if [ $((now - since)) -ge "$HEALTH_BAD_SECONDS" ]; then
    fc_log "ALERT $name has been bad for $((now - since)) s (GP fallback)"
  fi
}

health_disk_cpu() {
  local samples p99 steal bad=0 s0 s1
  samples=$(mktemp)
  fsync_samples "$FC_STATE_DIR" "${FSYNC_SAMPLE_COUNT:-20}" >"$samples"
  p99=$(percentile 99 "$samples")
  rm -f "$samples"
  awk -v v="$p99" -v l="$FSYNC_P99_LIMIT_MS" 'BEGIN { exit !(v > l) }' && bad=1
  _bad_since fsync "$bad"
  s0="$AGENT_RUN/stat-prev"
  s1=$(mktemp)
  cp "${FC_PROC_STAT:-/proc/stat}" "$s1"
  if [ -r "$s0" ]; then
    steal=$(steal_pct "$s0" "$s1")
    bad=0
    awk -v v="$steal" -v l="$STEAL_LIMIT_PCT" 'BEGIN { exit !(v > l) }' && bad=1
    _bad_since steal "$bad"
  fi
  mv -f "$s1" "$s0"
  fc_log "health: fsync p99 ${p99} ms"
}

health_sdn() {
  local peer
  for peer in $PEERS; do
    ping -c1 -W2 "$peer" >/dev/null 2>&1 || fc_log "SDN peer $peer does not answer ping"
  done
}

# Pings for the checks that pass; only URLs present in health.pings (Q5 open: none in v1).
health_pings() {
  local name url ok readyz
  desired_present || return 0
  while IFS=$'\t' read -r name url; do
    [ -n "$url" ] || continue
    ok=0
    case "$name" in
      agent) ok=1 ;;
      readyz)
        readyz=$(desired_jq -r --arg n "$NODE_NAME" '.health.readyz_urls[$n] // empty')
        [ -n "$readyz" ] && curl -sk --max-time 3 -o /dev/null -f "$readyz" && ok=1
        ;;
      litestream)
        lease_held && container_running "$LS_CONTAINER" && ok=1
        ;;
      lease)
        [ "$(headscale_status)" != stopped ] && ok=1
        ;;
      *) fc_log "unknown health ping $name: skipped" ;;
    esac
    if [ "$ok" = 1 ]; then
      curl -fsS --max-time 10 -o /dev/null "$url" || fc_log "health ping $name failed"
    fi
  done < <(desired_jq -r '.health.pings // {} | to_entries[] | [.key, .value] | @tsv')
}

HEALTH_COUNTER=0
health_step() {
  HEALTH_COUNTER=$((HEALTH_COUNTER + 1))
  step health-pings health_pings
  if [ $((HEALTH_COUNTER % 10)) -eq 1 ]; then
    step health-disk-cpu health_disk_cpu
    step health-sdn health_sdn
  fi
}
