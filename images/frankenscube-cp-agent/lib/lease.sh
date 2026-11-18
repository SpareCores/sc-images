#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The headscale lease (PLAN 2.5): one S3 object, conditional writes, self-fence at 20 s.
#
# Fencing (the safety rule): the holder must stop headscale and litestream when its last successful
# renewal is 20 s old, so before anyone can take the lease (expires_at + 10 s skew = 40 s). Three
# things make that independent of how slow the rest of the agent is:
#   - the watchdog (cp-agent watchdog, its own process, every second) checks the age on a
#     monotonic clock (fc_mono_epoch) and kills the containers; the lease tick also checks it;
#   - every renewal attempt is bounded by timeout(20 - age), so none outlives the fence time;
#   - slow work (certificate, restore, container start) runs as a background job, never on the
#     lease path.
# A SIGKILL of the whole agent container cannot be caught here: the unit stops the sibling
# containers in ExecStopPost (docs/frankenscube/NODE-RUNTIME.md in sc-infra).
#
# Local state ($AGENT_RUN/lease-state.json, tmpfs): {epoch, etag, last_renew_mono}. It exists only
# while this agent believes it holds the lease. A reboot loses it on purpose: the new instance
# waits for the old lease to expire like anyone else.

LEASE_KEY=lease/headscale.json

lease_state_file() { printf '%s/lease-state.json' "$AGENT_RUN"; }
lease_held() { [ -r "$(lease_state_file)" ]; }
lease_field() { jq -r --arg f "$1" '.[$f] // empty' "$(lease_state_file)" 2>/dev/null; }

# The watchdog and the lease tick both change the state: serialise them.
with_lease_lock() {
  (
    flock 8
    "$@"
  ) 8>"$AGENT_RUN/lease.lock"
}

_lease_clear() { rm -f "$(lease_state_file)"; }
lease_clear() { with_lease_lock _lease_clear; }

_lease_state_write() {
  jq -n --argjson epoch "$1" --arg etag "$2" --argjson last "$3" \
    '{epoch: $epoch, etag: $etag, last_renew_mono: $last}' >"$(lease_state_file).tmp"
  mv -f "$(lease_state_file).tmp" "$(lease_state_file)"
}
# lease_state_write <epoch> <etag> <monotonic time the successful request started>
lease_state_write() { with_lease_lock _lease_state_write "$@"; }

# seconds since the last successful renewal (monotonic)
lease_age() {
  local last
  last=$(lease_field last_renew_mono)
  [ -n "$last" ] || return 1
  echo $(($(fc_mono_epoch) - last))
}

_watchdog_locked() {
  local age
  [ -r "$(lease_state_file)" ] || return 0
  age=$(lease_age) || return 0
  if [ "$age" -ge "$LEASE_FENCE_SECONDS" ]; then
    fc_log "WATCHDOG FENCE: last renewal ${age} s ago, stopping headscale and litestream"
    holder_stop
    _lease_clear
  fi
}

# The watchdog: one check; the loop (cp-agent watchdog-loop) runs it every second.
watchdog_check() {
  lease_held || return 0
  with_lease_lock _watchdog_locked
}

# lease_body <holder> <uuid> <epoch> <expires-epoch> > file
lease_body() {
  jq -n --arg holder "$1" --arg uuid "$2" --argjson epoch "$3" --arg exp "$(fc_epoch_to_iso "$4")" \
    '{holder: $holder, instance_uuid: $uuid, epoch: $epoch, expires_at: $exp}'
}

# --- eligibility -------------------------------------------------------------------------------

# Does the desired state want this node to serve? (mode, agent.disabled, pending replacement)
# Sets LEASE_WHY. Losing "wanted" releases the lease.
lease_wanted() {
  local mode doc_gen
  LEASE_WHY=""
  if ! desired_present; then
    LEASE_WHY="no-desired-doc"
    return 1
  fi
  if [ "$DISABLE_AGENT" = true ] || [ "$(desired_jq -r '.agent.disabled // false')" = true ]; then
    LEASE_WHY="agent-disabled"
    return 1
  fi
  mode=$(desired_jq -r '.headscale.mode')
  case "$mode" in
    auto) ;;
    "pinned:${NODE_NAME}") ;;
    stopped)
      LEASE_WHY="mode-stopped"
      return 1
      ;;
    *)
      LEASE_WHY="pinned-elsewhere"
      return 1
      ;;
  esac
  doc_gen=$(desired_jq -r --arg n "$NODE_NAME" '.nodes[$n].generation // empty')
  if [ -n "$doc_gen" ] && [ "$doc_gen" -gt "$GENERATION" ]; then
    LEASE_WHY="replacement-pending"
    return 1
  fi
  return 0
}

# generation delay: this node's generation must be at least 4 min old on this instance
generation_age_ok() {
  local f="$FC_STATE_DIR/generation-seen" now seen_gen since
  now=$(fc_now_epoch)
  seen_gen=""
  since=0
  if [ -r "$f" ]; then
    read -r seen_gen since <"$f" || true
  fi
  if [ "$seen_gen" != "$GENERATION" ] || [ -z "$since" ]; then
    printf '%s %s\n' "$GENERATION" "$now" >"$f"
    since=$now
  fi
  [ $((now - since)) -ge "$GENERATION_DELAY_SECONDS" ]
}

ntp_synced() {
  [ "$(host_exec timedatectl show -p NTPSynchronized --value 2>/dev/null)" = yes ]
}

# May this node take the lease now? (wanted + generation delay + NTP)
lease_may_acquire() {
  lease_wanted || return 1
  if ! generation_age_ok; then
    LEASE_WHY="generation-delay"
    return 1
  fi
  if ! ntp_synced; then
    LEASE_WHY="ntp-not-synced"
    return 1
  fi
  return 0
}

# --- the step ----------------------------------------------------------------------------------

lease_fence() {
  fc_log "FENCE: stopping headscale and litestream ($1)"
  holder_stop
  lease_clear
}

lease_release() {
  local tmp etag epoch rc=0
  etag=$(lease_field etag)
  epoch=$(lease_field epoch)
  tmp=$(mktemp)
  jq -n --argjson epoch "$epoch" '{holder: "", instance_uuid: "", epoch: $epoch, expires_at: "1970-01-01T00:00:00Z"}' >"$tmp"
  # stop serving first: the lease is only given up once nothing runs
  holder_stop
  s3_put "$LEASE_KEY" "$tmp" if-match "$etag" >/dev/null || rc=$?
  rm -f "$tmp"
  lease_clear
  fc_log "lease released ($LEASE_WHY, put rc=$rc)"
}

# lease_renew <wall now> <monotonic now>. The call is bounded by the time left until the fence.
lease_renew() {
  local now=$1 mono=$2 tmp etag epoch new rc=0 age remaining
  age=$(lease_age) || return 0
  remaining=$((LEASE_FENCE_SECONDS - age))
  if [ "$remaining" -lt 1 ]; then
    return 0 # the next watchdog check or tick fences
  fi
  if [ "$remaining" -gt 12 ]; then
    remaining=12
  fi
  etag=$(lease_field etag)
  epoch=$(lease_field epoch)
  tmp=$(mktemp)
  lease_body "$NODE_NAME" "$INSTANCE_UUID" "$epoch" $((now + LEASE_TTL_SECONDS)) >"$tmp"
  new=$(AWS_CALL_TIMEOUT=$remaining s3_put "$LEASE_KEY" "$tmp" if-match "$etag") || rc=$?
  rm -f "$tmp"
  case "$rc" in
    0) lease_state_write "$epoch" "$new" "$mono" ;;
    "$S3_PRECONDITION") lease_fence "lease taken over (412 on renew)" ;;
    *) fc_log "lease renew failed (rc=$rc), last renewal ${age} s ago" ;;
  esac
}

lease_acquire() {
  local now=$1 mono=$2 tmp cur etag rc=0 epoch exp new
  tmp=$(mktemp)
  cur=$(mktemp)
  etag=$(s3_get "$LEASE_KEY" "$cur") || rc=$?
  case "$rc" in
    0)
      epoch=$(jq -r '.epoch // empty' "$cur")
      exp=$(fc_iso_to_epoch "$(jq -r '.expires_at // empty' "$cur")") || exp=""
      if [ -z "$epoch" ] || [ -z "$exp" ]; then
        fc_log "lease object unparseable: not touching it"
        rm -f "$tmp" "$cur"
        return 0
      fi
      if [ "$now" -le $((exp + LEASE_SKEW_SECONDS)) ]; then
        rm -f "$tmp" "$cur"
        return 0
      fi
      epoch=$((epoch + 1))
      lease_body "$NODE_NAME" "$INSTANCE_UUID" "$epoch" $((now + LEASE_TTL_SECONDS)) >"$tmp"
      rc=0
      new=$(s3_put "$LEASE_KEY" "$tmp" if-match "$etag") || rc=$?
      ;;
    "$S3_NOTFOUND")
      epoch=1
      lease_body "$NODE_NAME" "$INSTANCE_UUID" "$epoch" $((now + LEASE_TTL_SECONDS)) >"$tmp"
      rc=0
      new=$(s3_put "$LEASE_KEY" "$tmp" if-none-match) || rc=$?
      ;;
    *)
      rm -f "$tmp" "$cur"
      return 0
      ;;
  esac
  rm -f "$tmp" "$cur"
  case "$rc" in
    0)
      lease_state_write "$epoch" "$new" "$mono"
      fc_log "lease acquired, epoch $epoch"
      ;;
    "$S3_PRECONDITION") fc_log "lost the race for the lease" ;;
    *) fc_log "lease acquire failed (rc=$rc)" ;;
  esac
}

# One lease tick (every 10 s).
lease_step() {
  local now mono age
  now=$(fc_now_epoch)
  mono=$(fc_mono_epoch)
  if lease_held; then
    age=$(lease_age) || age=$LEASE_FENCE_SECONDS
    if [ "$age" -ge "$LEASE_FENCE_SECONDS" ]; then
      lease_fence "last renewal ${age} s ago"
      return 0
    fi
    if ! lease_wanted; then
      lease_release
      return 0
    fi
    lease_renew "$now" "$mono"
    return 0
  fi
  if lease_may_acquire; then
    lease_acquire "$now" "$mono"
  fi
  return 0
}
