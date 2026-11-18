#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The headscale lease (PLAN 2.5): one S3 object, conditional writes, self-fence at 20 s.
#
# Local state ($AGENT_RUN/lease-state.json, tmpfs): {epoch, etag, last_renew}. It exists only
# while this agent believes it holds the lease. A reboot loses it on purpose: the new instance
# waits for the old lease to expire like anyone else.

LEASE_KEY=lease/headscale.json

lease_state_file() { printf '%s/lease-state.json' "$AGENT_RUN"; }
lease_held() { [ -r "$(lease_state_file)" ]; }
lease_field() { jq -r --arg f "$1" '.[$f] // empty' "$(lease_state_file)"; }
lease_clear() { rm -f "$(lease_state_file)"; }

lease_state_write() {
  jq -n --argjson epoch "$1" --arg etag "$2" --argjson last "$3" \
    '{epoch: $epoch, etag: $etag, last_renew: $last}' >"$(lease_state_file).tmp"
  mv -f "$(lease_state_file).tmp" "$(lease_state_file)"
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

lease_renew() {
  local now=$1 tmp etag epoch new rc=0
  etag=$(lease_field etag)
  epoch=$(lease_field epoch)
  tmp=$(mktemp)
  lease_body "$NODE_NAME" "$INSTANCE_UUID" "$epoch" $((now + LEASE_TTL_SECONDS)) >"$tmp"
  new=$(s3_put "$LEASE_KEY" "$tmp" if-match "$etag") || rc=$?
  rm -f "$tmp"
  case "$rc" in
    0) lease_state_write "$epoch" "$new" "$now" ;;
    "$S3_PRECONDITION") lease_fence "lease taken over (412 on renew)" ;;
    *) fc_log "lease renew failed (rc=$rc), last renewal $((now - $(lease_field last_renew))) s ago" ;;
  esac
}

lease_acquire() {
  local now=$1 tmp cur etag rc=0 epoch exp new
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
      lease_state_write "$epoch" "$new" "$now"
      fc_log "lease acquired, epoch $epoch"
      ;;
    "$S3_PRECONDITION") fc_log "lost the race for the lease" ;;
    *) fc_log "lease acquire failed (rc=$rc)" ;;
  esac
}

# One lease tick (every 10 s).
lease_step() {
  local now age
  now=$(fc_now_epoch)
  if lease_held; then
    age=$((now - $(lease_field last_renew)))
    if [ "$age" -ge "$LEASE_FENCE_SECONDS" ]; then
      lease_fence "last renewal ${age} s ago"
      return 0
    fi
    if ! lease_wanted; then
      lease_release
      return 0
    fi
    lease_renew "$now"
    return 0
  fi
  if lease_may_acquire; then
    lease_acquire "$now"
  fi
  return 0
}
