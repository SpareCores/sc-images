#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The `reset` directive (PLAN 2.3, runbook 3.2) for a surviving CP: same interlock as prestart.
# The reset itself runs as a background job (it can take minutes and must not stall the lease).

# shellcheck source=/dev/null
. "${FC_SHARED_LIB}/interlock.sh"

reset_job() {
  host_exec systemctl stop k3s || return 1
  host_exec "$FC_K3S_BIN" server --cluster-reset || return 1
  host_exec systemctl start k3s
}

reset_step() {
  local state obj id refused
  state=$(bg_state reset)
  if [ "$state" = running ]; then
    return 0
  fi
  if [ "$state" = "done" ]; then
    id=$(cat "$AGENT_RUN/reset-id" 2>/dev/null || true)
    if [ -n "$id" ]; then
      if [ "$(bg_rc reset)" = 0 ]; then
        marker_put reset "$id" ok || return 1
        fc_log "reset $id done"
      else
        marker_put reset "$id" "refused:reset-failed" || true
        fc_log "ALERT reset $id failed; use a new id to retry"
      fi
      rm -f "$AGENT_RUN/reset-id"
    fi
    bg_clear reset
    return 0
  fi
  obj=$(directive_usable reset need-target) || return 0
  id=$(printf '%s' "$obj" | jq -r '.id')
  if interlock_ok "$NODE_NAME" "$INSTANCE_UUID"; then
    # consumed locally at once: a reset is never started twice
    local_consume "reset-${id}"
    printf '%s' "$id" >"$AGENT_RUN/reset-id"
    fc_log "reset $id: interlock passed, starting cluster-reset"
    bg_start reset reset_job
  else
    refused="$AGENT_RUN/refused-reset-${id}"
    if [ ! -e "$refused" ]; then
      : >"$refused"
      fc_log "reset $id refused by the interlock: $INTERLOCK_REASON"
      marker_put reset "$id" "refused:${INTERLOCK_REASON%%:*}" || true
    fi
  fi
}
