#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The CP's own tailscale login (WP-03 step 6), through the host's tailscale.

# Logs the host in when it is logged out and the secrets doc has a CP key.
tailscale_ensure() {
  local state key keyfile args
  secrets_present || return 0
  state=$(host_exec tailscale status --json 2>/dev/null | jq -r '.BackendState // empty') || return 0
  [ "$state" = NeedsLogin ] || return 0
  key=$(secrets_jq -r '.tailnet.cp_key // empty')
  if [ -z "$key" ]; then
    return 0 # before the headscale bootstrap there is no key
  fi
  keyfile="$FC_RUN_DIR/secrets/cp-key"
  printf '%s' "$key" >"$keyfile"
  # No --advertise-tags: the key carries tag:<cluster>-cp, and headscale (0.28+) rejects any requested tags
  # on a pre-auth key login. --reset clears tags a previous login may have left in the prefs.
  args=(--reset --login-server "$(desired_jq -r '.headscale.url')" "--auth-key=file:$keyfile"
    --accept-routes=false --accept-dns=false)
  if [ -n "$NODE_IP" ]; then
    args+=("--advertise-routes=${NODE_IP}/32")
  fi
  fc_log "tailscale is logged out: tailscale up"
  host_exec tailscale up "${args[@]}"
  local rc=$?
  rm -f "$keyfile"
  return "$rc"
}
