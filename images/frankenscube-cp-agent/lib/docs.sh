#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# Desired docs, markers and heartbeats (CONTRACTS sections 6 and 7).

DESIRED_CACHE() { printf '%s/desired.json' "$FC_RUN_DIR"; }
SECRETS_CACHE() { printf '%s/secrets/desired-secrets.json' "$FC_RUN_DIR"; }

# The required fields of desired.json. Nothing is acted on that does not pass this.
# shellcheck disable=SC2016
DESIRED_VALID_JQ='
def directive($withtarget):
  . == null or (type == "object"
    and (.id | type == "string" and test("^[a-z0-9-]{1,40}$"))
    and (.not_after | type == "string" and test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z$"))
    and (if $withtarget then (.target | type == "string" and test("^cp[0-9]+$")) else true end));
.version == 1
and .cluster == $cluster and .stack == $stack
and (.nodes | type == "object")
and (.peers | type == "array")
and (.images | type == "object")
and (.headscale | type == "object")
and (.headscale.mode | type == "string" and test("^(auto|stopped|pinned:cp[0-9]+)$"))
and (.headscale.url | type == "string")
and (.headscale.config | type == "string")
and (.headscale.litestream | type == "object")
and (.headscale.litestream.bucket | type == "string")
and (.headscale.litestream.endpoint | type == "string")
and (.headscale.derp_map == null or (.headscale.derp_map | type == "string"))
and (.headscale.acme | type == "object")
and (.headscale.acme.zone_id | type == "string")
and (.headscale.acme.email | type == "string")
and (.directives | type == "object")
and (.directives.bootstrap | directive(true))
and (.directives.restore | directive(true))
and (.directives.reset | directive(true))
and (.directives.headscale_bootstrap | directive(false))
and (.health | type == "object")
and (.agent | type == "object")
'

# shellcheck disable=SC2016
SECRETS_VALID_JQ='
.version == 1
and (.tailnet | type == "object")
and (.litestream | type == "object")
and (.litestream.access_key_id | type == "string")
and (.litestream.secret_access_key | type == "string")
and (.restore | type == "object")
and (.restore.bucket | type == "string")
and (.restore.endpoint | type == "string")
and (.restore.access_key_id | type == "string")
and (.restore.secret_access_key | type == "string")
'

desired_jq() { jq "$@" "$(DESIRED_CACHE)"; }
secrets_jq() { jq "$@" "$(SECRETS_CACHE)"; }
desired_present() { [ -r "$(DESIRED_CACHE)" ]; }
secrets_present() { [ -r "$(SECRETS_CACHE)" ]; }

# Fetch one doc, validate, replace the cache atomically. An unreadable or invalid doc keeps the
# last valid one.
_doc_sync() {
  local key=$1 cache=$2 check=$3 tmp rc=0
  tmp=$(mktemp)
  s3_get "$key" "$tmp" >/dev/null || rc=$?
  if [ "$rc" -ne 0 ]; then
    fc_log "cannot fetch $key (rc=$rc): keeping the last valid doc"
    rm -f "$tmp"
    return 1
  fi
  if ! jq -e --arg cluster "$CLUSTER" --arg stack "${STACK:-$CLUSTER}" "$check" "$tmp" >/dev/null 2>&1; then
    fc_log "INVALID $key: ignored, keeping the last valid doc"
    rm -f "$tmp"
    return 1
  fi
  chmod 0600 "$tmp"
  mv -f "$tmp" "$cache"
}

docs_sync() {
  local rc=0
  _doc_sync desired.json "$(DESIRED_CACHE)" "$DESIRED_VALID_JQ" || rc=1
  _doc_sync desired-secrets.json "$(SECRETS_CACHE)" "$SECRETS_VALID_JQ" || rc=1
  if secrets_present; then
    write_restore_env
  fi
  return "$rc"
}

# restore.env for prestart (0600). ETCD_S3_LATEST_SNAPSHOT is the newest snapshot name, resolved from
# the restore bucket while a restore directive exists; a failed resolution keeps the previous value.
write_restore_env() {
  local f="$FC_RUN_DIR/secrets/restore.env" latest=""
  if desired_present && [ "$(desired_jq -r '.directives.restore // empty | .id // empty')" != "" ]; then
    latest=$(restore_latest_snapshot) || latest=""
  fi
  if [ -z "$latest" ] && [ -r "$f" ]; then
    latest=$(sed -n 's/^ETCD_S3_LATEST_SNAPSHOT=//p' "$f")
  fi
  (
    umask 077
    secrets_jq -r '.restore | "ETCD_S3_ENDPOINT=\(.endpoint)\nETCD_S3_BUCKET=\(.bucket)\nETCD_S3_REGION=\(.region // "auto")\nETCD_S3_ACCESS_KEY_ID=\(.access_key_id)\nETCD_S3_SECRET_ACCESS_KEY=\(.secret_access_key)\nETCD_S3_FOLDER=snapshots"' >"$f.tmp"
    if [ -n "$latest" ]; then
      printf 'ETCD_S3_LATEST_SNAPSHOT=%s\n' "$latest" >>"$f.tmp"
    fi
    mv -f "$f.tmp" "$f"
  )
}

# --- markers -----------------------------------------------------------------------------------

MARKER_CACHE() { printf '%s/markers-cache' "$AGENT_RUN"; }

# Lists markers/, fetches the unknown ones, keeps those with result "ok" and writes markers.json.
# A marker with result "refused:*" is a report, not a consumption.
markers_sync() {
  local keys key name tmp
  mkdir -p "$(MARKER_CACHE)"
  keys=$(s3_list markers/) || {
    fc_log "cannot list markers/"
    return 1
  }
  tmp=$(mktemp)
  for key in $keys; do
    name=${key#markers/}
    name=${name%.json}
    [ -n "$name" ] || continue
    if [ -e "$(MARKER_CACHE)/$name" ]; then
      continue
    fi
    if s3_get "$key" "$tmp" >/dev/null && [ "$(jq -r '.result // empty' "$tmp" 2>/dev/null)" = ok ]; then
      : >"$(MARKER_CACHE)/$name"
    fi
  done
  rm -f "$tmp"
  # shellcheck disable=SC2046
  jq -n '$ARGS.positional | {markers: (. | sort)}' --args $(ls "$(MARKER_CACHE)" 2>/dev/null) >"$FC_RUN_DIR/markers.json.tmp"
  mv -f "$FC_RUN_DIR/markers.json.tmp" "$FC_RUN_DIR/markers.json"
}

# Is `<kind>-<id>` consumed? (an ok marker in S3, or consumed locally by this node)
marker_consumed() {
  local name=$1
  [ -e "$FC_STATE_DIR/consumed/$name" ] && return 0
  [ -e "$(MARKER_CACHE)/$name" ] && return 0
  return 1
}

local_consume() {
  mkdir -p "$FC_STATE_DIR/consumed"
  : >"$FC_STATE_DIR/consumed/$1"
}

# marker_put <kind> <id> <result>: an ok marker overwrites a refusal; a refusal never overwrites.
marker_put() {
  local kind=$1 id=$2 result=$3 tmp rc=0 key
  key="markers/${kind}-${id}.json"
  tmp=$(mktemp)
  jq -n --arg node "$NODE_NAME" --arg uuid "$INSTANCE_UUID" --arg ts "$(fc_now_iso)" --arg r "$result" \
    '{node: $node, instance_uuid: $uuid, ts: $ts, result: $r}' >"$tmp"
  if [ "$result" = ok ]; then
    s3_put "$key" "$tmp" >/dev/null || rc=$?
    if [ "$rc" -eq 0 ]; then
      : >"$(MARKER_CACHE)/${kind}-${id}"
    fi
  else
    s3_put "$key" "$tmp" if-none-match >/dev/null || rc=$?
    # 12: a marker exists already (ok or refusal): nothing to add
    [ "$rc" -eq "$S3_PRECONDITION" ] && rc=0
  fi
  rm -f "$tmp"
  return "$rc"
}

# Uploads the marker requests of prestart and deletes them once uploaded.
marker_requests_upload() {
  local f name body kind id result rc=0
  for f in "$FC_RUN_DIR"/marker-requests/*.json; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .json)
    body=$(cat "$f")
    result=$(printf '%s' "$body" | jq -r '.result // empty')
    # ids may contain dashes: the kind is the known prefix
    case "$name" in
      headscale-bootstrap-*) kind=headscale-bootstrap; id=${name#headscale-bootstrap-} ;;
      bootstrap-*) kind=bootstrap; id=${name#bootstrap-} ;;
      restore-*) kind=restore; id=${name#restore-} ;;
      reset-*) kind=reset; id=${name#reset-} ;;
      *) fc_log "unknown marker request $name"; rm -f "$f"; continue ;;
    esac
    if marker_put "$kind" "$id" "$result"; then
      rm -f "$f"
    else
      rc=1
      fc_log "marker $name not uploaded yet"
    fi
  done
  return "$rc"
}

# --- directives --------------------------------------------------------------------------------

# directive_usable <kind> [need-target]: prints the directive JSON when it is valid, not expired
# and not consumed (kind: bootstrap, restore, reset, headscale_bootstrap). With need-target the
# directive must target this node.
directive_usable() {
  local kind=$1 target=${2:-} obj id na now marker_kind
  desired_present || return 1
  obj=$(desired_jq -c --arg k "$kind" '.directives[$k] // empty') || return 1
  [ -n "$obj" ] || return 1
  if [ -n "$target" ] && [ "$(printf '%s' "$obj" | jq -r '.target // empty')" != "$NODE_NAME" ]; then
    return 1
  fi
  id=$(printf '%s' "$obj" | jq -r '.id')
  na=$(fc_iso_to_epoch "$(printf '%s' "$obj" | jq -r '.not_after')") || return 1
  now=$(fc_now_epoch)
  [ -n "$na" ] && [ "$now" -lt "$na" ] || return 1
  marker_kind=${kind//_/-}
  if marker_consumed "${marker_kind}-${id}"; then
    return 1
  fi
  printf '%s' "$obj"
}

# --- heartbeats --------------------------------------------------------------------------------

heartbeats_fetch() {
  local node tmp out hb=0 rc
  desired_present || return 1
  tmp=$(mktemp)
  out=$(jq -n --arg f "$(fc_now_iso)" '{fetched_at: $f}')
  for node in $(desired_jq -r '.nodes | keys[]'); do
    rc=0
    S3_LAST_MODIFIED=""
    s3_get "heartbeats/${node}.json" "$tmp" >/dev/null || rc=$?
    if [ "$rc" -eq "$S3_NOTFOUND" ]; then
      continue
    fi
    if [ "$rc" -ne 0 ]; then
      fc_log "cannot fetch the heartbeat of $node: heartbeats.json not refreshed"
      rm -f "$tmp"
      return 1
    fi
    if jq -e '.version == 1 and .etcd and .instance_uuid and .ts' "$tmp" >/dev/null 2>&1; then
      # last_modified is S3's clock: the interlock ages heartbeats with it, not with the writer's ts
      out=$(printf '%s' "$out" | jq --arg n "$node" --arg lm "$S3_LAST_MODIFIED" --slurpfile h "$tmp" \
        '.[$n] = ($h[0] + (if $lm == "" then {} else {last_modified: $lm} end))')
      hb=$((hb + 1))
    fi
  done
  rm -f "$tmp"
  printf '%s\n' "$out" >"$FC_RUN_DIR/heartbeats.json.tmp"
  mv -f "$FC_RUN_DIR/heartbeats.json.tmp" "$FC_RUN_DIR/heartbeats.json"
}

local_etcd_state() {
  if [ -d "$FC_K3S_DIR/server/db" ]; then echo member; else echo none; fi
}

local_phase() {
  local p="waiting-node-ip"
  if [ -r "$FC_RUN_DIR/phase" ]; then
    p=$(head -n1 "$FC_RUN_DIR/phase")
  fi
  case "$p" in
    waiting-node-ip | waiting-peer | running | stopped) printf '%s' "$p" ;;
    *) printf 'waiting-node-ip' ;;
  esac
}

heartbeat_put() {
  local tmp hs
  tmp=$(mktemp)
  hs=$(headscale_status)
  jq -n --arg node "$NODE_NAME" --arg uuid "$INSTANCE_UUID" --argjson gen "$GENERATION" \
    --arg ts "$(fc_now_iso)" --arg etcd "$(local_etcd_state)" --arg phase "$(local_phase)" --arg hs "$hs" \
    '{version: 1, node: $node, instance_uuid: $uuid, generation: $gen, ts: $ts, etcd: $etcd, phase: $phase, headscale: $hs}' >"$tmp"
  s3_put "heartbeats/${NODE_NAME}.json" "$tmp" >/dev/null
  local rc=$?
  rm -f "$tmp"
  return "$rc"
}
