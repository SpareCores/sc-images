#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The lease holder's services: headscale and litestream as sibling containers, restore from the
# highest litestream generation, bootstrap (M7), duplicate cleanup, daily backup.

HS_CONTAINER=headscale
LS_CONTAINER=litestream

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]
}

containers_stop() {
  docker rm -f "$HS_CONTAINER" "$LS_CONTAINER" >/dev/null 2>&1 || true
  rm -f "$AGENT_RUN/started-epoch" "$AGENT_RUN/empty-start"
}

# Stops everything the holder runs (fence, release).
holder_stop() {
  containers_stop
  bg_kill lego
}

hs_cli() {
  headscale -c "$FC_HS_DIR/config.yaml" "$@"
}

# headscale: holder | standby | stopped (for the heartbeat)
headscale_status() {
  if ! desired_present; then
    echo stopped
    return 0
  fi
  if [ "$(desired_jq -r '.headscale.mode')" = stopped ]; then
    echo stopped
    return 0
  fi
  if lease_held; then
    if container_running "$HS_CONTAINER"; then echo holder; else echo stopped; fi
  else
    echo standby
  fi
}

write_litestream_config() {
  local file=$1 gen=$2 bucket endpoint sync
  bucket=$(desired_jq -r '.headscale.litestream.bucket')
  endpoint=$(desired_jq -r '.headscale.litestream.endpoint')
  sync=$(desired_jq -r '.headscale.litestream.sync_interval // "10s"')
  cat >"$file" <<CFG
dbs:
  - path: $HS_DB
    replicas:
      - type: s3
        bucket: $bucket
        path: gen-$gen
        endpoint: $endpoint
        region: auto
        sync-interval: $sync
CFG
}

# Credentials go through the environment of the docker CLI (-e NAME without a value).
_litestream_env() {
  LITESTREAM_ACCESS_KEY_ID=$(secrets_jq -r '.litestream.access_key_id') \
    LITESTREAM_SECRET_ACCESS_KEY=$(secrets_jq -r '.litestream.secret_access_key') \
    "$@"
}

litestream_image() { desired_jq -r '.images.litestream'; }
headscale_image() { desired_jq -r '.images.headscale'; }

start_containers() {
  local epoch=$1
  write_litestream_config "$FC_HS_DIR/litestream.yml" "$epoch"
  desired_jq -r '.headscale.config' >"$FC_HS_DIR/config.yaml"
  if ! container_running "$HS_CONTAINER"; then
    docker rm -f "$HS_CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$HS_CONTAINER" --network host -v "$FC_HS_DIR:$FC_HS_DIR" \
      "$(headscale_image)" serve -c "$FC_HS_DIR/config.yaml" >/dev/null || return 1
  fi
  if ! container_running "$LS_CONTAINER"; then
    docker rm -f "$LS_CONTAINER" >/dev/null 2>&1 || true
    _litestream_env docker run -d --name "$LS_CONTAINER" --network host -v "$FC_HS_DIR:$FC_HS_DIR" \
      -e LITESTREAM_ACCESS_KEY_ID -e LITESTREAM_SECRET_ACCESS_KEY \
      "$(litestream_image)" replicate -config "$FC_HS_DIR/litestream.yml" >/dev/null || return 1
  fi
}

# First start under a lease epoch: restore the newest replica, then serve and replicate to the
# own epoch's prefix. A local database is never trusted: a newer holder may have written since.
holder_start_full() {
  local epoch=$1 highest rc=0 empty=0 directive
  containers_stop
  rm -f "$HS_DB" "$HS_DB-wal" "$HS_DB-shm"
  highest=$(r2_highest_gen) || rc=$?
  if [ "$rc" -ne 0 ]; then
    fc_log "cannot list litestream generations: not starting headscale"
    return 1
  fi
  if [ -n "$highest" ] && [ "$highest" -ge "$epoch" ]; then
    fc_log "a replica gen-$highest is not older than our epoch $epoch: this holder is stale, not starting"
    return 1
  fi
  if [ -n "$highest" ]; then
    write_litestream_config "$FC_HS_DIR/litestream-restore.yml" "$highest"
    fc_log "restoring headscale from gen-$highest"
    _litestream_env docker run --rm --network host -v "$FC_HS_DIR:$FC_HS_DIR" \
      -e LITESTREAM_ACCESS_KEY_ID -e LITESTREAM_SECRET_ACCESS_KEY \
      "$(litestream_image)" restore -config "$FC_HS_DIR/litestream-restore.yml" -if-replica-exists "$HS_DB" >/dev/null || {
      fc_log "litestream restore failed: not starting headscale"
      return 1
    }
  fi
  if [ ! -s "$HS_DB" ]; then
    empty=1
    if ! directive=$(directive_usable headscale_bootstrap); then
      if [ ! -e "$AGENT_RUN/alerted-empty-$epoch" ]; then
        : >"$AGENT_RUN/alerted-empty-$epoch"
        fc_log "ALERT empty headscale replica and no headscale_bootstrap directive: headscale stays stopped"
      fi
      return 1
    fi
  fi
  start_containers "$epoch" || {
    fc_log "could not start the headscale containers"
    containers_stop
    return 1
  }
  echo "$epoch" >"$AGENT_RUN/started-epoch"
  if [ "$empty" = 1 ]; then
    printf '%s' "$directive" | jq -r '.id' >"$AGENT_RUN/empty-start"
  fi
  fc_log "headscale serving under epoch $epoch"
}

# Everything the holder does after a lease step. Renewal calls in between keep the lease alive
# during the slow parts.
holder_reconcile() {
  local epoch
  if ! lease_held; then
    # a non-holder runs neither headscale nor litestream
    if container_running "$HS_CONTAINER" || container_running "$LS_CONTAINER"; then
      fc_log "not the lease holder: stopping headscale and litestream"
      holder_stop
    fi
    return 0
  fi
  epoch=$(lease_field epoch)
  if ! cert_ensure; then
    fc_log "no usable <hs> certificate yet: headscale not started"
    return 0
  fi
  if [ "$(cat "$AGENT_RUN/started-epoch" 2>/dev/null)" != "$epoch" ]; then
    holder_start_full "$epoch" || return 0
    # the restore can take a while: renew right away
    lease_renew "$(fc_now_epoch)"
    lease_held || return 0
  else
    if [ "$CERT_CHANGED" = 1 ] && container_running "$HS_CONTAINER"; then
      docker restart "$HS_CONTAINER" >/dev/null 2>&1 || true
    fi
    if ! container_running "$HS_CONTAINER" || ! container_running "$LS_CONTAINER"; then
      fc_log "a headscale container is not running: starting it"
      start_containers "$epoch" || return 0
    fi
  fi
  return 0
}

# --- bootstrap (M7) ----------------------------------------------------------------------------

_ssm_put_new() {
  local name=$1 file=$2 err
  err=$(mktemp)
  if _aws --region "$AWS_REGION" ssm put-parameter --name "$name" --type SecureString --value "file://$file" >/dev/null 2>"$err"; then
    rm -f "$err"
    return 0
  fi
  if grep -q ParameterAlreadyExists "$err"; then
    fc_log "ALERT SSM parameter $name exists already (not overwritten)"
    rm -f "$err"
    return 0
  fi
  rm -f "$err"
  return 1
}

hs_bootstrap_step() {
  local id d uid key
  [ -r "$AGENT_RUN/empty-start" ] || return 0
  id=$(cat "$AGENT_RUN/empty-start")
  if marker_consumed "headscale-bootstrap-${id}" || ! directive_usable headscale_bootstrap >/dev/null; then
    rm -f "$AGENT_RUN/empty-start"
    return 0
  fi
  container_running "$HS_CONTAINER" || return 0
  hs_cli users list -o json >/dev/null 2>&1 || return 0 # not ready yet: next cycle
  d="$AGENT_RUN/hs-bootstrap"
  mkdir -p "$d"
  chmod 0700 "$d"
  uid=$(hs_cli users list -o json | jq -r '.[]? | select(.name == "infra") | .id' | head -n1)
  if [ -z "$uid" ]; then
    hs_cli users create infra >/dev/null || return 1
    uid=$(hs_cli users list -o json | jq -r '.[]? | select(.name == "infra") | .id' | head -n1)
  fi
  if [ ! -s "$d/api_key" ]; then
    key=$(hs_cli apikeys create --expiration 365d -o json | jq -r 'if type == "string" then . else (.key // empty) end') || return 1
    [ -n "$key" ] || return 1
    printf '%s' "$key" >"$d/api_key"
  fi
  if [ ! -s "$d/cp_key" ]; then
    key=$(hs_cli preauthkeys create --user "$uid" --reusable --tags "tag:${CLUSTER}-cp" --expiration 365d -o json | jq -r '.key // empty') || return 1
    [ -n "$key" ] || return 1
    printf '%s' "$key" >"$d/cp_key"
  fi
  # put-only: never --overwrite (M7)
  _ssm_put_new "/sc-infra/${CLUSTER}/headscale/bootstrap/api_key" "$d/api_key" || return 1
  _ssm_put_new "/sc-infra/${CLUSTER}/headscale/bootstrap/cp_key" "$d/cp_key" || return 1
  marker_put headscale-bootstrap "$id" ok || return 1
  rm -rf "$d" "$AGENT_RUN/empty-start"
  fc_log "headscale bootstrap $id done"
}

# --- duplicate CP nodes ------------------------------------------------------------------------

# Deletes offline nodes tagged tag:<name>-cp whose hostname is shared with a newer or an online
# node (a replaced CP re-registers under the same hostname).
hs_cleanup_duplicates() {
  local json tag victims id
  container_running "$HS_CONTAINER" || return 0
  tag="tag:${CLUSTER}-cp"
  json=$(hs_cli nodes list -o json 2>/dev/null) || return 0
  victims=$(printf '%s' "$json" | jq -r --arg tag "$tag" '
    [.[]? | select((((.tags // []) + (.forced_tags // [])) | index($tag)) != null)
          | {id, host: (.given_name // .name), online: (.online // false)}]
    | group_by(.host)[]
    | select(length > 1)
    | (map(select(.online)) | length) as $on
    | (max_by(.id) | .id) as $newest
    | .[] | select(.online | not)
    | select($on > 0 or .id != $newest)
    | .id')
  for id in $victims; do
    fc_log "deleting offline duplicate CP node $id"
    hs_cli nodes delete --identifier "$id" --force >/dev/null 2>&1 || true
  done
}

# --- daily backup ------------------------------------------------------------------------------

hs_daily_backup() {
  local day tmp rc=0
  day=$(date -u -d "@$(fc_now_epoch)" +%Y-%m-%d)
  [ -e "$FC_STATE_DIR/daily-$day" ] && return 0
  container_running "$HS_CONTAINER" || return 0
  [ -s "$HS_DB" ] || return 0
  tmp=$(mktemp -d)
  sqlite3 "$HS_DB" ".backup '$tmp/db.sqlite3'" || {
    rm -rf "$tmp"
    return 1
  }
  gzip -c "$tmp/db.sqlite3" >"$tmp/daily.gz"
  r2_put_new "daily/${day}.sqlite3.gz" "$tmp/daily.gz" || rc=$?
  rm -rf "$tmp"
  # 12: another holder uploaded today's copy already
  if [ "$rc" -eq 0 ] || [ "$rc" -eq "$S3_PRECONDITION" ]; then
    : >"$FC_STATE_DIR/daily-$day"
    return 0
  fi
  return 1
}

holder_extras() {
  lease_held || return 0
  step hs-bootstrap hs_bootstrap_step
  step hs-duplicates hs_cleanup_duplicates
  step daily-backup hs_daily_backup
}
