#!/usr/bin/env bats
# cp-agent (PLAN WP-03): lease, fencing, holder services, certificate, directives, docs.
# S3, docker, the host (nsenter) and headscale are fakes; the clock is a file.

load helpers

setup() { setup_env; }
teardown() { teardown_env; }

# a fully working holder setup: valid certificate stored, headscale started on an empty replica
holder_ready() {
  store_cert "$TMP/cert-valid"
  bootstrap_hs_directive hb1
  age_generation cp1
  age_generation cp2
  age_generation cp3
}

# --- lease -------------------------------------------------------------------------------------

@test "lease: two agents race for an absent lease and exactly one wins" {
  local i
  for i in 1 2 3 4 5; do
    rm -rf "${FAKE_S3:?}/$CTRL/frankenscube/lease"
    rm -f "$TMP/cp1/run/agent/lease-state.json" "$TMP/cp2/run/agent/lease-state.json"
    age_generation cp1
    age_generation cp2
    agent cp1 lease &
    p1=$!
    agent cp2 lease &
    p2=$!
    wait "$p1" "$p2"
    holders=0
    [ -e "$TMP/cp1/run/agent/lease-state.json" ] && holders=$((holders + 1))
    [ -e "$TMP/cp2/run/agent/lease-state.json" ] && holders=$((holders + 1))
    [ "$holders" -eq 1 ]
    [ "$(lease_epoch)" -eq 1 ]
    winner=$(lease_holder)
    [ -e "$TMP/$winner/run/agent/lease-state.json" ]
  done
}

@test "lease: renewal keeps it, the standby never takes it while it is renewed" {
  age_generation cp1
  age_generation cp2
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
  for i in 1 2 3 4 5 6 7 8; do
    advance 10
    agent cp1 lease
    agent cp2 lease
    [ "$(lease_holder)" = cp1 ]
    [ "$(lease_epoch)" -eq 1 ]
  done
  [ "$(lease | jq -r .expires_at)" = "$(iso $(($(now) + 30)))" ]
  [ ! -e "$TMP/cp2/run/agent/lease-state.json" ]
}

@test "lease: a holder that cannot reach S3 stops headscale within 20 s, before anyone can take over" {
  holder_ready
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
  hs_running cp1
  ls_running cp1
  put_replica 1 # what litestream of the epoch-1 holder wrote
  local t0
  t0=$(now)
  touch "$FAKE_S3/.down"
  advance 10
  agent cp1 lease
  hs_running cp1 # 10 s since the last renewal: still serving
  advance 10
  agent cp1 lease # 20 s: self-fence, without any S3 access
  not hs_running cp1
  not ls_running cp1
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
  # S3 is back at t0+20; cp2 must not take over before expires_at (t0+30) + 10 s skew
  rm "$FAKE_S3/.down"
  [ "$(now)" -eq $((t0 + 20)) ]
  agent cp2 lease
  [ "$(lease_holder)" = cp1 ]
  advance 10
  agent cp2 lease
  [ "$(lease_holder)" = cp1 ]
  advance 10 # t0+40: not yet (needs now > t0+40)
  agent cp2 lease
  [ "$(lease_holder)" = cp1 ]
  advance 1
  agent cp2 lease
  [ "$(lease_holder)" = cp2 ]
  [ "$(lease_epoch)" -eq 2 ]
}

@test "lease: takeover needs now > expires_at + 10 s skew (not at +10, yes at +11)" {
  age_generation cp2
  jq -n --arg e "$(iso $((START_NOW - 10)))" '{holder: "cp1", instance_uuid: "u1", epoch: 3, expires_at: $e}' | s3_write lease/headscale.json
  agent cp2 lease
  [ "$(lease_holder)" = cp1 ]
  advance 1
  agent cp2 lease
  [ "$(lease_holder)" = cp2 ]
  [ "$(lease_epoch)" -eq 4 ]
}

@test "lease: an expired lease is taken with epoch + 1 and headscale restored from the previous epoch" {
  holder_ready
  put_replica 4
  jq -n --arg e "$(iso $((START_NOW - 100)))" '{holder: "cp3", instance_uuid: "u3", epoch: 4, expires_at: $e}' | s3_write lease/headscale.json
  agent cp2 lease
  [ "$(lease_holder)" = cp2 ]
  [ "$(lease_epoch)" -eq 5 ]
  hs_running cp2
  ls_running cp2
  grep -q 'path: gen-4' "$(fd cp2)"/ls-cfg/*-restore
  grep -q 'path: gen-5' "$(fd cp2)"/ls-cfg/*-replicate
  not grep -q 'path: gen-5' "$(fd cp2)"/ls-cfg/*-restore
  [ "$(cat "$TMP/cp2/hs/db.sqlite")" = "replica of gen-4" ]
}

@test "stale holder: never writes a newer prefix and stops when the lease was taken (412)" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  put_replica 1
  # someone else takes the lease behind its back (as after a long pause)
  jq -n --arg e "$(iso $(($(now) + 30)))" '{holder: "cp2", instance_uuid: "u2", epoch: 2, expires_at: $e}' | s3_write lease/headscale.json
  advance 10
  agent cp1 lease
  not hs_running cp1
  not ls_running cp1
  # every litestream config cp1 ever used points at gen-1
  [ -z "$(grep -h 'path: gen-' "$(fd cp1)"/ls-cfg/* | grep -v 'gen-1$' || true)" ]
}

@test "stale holder: refuses to serve when a replica generation is not older than its epoch" {
  holder_ready
  put_replica 3
  jq -n --arg e "$(iso $((START_NOW - 100)))" '{holder: "cp3", instance_uuid: "u3", epoch: 2, expires_at: $e}' | s3_write lease/headscale.json
  agent cp1 lease
  [ "$(lease_epoch)" -eq 3 ]
  not hs_running cp1
}

@test "lease: pinned to another node means no acquisition, pinned to self does" {
  age_generation cp1
  age_generation cp2
  put_desired '.headscale.mode = "pinned:cp2"'
  agent cp1 lease
  not s3_has lease/headscale.json
  agent cp2 lease
  [ "$(lease_holder)" = cp2 ]
}

@test "lease: changing the pin releases the lease and stops headscale; the new pin takes over" {
  holder_ready
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
  hs_running cp1
  put_replica 1
  put_desired '.headscale.mode = "pinned:cp2"'
  advance 10
  agent cp1 full
  not hs_running cp1
  [ "$(lease | jq -r .holder)" = "" ]
  agent cp2 lease
  [ "$(lease_holder)" = cp2 ]
  [ "$(lease_epoch)" -eq 2 ]
  hs_running cp2
}

@test "lease: mode stopped releases the lease and nobody takes it" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  put_desired '.headscale.mode = "stopped"'
  advance 10
  agent cp1 full
  not hs_running cp1
  [ "$(lease | jq -r .holder)" = "" ]
  advance 100
  agent cp2 lease
  agent cp3 lease
  [ "$(lease | jq -r .holder)" = "" ]
  agent cp2 full
  [ "$(s3_cat heartbeats/cp2.json | jq -r .headscale)" = stopped ]
}

@test "lease: generation delay of 4 minutes after the generation is first seen" {
  # no age_generation: this instance has just started
  agent cp1 lease
  not s3_has lease/headscale.json
  advance 239
  agent cp1 lease
  not s3_has lease/headscale.json
  advance 1
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
}

@test "lease: a node whose generation in the doc is newer than its own never holds (replacement pending)" {
  age_generation cp1
  put_desired '.nodes.cp1.generation = 2'
  agent cp1 lease
  not s3_has lease/headscale.json
}

@test "lease: no acquisition without NTP sync" {
  age_generation cp1
  touch "$(fd cp1)/ntp-off"
  agent cp1 lease
  not s3_has lease/headscale.json
  rm "$(fd cp1)/ntp-off"
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
}

@test "lease: a restarted instance does not reuse its old lease before it expires" {
  age_generation cp1
  agent cp1 lease
  rm "$TMP/cp1/run/agent/lease-state.json" # tmpfs lost across a reboot
  advance 10
  agent cp1 lease
  [ "$(lease_epoch)" -eq 1 ]
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
  advance 31
  agent cp1 lease
  [ "$(lease_epoch)" -eq 2 ]
}

@test "non-holder: headscale and litestream are stopped" {
  holder_ready
  mkdir -p "$(fd cp3)/containers"
  echo x >"$(fd cp3)/containers/headscale"
  echo x >"$(fd cp3)/containers/litestream"
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
  agent cp3 lease
  not hs_running cp3
  not ls_running cp3
}

# --- empty replica, bootstrap (M7) -------------------------------------------------------------

@test "empty replica without a directive: the holder keeps headscale stopped and alerts" {
  store_cert "$TMP/cert-valid"
  age_generation cp1
  agent cp1 full
  [ "$(lease_holder)" = cp1 ]
  not hs_running cp1
  not ls_running cp1
  grep -q 'ALERT empty headscale replica' "$TMP/cp1.log"
  [ "$(s3_cat heartbeats/cp1.json | jq -r .headscale)" = stopped ]
}

@test "headscale bootstrap: user, API key, CP key to put-only SSM parameters, marker, once" {
  holder_ready
  agent cp1 full
  hs_running cp1
  agent cp1 full
  grep -q 'users create infra' "$(fd cp1)/headscale.log"
  grep -q 'apikeys create --expiration 365d' "$(fd cp1)/headscale.log"
  grep -q 'preauthkeys create --user 7 --reusable --tags tag:frankenscube-cp --expiration 365d' "$(fd cp1)/headscale.log"
  [ "$(cat "$FAKE_S3/.ssm/_sc-infra_frankenscube_headscale_bootstrap_api_key")" = hskey-api-1 ]
  [ "$(cat "$FAKE_S3/.ssm/_sc-infra_frankenscube_headscale_bootstrap_cp_key")" = hskey-cp-1 ]
  s3_has markers/headscale-bootstrap-hb1.json
  jq -e '.result == "ok"' "$(s3_path markers/headscale-bootstrap-hb1.json)"
  : >"$(fd cp1)/headscale.log"
  settle cp1
  not grep -q 'users create' "$(fd cp1)/headscale.log"
}

@test "headscale bootstrap: waits while the CLI is not ready, mints keys once" {
  holder_ready
  mkdir -p "$(fd cp1)/hs"
  touch "$(fd cp1)/hs/notready"
  agent cp1 full
  not s3_has markers/headscale-bootstrap-hb1.json
  rm "$(fd cp1)/hs/notready"
  settle cp1
  s3_has markers/headscale-bootstrap-hb1.json
  [ "$(grep -c 'apikeys create' "$(fd cp1)/headscale.log")" -eq 1 ]
}

@test "headscale bootstrap: an expired directive is not acted on" {
  store_cert "$TMP/cert-valid"
  age_generation cp1
  put_desired ".directives.headscale_bootstrap = {id: \"old\", not_after: \"$(iso $((START_NOW - 5)))\"}"
  agent cp1 full
  not hs_running cp1
}

@test "headscale bootstrap: a consumed directive (marker present) is not acted on" {
  store_cert "$TMP/cert-valid"
  age_generation cp1
  bootstrap_hs_directive hb1
  echo '{"node":"cp2","instance_uuid":"x","ts":"2027-01-01T00:00:00Z","result":"ok"}' | s3_write markers/headscale-bootstrap-hb1.json
  agent cp1 full
  not hs_running cp1
}

# --- certificate -------------------------------------------------------------------------------

@test "certificate: a valid stored certificate is reused, no ACME call" {
  holder_ready
  agent cp1 full
  hs_running cp1
  [ ! -e "$(fd cp1)/lego.log" ]
  cmp "$TMP/cp1/hs/tls/fullchain.pem" "$TMP/cert-valid.crt"
}

@test "certificate: issued once with lego when none is stored, then reused by the next holder" {
  bootstrap_hs_directive hb1
  age_generation cp1
  age_generation cp2
  agent cp1 full
  not hs_running cp1 # waiting for the issuance
  wait_bg cp1 lego
  settle cp1
  hs_running cp1
  s3_has certs/hs/fullchain.pem
  s3_has certs/hs/privkey.pem
  [ "$(grep -c '^lego' "$(fd cp1)/lego.log")" -eq 1 ]
  grep -q -- '--dns route53' "$(fd cp1)/lego.log"
  grep -q -- '--domains frankenscube.hs.sparecores.net' "$(fd cp1)/lego.log"
  grep -q 'zone=Z123' "$(fd cp1)/lego.log"
  # cp1 goes away; cp2 takes over with the stored certificate
  put_replica 1
  advance 60
  agent cp2 full
  hs_running cp2
  [ ! -e "$(fd cp2)/lego.log" ]
}

@test "certificate: a stored certificate with under 30 days left is served and renewed" {
  make_cert 10 "$TMP/cert-short"
  store_cert "$TMP/cert-short"
  bootstrap_hs_directive hb1
  age_generation cp1
  agent cp1 full
  hs_running cp1 # serving with the stop-gap certificate
  wait_bg cp1 lego
  settle cp1
  [ "$(grep -c '^lego' "$(fd cp1)/lego.log")" -eq 1 ]
  cmp "$TMP/cp1/hs/tls/fullchain.pem" "$(s3_path certs/hs/fullchain.pem)"
  not cmp -s "$TMP/cp1/hs/tls/fullchain.pem" "$TMP/cert-short.crt"
  [ -s "$(fd cp1)/restarts.log" ]
}

@test "certificate: S3 unreadable means no blind issuance" {
  bootstrap_hs_directive hb1
  age_generation cp1
  touch "$FAKE_S3/.down-certs" # the lease works, reading certs/hs/ fails
  agent cp1 lease
  [ "$(lease_holder)" = cp1 ]
  settle cp1
  [ ! -e "$(fd cp1)/lego.log" ]
  not hs_running cp1
}

# --- reset directive ---------------------------------------------------------------------------

reset_directive() {
  put_desired ".directives.reset = {id: \"$1\", target: \"cp1\", not_after: \"$(iso $((START_NOW + 7200)))\"}"
}

@test "reset: runs once the interlock passes (3 min of failed probes) and is consumed once" {
  reset_directive rs1
  age_generation cp1
  local i
  for i in $(seq 1 9); do
    settle cp1
  done
  [ "$(grep -c 'k3s server --cluster-reset' "$(fd cp1)/host.log")" -eq 1 ]
  grep -q 'systemctl stop k3s' "$(fd cp1)/host.log"
  grep -q 'systemctl start k3s' "$(fd cp1)/host.log"
  jq -e '.result == "ok"' "$(s3_path markers/reset-rs1.json)"
  for i in 1 2 3 4; do
    settle cp1
  done
  [ "$(grep -c 'k3s server --cluster-reset' "$(fd cp1)/host.log")" -eq 1 ]
  [ "$(jq -r '.markers | index("reset-rs1") != null' "$TMP/cp1/run/markers.json")" = true ]
}

@test "reset: refused while a peer answers /readyz, with one refusal marker, nothing runs" {
  reset_directive rs2
  age_generation cp1
  echo 200 >"$(fd cp1)/peer-10.0.0.2"
  local i
  for i in $(seq 1 10); do
    settle cp1
  done
  [ ! -e "$(fd cp1)/host.log" ] || ! grep -q 'cluster-reset' "$(fd cp1)/host.log"
  jq -e '.result == "refused:peer-answers"' "$(s3_path markers/reset-rs2.json)"
  [ "$(grep -c 'put-object .*key=frankenscube/markers/reset-rs2' "$FAKE_S3/.calls")" -eq 1 ]
}

@test "reset: refused on a fresh member heartbeat of another instance" {
  reset_directive rs3
  age_generation cp1
  jq -n --arg ts "$(iso "$START_NOW")" '{version: 1, node: "cp2", instance_uuid: "other", generation: 1, ts: $ts, etcd: "member", phase: "running", headscale: "standby"}' | s3_write heartbeats/cp2.json
  agent cp1 full
  settle cp1
  [ ! -e "$(fd cp1)/host.log" ] || ! grep -q 'cluster-reset' "$(fd cp1)/host.log"
  jq -e '.result == "refused:member-heartbeat"' "$(s3_path markers/reset-rs3.json)"
}

@test "reset: targeted elsewhere or expired is ignored" {
  age_generation cp1
  put_desired ".directives.reset = {id: \"x1\", target: \"cp2\", not_after: \"$(iso $((START_NOW + 7200)))\"}"
  for i in $(seq 1 8); do
    settle cp1
  done
  put_desired ".directives.reset = {id: \"x2\", target: \"cp1\", not_after: \"$(iso $((START_NOW - 5)))\"}"
  for i in $(seq 1 8); do
    settle cp1
  done
  [ ! -e "$(fd cp1)/host.log" ] || ! grep -q 'cluster-reset' "$(fd cp1)/host.log"
}

# --- docs, markers, heartbeats -----------------------------------------------------------------

@test "docs: both valid docs are cached; restore.env is written with mode 0600" {
  agent cp1 full
  jq -e '.version == 1 and .headscale.mode == "auto"' "$TMP/cp1/run/desired.json"
  jq -e '.litestream.access_key_id == "LSAK"' "$TMP/cp1/run/secrets/desired-secrets.json"
  [ "$(stat -c %a "$TMP/cp1/run/secrets/restore.env")" = 600 ]
  grep -q '^ETCD_S3_BUCKET=sc-frankenscube-etcd$' "$TMP/cp1/run/secrets/restore.env"
  grep -q '^ETCD_S3_ACCESS_KEY_ID=RAK$' "$TMP/cp1/run/secrets/restore.env"
  grep -q '^ETCD_S3_ENDPOINT=https://acct.eu.r2.cloudflarestorage.com$' "$TMP/cp1/run/secrets/restore.env"
}

@test "docs: an invalid doc is ignored, the last valid one is kept and never acted on" {
  age_generation cp1
  agent cp1 full
  cp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  # missing headscale.config: invalid
  put_desired 'del(.headscale.config) | .headscale.mode = "stopped"'
  advance 10
  agent cp1 full
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  # wrong cluster: invalid
  put_desired '.cluster = "other" | .headscale.mode = "stopped"'
  agent cp1 full
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  # not JSON
  echo '{"version": 1' | s3_write desired.json
  agent cp1 full
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  grep -q 'INVALID desired.json' "$TMP/cp1.log"
  # invalid directive id inside an otherwise complete doc
  put_desired '.directives.reset = {id: "Bad_ID", target: "cp1", not_after: "2099-01-01T00:00:00Z"}'
  agent cp1 full
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  # the old doc (mode auto) still drives the lease
  [ "$(lease_holder)" = cp1 ]
}

@test "docs: S3 down keeps the last valid doc and the state (fail-static)" {
  agent cp1 full
  cp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  touch "$FAKE_S3/.down"
  advance 10
  agent cp1 full
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
}

@test "markers: markers.json lists ok markers only; refusals are not consumption" {
  echo '{"node":"cp1","instance_uuid":"u","ts":"x","result":"ok"}' | s3_write markers/bootstrap-b1.json
  echo '{"node":"cp1","instance_uuid":"u","ts":"x","result":"refused:member-heartbeat"}' | s3_write markers/restore-r1.json
  agent cp1 full
  jq -e '.markers == ["bootstrap-b1"]' "$TMP/cp1/run/markers.json"
}

@test "markers: prestart's marker requests are uploaded and deleted; a refusal never overwrites an ok" {
  mkdir -p "$TMP/cp1/run/marker-requests"
  echo '{"node":"cp1","instance_uuid":"u","ts":"x","result":"ok"}' >"$TMP/cp1/run/marker-requests/bootstrap-b1.json"
  echo '{"node":"cp1","instance_uuid":"u","ts":"x","result":"refused:probe-window"}' >"$TMP/cp1/run/marker-requests/restore-r-9.json"
  agent cp1 full
  [ ! -e "$TMP/cp1/run/marker-requests/bootstrap-b1.json" ]
  [ ! -e "$TMP/cp1/run/marker-requests/restore-r-9.json" ]
  jq -e '.result == "ok"' "$(s3_path markers/bootstrap-b1.json)"
  jq -e '.result == "refused:probe-window"' "$(s3_path markers/restore-r-9.json)"
  # a refusal after the ok marker exists is dropped, the ok stays
  echo '{"node":"cp1","instance_uuid":"u","ts":"x","result":"refused:late"}' >"$TMP/cp1/run/marker-requests/bootstrap-b1.json"
  agent cp1 full
  jq -e '.result == "ok"' "$(s3_path markers/bootstrap-b1.json)"
  jq -e '.markers == ["bootstrap-b1"]' "$TMP/cp1/run/markers.json"
}

@test "heartbeats: etcd none / member, phase and instance uuid" {
  echo running >"$TMP/cp1/run/phase"
  agent cp1 full
  hb=$(s3_cat heartbeats/cp1.json)
  [ "$(jq -r .etcd <<<"$hb")" = none ]
  [ "$(jq -r .phase <<<"$hb")" = running ]
  [ "$(jq -r .node <<<"$hb")" = cp1 ]
  [ "$(jq -r .instance_uuid <<<"$hb")" = "$(cat "$TMP/cp1/state/instance-uuid")" ]
  [ "$(jq -r .ts <<<"$hb")" = "$(iso "$(now)")" ]
  mkdir -p "$TMP/cp1/k3s/server/db"
  settle cp1
  [ "$(s3_cat heartbeats/cp1.json | jq -r .etcd)" = member ]
}

@test "heartbeats: the holder reports holder, heartbeats.json carries the peers and fetched_at" {
  holder_ready
  settle cp1
  agent cp2 full
  settle cp1
  [ "$(s3_cat heartbeats/cp1.json | jq -r .headscale)" = holder ]
  settle cp2
  jq -e --arg f "$(iso "$(now)")" '.fetched_at == $f and .cp1.headscale == "holder" and .cp2.node == "cp2"' "$TMP/cp2/run/heartbeats.json"
  [ "$(s3_cat heartbeats/cp2.json | jq -r .headscale)" = standby ]
}

# --- host, tailscale, duplicates, backup, health -----------------------------------------------

@test "tailscale: logged out with a CP key runs tailscale up with the WP-03 flags" {
  put_secrets '.tailnet.cp_key = "hskey-cp-secret"'
  echo NeedsLogin >"$(fd cp1)/ts-state"
  agent cp1 full
  line=$(grep 'tailscale up' "$(fd cp1)/host.log")
  [[ "$line" != *"--advertise-tags"* ]] # the key carries the tag; headscale rejects requested tags with a key
  [[ "$line" == *"--reset"* ]]
  [[ "$line" == *"--advertise-routes=10.0.0.1/32"* ]]
  [[ "$line" == *"--accept-routes=false"* ]]
  [[ "$line" == *"--accept-dns=false"* ]]
  [[ "$line" == *"--login-server https://frankenscube.hs.sparecores.net"* ]]
  [[ "$line" != *"hskey-cp-secret"* ]]
  grep -q 'authkey=hskey-cp-secret' "$(fd cp1)/host.log"
  [ ! -e "$TMP/cp1/run/secrets/cp-key" ]
}

@test "tailscale: already logged in, or no CP key yet: nothing" {
  put_secrets '.tailnet.cp_key = "k"'
  echo Running >"$(fd cp1)/ts-state"
  agent cp1 full
  not grep -q 'tailscale up' "$(fd cp1)/host.log"
  put_secrets '.tailnet.cp_key = null'
  echo NeedsLogin >"$(fd cp1)/ts-state"
  settle cp1
  not grep -q 'tailscale up' "$(fd cp1)/host.log"
}

@test "holder: deletes offline duplicate CP nodes by hostname and renames the survivor to the bare hostname" {
  holder_ready
  mkdir -p "$(fd cp1)/hs"
  # headscale 0.29: `name` is the registered hostname, `given_name` is unique (suffixed), and an
  # offline node has no `online` key
  cat >"$(fd cp1)/hs/nodes.json" <<'JSON'
[
 {"id": 1, "name": "cp2", "given_name": "cp2", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "approved_routes": ["10.0.0.2/32"], "available_routes": ["10.0.0.2/32"], "tags": ["tag:frankenscube-cp"]},
 {"id": 4, "name": "cp2", "given_name": "cp2-1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "online": true, "approved_routes": ["10.0.0.2/32"], "available_routes": ["10.0.0.2/32"], "subnet_routes": ["10.0.0.2/32"], "tags": ["tag:frankenscube-cp"]},
 {"id": 2, "name": "cp3", "given_name": "cp3", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "tags": ["tag:frankenscube-cp"]},
 {"id": 5, "name": "cp3", "given_name": "cp3-1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "tags": ["tag:frankenscube-cp"]},
 {"id": 6, "name": "cp1", "given_name": "cp1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "online": true, "tags": ["tag:frankenscube-cp"]},
 {"id": 3, "name": "admin", "given_name": "admin", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "tags": ["tag:frankenscube-admin"]},
 {"id": 9, "name": "admin", "given_name": "admin-1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "online": true, "tags": ["tag:frankenscube-admin"]},
 {"id": 7, "name": "w1", "given_name": "w1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "tags": ["tag:frankenscube-node"]},
 {"id": 8, "name": "w1", "given_name": "w1-1", "user": {"id": 2147455555, "name": "tagged-devices"}, "register_method": 1, "tags": ["tag:frankenscube-node"]}
]
JSON
  agent cp1 full
  settle cp1
  log="$(fd cp1)/headscale.log"
  grep -q 'nodes delete --identifier 1 --force' "$log"
  grep -q 'nodes delete --identifier 2 --force' "$log"
  not grep -q 'nodes delete --identifier [3456789] ' "$log"
  # the online survivor of cp2, and the newest of cp3 (none online), get the bare hostname back
  grep -q 'nodes rename --identifier 4 cp2$' "$log"
  grep -q 'nodes rename --identifier 5 cp3$' "$log"
  not grep -q 'nodes rename --identifier [1236789] ' "$log"
  # deletes come first: the bare name must be free before the rename
  [ "$(grep -n 'delete --identifier 1 ' "$log" | head -n1 | cut -d: -f1)" -lt "$(grep -n 'rename --identifier 4 ' "$log" | head -n1 | cut -d: -f1)" ]
}

@test "holder: two online CP nodes with one hostname: no delete, no rename" {
  holder_ready
  mkdir -p "$(fd cp1)/hs"
  cat >"$(fd cp1)/hs/nodes.json" <<'JSON'
[
 {"id": 1, "name": "cp2", "given_name": "cp2", "online": true, "tags": ["tag:frankenscube-cp"]},
 {"id": 4, "name": "cp2", "given_name": "cp2-1", "online": true, "tags": ["tag:frankenscube-cp"]}
]
JSON
  agent cp1 full
  settle cp1
  not grep -q 'nodes delete' "$(fd cp1)/headscale.log"
  not grep -q 'nodes rename' "$(fd cp1)/headscale.log"
}

@test "daily backup: sqlite .backup to daily/<date>.sqlite3.gz once a day" {
  holder_ready
  agent cp1 full
  echo "headscale database" >"$TMP/cp1/hs/db.sqlite" # headscale creates it
  settle cp1
  day=$(date -u -d "@$(now)" +%Y-%m-%d)
  [ -f "$FAKE_S3/$HSB/daily/$day.sqlite3.gz" ]
  gzip -t "$FAKE_S3/$HSB/daily/$day.sqlite3.gz"
  n=$(grep -c "key=daily/$day" "$FAKE_S3/.calls")
  settle cp1
  [ "$(grep -c "key=daily/$day" "$FAKE_S3/.calls")" -eq "$n" ]
}

@test "health pings: only URLs present in health.pings are called" {
  agent cp1 full
  [ ! -e "$(fd cp1)/curl.log" ] || ! grep -q 'hc-ping' "$(fd cp1)/curl.log"
  put_desired '.health.pings = {agent: "https://hc-ping.example/abc"}'
  settle cp1
  grep -q 'https://hc-ping.example/abc' "$(fd cp1)/curl.log"
  [ "$(grep -c 'hc-ping' "$(fd cp1)/curl.log")" -eq 1 ]
}

@test "apiserver_serving: 200 and 401 serve; 503 and no answer do not" {
  . "$IMAGE_DIR/lib/health.sh"
  fb=$(fd cp1)
  for c in 200 401; do
    echo $c >"$fb/peer-10.0.0.9"
    FAKE_DIR=$fb apiserver_serving https://10.0.0.9:6443/readyz
  done
  echo 503 >"$fb/peer-10.0.0.9"
  not env FAKE_DIR=$fb bash -c ". '$IMAGE_DIR/lib/health.sh'; apiserver_serving https://10.0.0.9:6443/readyz"
  rm "$fb/peer-10.0.0.9"
  not env FAKE_DIR=$fb bash -c ". '$IMAGE_DIR/lib/health.sh'; apiserver_serving https://10.0.0.9:6443/readyz"
}

@test "health pings: the readyz ping fires on a 401 and not on a 503" {
  agent cp1 full
  echo 401 >"$(fd cp1)/peer-10.0.0.9"
  put_desired '.health.readyz_urls = {cp1: "https://10.0.0.9:6443/readyz"} | .health.pings = {readyz: "https://hc-ping.example/rz"}'
  settle cp1
  grep -q 'hc-ping.example/rz' "$(fd cp1)/curl.log"
  : >"$(fd cp1)/curl.log"
  echo 503 >"$(fd cp1)/peer-10.0.0.9"
  settle cp1
  not grep -q 'hc-ping.example/rz' "$(fd cp1)/curl.log"
}

@test "agent.disabled in the desired doc: the holder lets go of the lease" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  put_desired '.agent.disabled = true'
  advance 10
  agent cp1 full
  not hs_running cp1
  [ "$(lease | jq -r .holder)" = "" ]
}

# --- fsync / steal check on fixtures -----------------------------------------------------------

@test "fsync check: nearest-rank p99 and p50 of a fixture" {
  . "$IMAGE_DIR/lib/health.sh"
  [ "$(percentile 99 "$BATS_TEST_DIRNAME/fixtures/fsync-ms.txt")" = "31.00" ]
  [ "$(percentile 50 "$BATS_TEST_DIRNAME/fixtures/fsync-ms.txt")" = "2.10" ]
}

@test "steal check: percentage between two /proc/stat fixtures" {
  . "$IMAGE_DIR/lib/health.sh"
  # 100 jiffies elapsed in total, 12 of them stolen
  [ "$(steal_pct "$BATS_TEST_DIRNAME/fixtures/stat-before" "$BATS_TEST_DIRNAME/fixtures/stat-after")" = "12.0" ]
}

@test "fsync check: a real run produces samples; sustained badness raises the ALERT after 15 min" {
  . "$IMAGE_DIR/lib/health.sh"
  . "$FC_SHARED_LIB/common.sh"
  FC_LOG_TAG=cp-agent
  samples=$(fsync_samples "$TMP" 3)
  [ "$(printf '%s\n' "$samples" | wc -l)" -eq 3 ]
  AGENT_RUN="$TMP/run"
  mkdir -p "$AGENT_RUN"
  _bad_since fsync 1 2>"$TMP/alert.log"
  not grep -q ALERT "$TMP/alert.log"
  advance 899
  _bad_since fsync 1 2>"$TMP/alert.log"
  not grep -q ALERT "$TMP/alert.log"
  advance 1
  _bad_since fsync 1 2>"$TMP/alert.log"
  grep -q 'ALERT fsync' "$TMP/alert.log"
  _bad_since fsync 0 2>"$TMP/alert.log"
  [ ! -e "$AGENT_RUN/bad-fsync" ]
}

# --- fencing: watchdog, bounded renewals, restore off the lease path ---------------------------

@test "watchdog: fences a holder 20 s after its last renewal without any lease tick or S3 access" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  ls_running cp1
  touch "$FAKE_S3/.down"
  echo 30 >"$FAKE_S3/.slow" # S3 would hang for 30 s: the watchdog must not care
  advance 19
  SECONDS=0
  agent cp1 watchdog
  hs_running cp1 # 19 s: still serving
  advance 1
  agent cp1 watchdog
  [ "$SECONDS" -lt 5 ]
  not hs_running cp1
  not ls_running cp1
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
  grep -q 'WATCHDOG FENCE' "$TMP/cp1.log"
}

@test "watchdog: uses the monotonic clock, not the wall clock (a stepped wall clock changes nothing)" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  # fc_mono_epoch is the only clock of the fence: the state records it, not wall time
  jq -e '.last_renew_mono != null and .last_renew == null' "$TMP/cp1/run/agent/lease-state.json"
  grep -q '/proc/uptime' "$IMAGE_DIR/lib/agent.sh"
}

@test "renewal is bounded by the time left until the fence (S3 hangs, the tick still returns before 20 s)" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  advance 18 # 2 s left until the fence
  echo 30 >"$FAKE_S3/.slow"
  SECONDS=0
  agent cp1 lease
  [ "$SECONDS" -lt 6 ] # timeout(20 - 18) = 2 s, not the default 12 s
  # no success: the state is unchanged (still held, last renewal 18 s ago) and headscale still serves
  [ -e "$TMP/cp1/run/agent/lease-state.json" ]
  hs_running cp1
  # at 20 s the next check fences
  advance 2
  agent cp1 watchdog
  not hs_running cp1
}

@test "renewal: a successful renewal moves the fence out by 20 s" {
  holder_ready
  agent cp1 lease
  advance 10
  agent cp1 lease
  advance 15 # 25 s after the first renewal, 15 s after the second
  agent cp1 watchdog
  hs_running cp1
  advance 5
  agent cp1 watchdog
  not hs_running cp1
}

@test "restore runs as a job: the lease tick returns at once and keeps renewing during a slow restore" {
  holder_ready
  put_replica 4
  jq -n --arg e "$(iso $((START_NOW - 100)))" '{holder: "cp3", instance_uuid: "u3", epoch: 4, expires_at: $e}' | s3_write lease/headscale.json
  echo 4 >"$(fd cp2)/restore-slow"
  SECONDS=0
  NOWAIT=1 agent cp2 lease
  [ "$SECONDS" -lt 3 ]
  [ "$(lease_holder)" = cp2 ]
  not hs_running cp2 # still restoring
  advance 10
  NOWAIT=1 agent cp2 lease # the lease is renewed while the restore job runs
  [ "$(lease | jq -r .expires_at)" = "$(iso $(($(now) + 30)))" ]
  wait_start_job cp2
  hs_running cp2
  [ "$(cat "$TMP/cp2/hs/db.sqlite")" = "replica of gen-4" ]
}

@test "fence while the start job is still restoring: nothing is started behind the stop" {
  holder_ready
  put_replica 4
  jq -n --arg e "$(iso $((START_NOW - 100)))" '{holder: "cp3", instance_uuid: "u3", epoch: 4, expires_at: $e}' | s3_write lease/headscale.json
  echo 3 >"$(fd cp2)/restore-slow"
  NOWAIT=1 agent cp2 lease
  advance 20
  agent cp2 watchdog
  sleep 4
  not hs_running cp2
  not ls_running cp2
}

# --- derp map ----------------------------------------------------------------------------------

@test "derp map: written to derp.yaml atomically before headscale starts, rewritten and restarted on change" {
  holder_ready
  put_desired '.headscale.derp_map = "regions:\n  900: {regionid: 900}\n" | .directives.headscale_bootstrap = {id: "hb1", not_after: "2099-01-01T00:00:00Z"}'
  agent cp1 full
  hs_running cp1
  [ "$(cat "$TMP/cp1/hs/derp.yaml")" = "$(printf 'regions:\n  900: {regionid: 900}')" ]
  [ ! -e "$TMP/cp1/hs/derp.yaml.tmp" ]
  [ ! -s "$(fd cp1)/restarts.log" ]
  put_desired '.headscale.derp_map = "regions:\n  901: {regionid: 901}\n" | .directives.headscale_bootstrap = {id: "hb1", not_after: "2099-01-01T00:00:00Z"}'
  settle cp1
  [ "$(cat "$TMP/cp1/hs/derp.yaml")" = "$(printf 'regions:\n  901: {regionid: 901}')" ]
  grep -q 'restart headscale' "$(fd cp1)/restarts.log"
}

@test "derp map: absent means no file; a non-string value makes the doc invalid" {
  holder_ready
  agent cp1 full
  [ ! -e "$TMP/cp1/hs/derp.yaml" ]
  cp "$TMP/cp1/run/desired.json" "$TMP/before.json"
  put_desired '.headscale.derp_map = 5 | .headscale.mode = "stopped"'
  settle cp1
  cmp "$TMP/cp1/run/desired.json" "$TMP/before.json"
}

# --- restore.env: the newest snapshot, resolved by the agent ------------------------------------

@test "restore.env: the agent resolves the newest etcd snapshot while a restore directive exists" {
  mkdir -p "$FAKE_S3/sc-frankenscube-etcd/snapshots"
  for n in etcd-snapshot-cp1-100 etcd-snapshot-cp2-300 etcd-snapshot-cp3-200.zip; do echo x >"$FAKE_S3/sc-frankenscube-etcd/snapshots/$n"; done
  agent cp1 full
  not grep -q ETCD_S3_LATEST_SNAPSHOT "$TMP/cp1/run/secrets/restore.env" # no restore directive: no listing
  put_desired ".directives.restore = {id: \"r1\", target: \"cp1\", snapshot: \"latest\", not_after: \"$(iso $((START_NOW + 7200)))\"}"
  settle cp1
  grep -q '^ETCD_S3_LATEST_SNAPSHOT=etcd-snapshot-cp2-300$' "$TMP/cp1/run/secrets/restore.env"
  echo x >"$FAKE_S3/sc-frankenscube-etcd/snapshots/etcd-snapshot-cp1-400.zip"
  settle cp1
  grep -q '^ETCD_S3_LATEST_SNAPSHOT=etcd-snapshot-cp1-400.zip$' "$TMP/cp1/run/secrets/restore.env"
  # a failed listing keeps the previous value
  touch "$FAKE_S3/.down-r2"
  settle cp1
  grep -q '^ETCD_S3_LATEST_SNAPSHOT=etcd-snapshot-cp1-400.zip$' "$TMP/cp1/run/secrets/restore.env"
}

# --- reset waits for the private NIC -----------------------------------------------------------

@test "reset: waits for the node IP on the private NIC before touching k3s" {
  export FC_NIC_POLL=0.2
  reset_directive nic1
  age_generation cp1
  touch "$(fd cp1)/ip-missing"
  local i
  for i in $(seq 1 9); do settle cp1; done
  sleep 1
  [ ! -e "$(fd cp1)/host.log" ] || ! grep -q 'systemctl stop k3s' "$(fd cp1)/host.log"
  rm "$(fd cp1)/ip-missing"
  sleep 1
  settle cp1
  grep -q 'systemctl stop k3s' "$(fd cp1)/host.log"
}

# --- heartbeat age: S3's clock -----------------------------------------------------------------

@test "heartbeats.json carries S3's LastModified of each heartbeat as last_modified" {
  agent cp1 full
  advance 30
  agent cp2 full
  advance 30
  agent cp1 full
  jq -e --arg lm "$(iso $((START_NOW + 30)))" '.cp2.last_modified | startswith(($lm | sub("Z$"; "")))' "$TMP/cp1/run/heartbeats.json"
}

# --- races between the watchdog and the lease path ---------------------------------------------

@test "race: containers started right after a fence are stopped again (epoch re-checked after docker run)" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  rm "$(fd cp1)/containers/headscale" # headscale died: the next tick restarts it
  echo 3 >"$(fd cp1)/run-slow"        # docker run takes 3 s
  advance 5
  NOWAIT=1 agent cp1 lease &          # renews, then starts headscale (slow)
  bgpid=$!
  sleep 1.2
  advance 20                          # 20 s since the renewal of that tick: the watchdog fences
  agent cp1 watchdog
  wait "$bgpid"
  not hs_running cp1
  not ls_running cp1
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
}

@test "race: a renewal that succeeds while the watchdog fences does not bring the cleared state back" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  advance 18
  echo 1.2 >"$FAKE_S3/.slow" # the renewal (2 s left) succeeds after 1.2 s
  NOWAIT=1 agent cp1 lease &
  bgpid=$!
  sleep 0.4
  advance 2 # the fence time passes while the renewal is in flight
  agent cp1 watchdog
  wait "$bgpid"
  rm "$FAKE_S3/.slow"
  not hs_running cp1
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
  grep -q 'renewal succeeded after a fence' "$TMP/cp1.log"
  # the next ticks neither resurrect headscale nor refuse to start it later: once the lease expired
  # it is taken with epoch + 1 and headscale starts again from the replica
  put_replica 1
  advance 45
  agent cp1 lease
  [ "$(lease_epoch)" -eq 2 ]
  hs_running cp1
}

@test "watchdog fails closed: an unreadable lease state is fenced" {
  holder_ready
  agent cp1 lease
  hs_running cp1
  echo 'not json' >"$TMP/cp1/run/agent/lease-state.json"
  agent cp1 watchdog
  not hs_running cp1
  [ ! -e "$TMP/cp1/run/agent/lease-state.json" ]
}

@test "litestream: sync-interval is pinned to 10s whatever the desired doc says" {
  holder_ready
  put_desired '.headscale.litestream.sync_interval = "1s" | .directives.headscale_bootstrap = {id: "hb1", not_after: "2099-01-01T00:00:00Z"}'
  agent cp1 full
  hs_running cp1
  grep -q 'sync-interval: 10s' "$TMP/cp1/hs/litestream.yml"
  not grep -q '1s$' "$TMP/cp1/hs/litestream.yml"
}
