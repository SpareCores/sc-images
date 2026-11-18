# Shared helpers of the cp-agent suite (bats-core).
# shellcheck shell=bash

IMAGE_DIR="$(cd "$BATS_TEST_DIRNAME/../.." && pwd)"
AGENT="$IMAGE_DIR/bin/cp-agent"
START_NOW=1800000000
CTRL=sc-frankenscube-control
HSB=sc-frankenscube-headscale

iso() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

setup_env() {
  TMP="$(mktemp -d)"
  export FAKE_S3="$TMP/s3"
  export FC_NOW_FILE="$TMP/now"
  export FC_SHARED_LIB="$IMAGE_DIR/lib/shared"
  export PATH="$BATS_TEST_DIRNAME/fakes:$PATH"
  mkdir -p "$FAKE_S3/$CTRL" "$FAKE_S3/$HSB"
  echo "$START_NOW" >"$FC_NOW_FILE"
  for n in cp1 cp2 cp3; do mkdir -p "$TMP/$n"/{run,state,hs,k3s,fake}; done
  put_desired '.'
  put_secrets '.'
  make_cert 90 "$TMP/cert-valid"
}

teardown_env() { sleep 0.3; rm -rf "${TMP:?}" || { sleep 1; rm -rf "${TMP:?}"; }; }

now() { cat "$FC_NOW_FILE"; }
advance() { echo $(($(now) + $1)) >"$FC_NOW_FILE"; }

# agent <node> <subcommand>: one run of the agent as that node (own run/state/hs/k3s dirs)
agent() {
  local node=$1 idx=${1#cp} cmd=${2:-full}
  shift 2 || true
  env NODE_NAME="$node" CLUSTER=frankenscube STACK=frankenscube CONTROL_BUCKET="$CTRL" PREFIX=frankenscube/ \
    AWS_REGION=us-east-1 NODE_IP="10.0.0.$idx" GENERATION="${GENERATION:-1}" HS_NAME=frankenscube.hs.sparecores.net \
    PEERS="${PEERS_OF_NODE:-10.0.0.2 10.0.0.3}" DISABLE_AGENT=false FC_NODE_ENV=/nonexistent FAKE_WHO="$node" \
    FAKE_DIR="$TMP/$node/fake" FC_RUN_DIR="$TMP/$node/run" FC_STATE_DIR="$TMP/$node/state" FC_HS_DIR="$TMP/$node/hs" \
    FC_K3S_DIR="$TMP/$node/k3s" FC_LEGO_DIR="$TMP/$node/state/lego" FSYNC_SAMPLE_COUNT=3 \
    bash "$AGENT" "$cmd" "$@" 2>>"$TMP/$node.log"
}

# peers of cp1 are cp2, cp3 (and so on): the interlock probes these addresses
peers_of() {
  case "$1" in
    cp1) echo "10.0.0.2 10.0.0.3" ;;
    cp2) echo "10.0.0.1 10.0.0.3" ;;
    *) echo "10.0.0.1 10.0.0.2" ;;
  esac
}

# --- the fake S3 -------------------------------------------------------------------------------

s3_path() { printf '%s/%s/frankenscube/%s' "$FAKE_S3" "$CTRL" "$1"; }
s3_has() { [ -f "$(s3_path "$1")" ]; }
s3_cat() { cat "$(s3_path "$1")"; }
s3_write() { mkdir -p "$(dirname "$(s3_path "$1")")"; cat >"$(s3_path "$1")"; }
lease() { s3_cat lease/headscale.json; }
lease_holder() { s3_cat lease/headscale.json | jq -r .holder; }
lease_epoch() { s3_cat lease/headscale.json | jq -r .epoch; }

# put_desired '<jq filter over the default doc>'
put_desired() {
  local base
  base=$(jq -n --arg na "$(iso $((START_NOW + 86400)))" '{
    version: 1, cluster: "frankenscube", stack: "frankenscube",
    nodes: {
      cp1: {generation: 1, zone: "de-fra1", private_ip: "10.0.0.1", public_ipv4: "198.51.100.1"},
      cp2: {generation: 1, zone: "nl-ams1", private_ip: "10.0.0.2", public_ipv4: "198.51.100.2"},
      cp3: {generation: 1, zone: "uk-lon1", private_ip: "10.0.0.3", public_ipv4: "198.51.100.3"}},
    peers: ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    images: {headscale: "headscale@sha256:aaa", litestream: "litestream@sha256:bbb", derper: "derper@sha256:ccc", cp_agent: "agent@sha256:ddd"},
    headscale: {mode: "auto", url: "https://frankenscube.hs.sparecores.net",
      config: "server_url: https://frankenscube.hs.sparecores.net\nlisten_addr: 0.0.0.0:443\n",
      litestream: {bucket: "sc-frankenscube-headscale", endpoint: "https://acct.eu.r2.cloudflarestorage.com", sync_interval: "10s"},
      acme: {zone_id: "Z123", record: "_acme-challenge.frankenscube.hs.sparecores.net", email: "ops@example.com", staging: false}},
    directives: {bootstrap: null, restore: null, reset: null, headscale_bootstrap: null},
    health: {readyz_urls: {}, pings: {}},
    agent: {disabled: false, etcd_snapshots: true}}')
  printf '%s' "$base" | jq "$1" | s3_write desired.json
}

put_secrets() {
  local base
  base='{"version":1,"tailnet":{"cp_key":null},"litestream":{"access_key_id":"LSAK","secret_access_key":"LSSK"},"restore":{"bucket":"sc-frankenscube-etcd","endpoint":"https://acct.eu.r2.cloudflarestorage.com","region":"auto","access_key_id":"RAK","secret_access_key":"RSK"}}'
  printf '%s' "$base" | jq "$1" | s3_write desired-secrets.json
}

# a directive that is valid for a day: dir_obj <id> [target]
bootstrap_hs_directive() {
  put_desired ".directives.headscale_bootstrap = {id: \"$1\", not_after: \"$(iso $((START_NOW + 86400)))\"}"
}

make_cert() {
  local days=$1 prefix=$2
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days "$days" -subj "/CN=frankenscube.hs.sparecores.net" \
    -keyout "$prefix.key" -out "$prefix.crt" 2>/dev/null
}

store_cert() {
  cp "$1.crt" "$(s3_path certs/hs/fullchain.pem)" 2>/dev/null || {
    mkdir -p "$(dirname "$(s3_path certs/hs/fullchain.pem)")"
    cp "$1.crt" "$(s3_path certs/hs/fullchain.pem)"
  }
  cp "$1.key" "$(s3_path certs/hs/privkey.pem)"
}

# headscale replica generations live in the headscale bucket
put_replica() { # gen
  mkdir -p "$FAKE_S3/$HSB/gen-$1" "$FAKE_S3/.replicas"
  echo data >"$FAKE_S3/$HSB/gen-$1/0001.ltx"
  echo "replica of gen-$1" >"$FAKE_S3/.replicas/gen-$1"
}

# the agent has been alive long enough for the generation delay
age_generation() {
  echo "1 $((START_NOW - 100000))" >"$TMP/$1/state/generation-seen"
}

# per-node fake host state
fd() { printf '%s/%s/fake' "$TMP" "$1"; }
hs_running() { [ -e "$(fd "$1")/containers/headscale" ]; }
ls_running() { [ -e "$(fd "$1")/containers/litestream" ]; }
docker_log() { cat "$(fd "$1")/docker.log" 2>/dev/null; }
wait_bg() { # node name
  local i f="$TMP/$1/run/agent/bg/$2.rc"
  for i in $(seq 1 100); do [ -e "$f" ] && return 0; sleep 0.1; done
  return 1
}

# settle <node>: the 10 s lease ticks and the 30 s full cycle of 30 s of real time
settle() {
  advance 10
  agent "$1" lease
  advance 10
  agent "$1" lease
  advance 10
  agent "$1" full
}
