#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# The <hs> certificate: stored in S3 (certs/hs/), issued by lego (ACME DNS-01 against Route53) by
# the lease holder only when none is stored or the stored one has under 30 days left. Issuance
# runs as a background job so the lease tick never blocks.

CERT_RENEW_BEFORE_SECONDS=$((30 * 86400))
CERT_USABLE_SECONDS=60
CERT_RETRY_SECONDS=300

CERT_DIR() { printf '%s/tls' "$FC_HS_DIR"; }

# cert_valid <fullchain> <seconds>: still valid for that long?
cert_valid() {
  [ -s "$1" ] && openssl x509 -in "$1" -noout -checkend "$2" >/dev/null 2>&1
}

cert_install() {
  local chain=$1 key=$2 d
  d=$(CERT_DIR)
  mkdir -p "$d"
  (
    umask 077
    cp "$key" "$d/privkey.pem.tmp"
    cp "$chain" "$d/fullchain.pem.tmp"
  )
  mv -f "$d/privkey.pem.tmp" "$d/privkey.pem"
  mv -f "$d/fullchain.pem.tmp" "$d/fullchain.pem"
  CERT_CHANGED=1
}

# Job body: issue with lego and leave the files in $AGENT_RUN/cert-new/.
cert_issue_job() {
  local zone email staging domain server=() out
  zone=$(desired_jq -r '.headscale.acme.zone_id')
  email=$(desired_jq -r '.headscale.acme.email')
  staging=$(desired_jq -r '.headscale.acme.staging // false')
  domain=${HS_NAME:-$(desired_jq -r '.headscale.url' | sed 's|^https://||; s|/.*||')}
  if [ "$staging" = true ]; then
    server=(--server https://acme-staging-v02.api.letsencrypt.org/directory)
  fi
  mkdir -p "$FC_LEGO_DIR"
  AWS_HOSTED_ZONE_ID=$zone lego --accept-tos --email "$email" --dns route53 --domains "$domain" \
    --path "$FC_LEGO_DIR" "${server[@]}" run || return 1
  out="$AGENT_RUN/cert-new"
  rm -rf "$out"
  mkdir -p "$out"
  cp "$FC_LEGO_DIR/certificates/${domain}.crt" "$out/fullchain.pem"
  cp "$FC_LEGO_DIR/certificates/${domain}.key" "$out/privkey.pem"
}

# cert_ensure: returns 0 when a usable certificate is installed locally (after trying to get a
# fresh one), 1 when there is none. Sets CERT_CHANGED=1 when the installed files changed.
cert_ensure() {
  local d tmpc tmpk rc=0 state backoff now
  CERT_CHANGED=0
  d=$(CERT_DIR)
  mkdir -p "$d"
  if cert_valid "$d/fullchain.pem" "$CERT_RENEW_BEFORE_SECONDS" && [ -s "$d/privkey.pem" ]; then
    return 0
  fi

  # a valid stored certificate is always preferred over issuing one
  tmpc=$(mktemp)
  tmpk=$(mktemp)
  rc=0
  s3_get certs/hs/fullchain.pem "$tmpc" >/dev/null || rc=$?
  if [ "$rc" -eq 0 ]; then
    s3_get certs/hs/privkey.pem "$tmpk" >/dev/null || rc=$?
  fi
  if [ "$rc" -eq 0 ] && cert_valid "$tmpc" "$CERT_RENEW_BEFORE_SECONDS"; then
    cert_install "$tmpc" "$tmpk"
    rm -f "$tmpc" "$tmpk"
    fc_log "certificate installed from certs/hs/"
    return 0
  fi
  if [ "$rc" -eq 0 ] && cert_valid "$tmpc" "$CERT_USABLE_SECONDS" && ! cert_valid "$d/fullchain.pem" "$CERT_USABLE_SECONDS"; then
    # stop-gap: the stored certificate is short of the 30 d margin but still valid; serve with
    # it while the renewal below runs
    cert_install "$tmpc" "$tmpk"
    fc_log "serving with the stored certificate (renewal due)"
  fi
  rm -f "$tmpc" "$tmpk"
  if [ "$rc" -ne 0 ] && [ "$rc" -ne "$S3_NOTFOUND" ]; then
    # S3 is not readable: a certificate may well be stored. Never issue blind.
    fc_log "cannot read certs/hs/ (rc=$rc): not issuing"
    cert_valid "$d/fullchain.pem" "$CERT_USABLE_SECONDS"
    return
  fi

  state=$(bg_state lego)
  now=$(fc_now_epoch)
  case "$state" in
    "done")
      if [ "$(bg_rc lego)" = 0 ] && cert_valid "$AGENT_RUN/cert-new/fullchain.pem" 86400; then
        if s3_put certs/hs/privkey.pem "$AGENT_RUN/cert-new/privkey.pem" >/dev/null \
          && s3_put certs/hs/fullchain.pem "$AGENT_RUN/cert-new/fullchain.pem" >/dev/null; then
          fc_log "certificate stored in certs/hs/"
        else
          fc_log "could not store the new certificate in S3"
        fi
        cert_install "$AGENT_RUN/cert-new/fullchain.pem" "$AGENT_RUN/cert-new/privkey.pem"
      else
        fc_log "certificate issuance failed; retry in ${CERT_RETRY_SECONDS} s"
        echo "$now" >"$AGENT_RUN/cert-failed-at"
      fi
      bg_clear lego
      rm -rf "$AGENT_RUN/cert-new"
      ;;
    running) ;;
    none)
      backoff=0
      if [ -r "$AGENT_RUN/cert-failed-at" ]; then
        backoff=$(cat "$AGENT_RUN/cert-failed-at")
      fi
      if [ $((now - backoff)) -ge "$CERT_RETRY_SECONDS" ]; then
        fc_log "issuing the certificate (lego, DNS-01)"
        bg_start lego cert_issue_job
      fi
      ;;
  esac
  cert_valid "$d/fullchain.pem" "$CERT_USABLE_SECONDS"
}
