#!/usr/bin/env bash
# Smoke test: the image's derper starts and prints its flags (-h exits 0 or 2 depending on the flag package), and
# serves TLS with -certmode letsencrypt on a port other than 443 (the Dockerfile's patch; no certificate is requested
# until a client connects).
set -euo pipefail
image=${1:?usage: smoke.sh <image>}
out=$(docker run --rm "$image" -h 2>&1 || true)
grep -q -- '-verify-clients' <<<"$out"
grep -q -- '-certmode' <<<"$out"
grep -q -- '-stun-port' <<<"$out"
cid=$(docker run -d "$image" -hostname derp.example.com -a :8443 -http-port -1 -stun=false \
  -certmode letsencrypt -certdir /tmp/certs -c /tmp/derper.key)
trap 'docker rm -f "$cid" >/dev/null' EXIT
for _ in $(seq 20); do
  if docker logs "$cid" 2>&1 | grep -q 'derper: serving on :8443 with TLS'; then
    echo "derper smoke test ok"
    exit 0
  fi
  sleep 0.5
done
docker logs "$cid" 2>&1 | tail -20
echo "derper does not serve TLS on :8443 with -certmode letsencrypt" >&2
exit 1
