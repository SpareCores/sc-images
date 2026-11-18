#!/usr/bin/env bash
# Smoke test: the image's derper starts and prints its flags (-h exits 0 or 2 depending on the flag package).
set -euo pipefail
image=${1:?usage: smoke.sh <image>}
out=$(docker run --rm "$image" -h 2>&1 || true)
grep -q -- '-verify-clients' <<<"$out"
grep -q -- '-certmode' <<<"$out"
grep -q -- '-stun-port' <<<"$out"
echo "derper smoke test ok"
