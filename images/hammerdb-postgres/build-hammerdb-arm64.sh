#!/usr/bin/env bash
# Build HammerDB production distribution from source on Linux arm64 via BAWT.
set -euo pipefail

: "${HAMMERDB_VERSION:?HAMMERDB_VERSION required}"

NUM_JOBS="${NUM_JOBS:-$(nproc)}"
INSTALL_ROOT="${HAMMERDB_INSTALL_ROOT:-/home}"
BAWT_SETUP="${BAWT_SETUP:-$(cd "$(dirname "$0")" && pwd)/HammerDB-Postgres-Linux.bawt}"
HAMMERDB_REPO="${HAMMERDB_REPO:-https://github.com/TPC-Council/HammerDB.git}"

cleanup() {
  if [ "${_cloned:-}" = "1" ] && [ -n "${HAMMERDB_SRC:-}" ]; then
    rm -rf "${HAMMERDB_SRC}"
  fi
}
trap cleanup EXIT

if [ -z "${HAMMERDB_SRC:-}" ]; then
  HAMMERDB_SRC="$(mktemp -d)"
  _cloned=1
  git clone --depth 1 --branch "v${HAMMERDB_VERSION}" "${HAMMERDB_REPO}" "${HAMMERDB_SRC}"
fi

BAWT_DIR="${HAMMERDB_SRC}/Build/Bawt-2.1.0"
SETUP="Setup/HammerDB-Postgres-Linux.bawt"

if [ ! -f "${BAWT_SETUP}" ]; then
  echo "build-hammerdb-arm64: missing BAWT setup file ${BAWT_SETUP}" >&2
  exit 1
fi
cp "${BAWT_SETUP}" "${BAWT_DIR}/${SETUP}"

if ! command -v pg_config >/dev/null; then
  echo "build-hammerdb-arm64: pg_config not found (install libpq-dev)" >&2
  exit 1
fi
# BAWT/pgtcl expects PG_CONFIG to be the directory containing pg_config, not the binary.
export PG_CONFIG="${PG_CONFIG:-$(dirname "$(command -v pg_config)")}"
export DEBIAN_FRONTEND=noninteractive
# Tcl/Tk 9 zipfs prefers system zip; without it the build falls back to minizip and
# can fail under parallel make (see Tcl ticket b38c726c / LFS #5570).
if ! command -v zip >/dev/null; then
  echo "build-hammerdb-arm64: zip(1) required for Tcl/Tk zipfs build" >&2
  exit 1
fi

cd "${BAWT_DIR}"

patch_bawt_for_aarch64() {
  [ "$(uname -m)" = "aarch64" ] || return 0
  local bawt_tcl="${BAWT_DIR}/Bawt.tcl"
  if grep -q 'aarch64-unknown-linux-gnu' "${bawt_tcl}"; then
    return 0
  fi
  python3 - "${bawt_tcl}" <<'PY'
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
needle = """        if { [IsWindows] } {
            append cmd "--build=[GetMingwVersion] "
        }
"""
insert = """        if { [IsLinux] } {
            catch {
                if {[string equal [exec uname -m] aarch64]} {
                    append cmd "--build=aarch64-unknown-linux-gnu "
                    append cmd "--host=aarch64-unknown-linux-gnu "
                }
            }
        }
"""
if needle not in text:
    raise SystemExit("build-hammerdb-arm64: Bawt.tcl aarch64 patch point not found")
patched = text.replace(needle, needle + insert)
if patched == text:
    raise SystemExit("build-hammerdb-arm64: Bawt.tcl aarch64 patch made no changes")
open(path, "w", encoding="utf-8").write(patched)
PY
}
patch_bawt_for_aarch64

# Upstream Build-Linux.sh uses tclkit-Linux64 (amd64-only) to run Bawt.tcl.
# System tclsh runs the same orchestrator on arm64. --architecture x64 is BAWT's
# "64-bit" mode (output under BawtBuild/Linux/x64/...); gcc still targets the host.
tclsh Bawt.tcl \
  --rootdir ../BawtBuild \
  --architecture x64 \
  --numjobs "${NUM_JOBS}" \
  --url http://www.hammerdb.com/build5 \
  --finalizefile Setup/HammerDBFinalize.bawt \
  --update "${SETUP}" all

dist_dir="../BawtBuild/Linux/x64/Release/Distribution"
prod_tgz=""
for candidate in \
  "${dist_dir}/HammerDB-${HAMMERDB_VERSION}-Prod-Lin.tar.gz" \
  "${dist_dir}/HammerDB-${HAMMERDB_VERSION}-Prod-Linux.tar.gz"; do
  if [ -f "${candidate}" ]; then
    prod_tgz="${candidate}"
    break
  fi
done
if [ -z "${prod_tgz}" ]; then
  prod_tgz="$(find "${dist_dir}" -maxdepth 1 -name 'HammerDB-*-Prod-Lin*.tar.gz' | head -1)"
fi
if [ -z "${prod_tgz}" ] || [ ! -f "${prod_tgz}" ]; then
  echo "build-hammerdb-arm64: production tarball not found under ${dist_dir}" >&2
  ls -la "${dist_dir}" >&2 || true
  exit 1
fi

mkdir -p "${INSTALL_ROOT}"
tar -xzf "${prod_tgz}" -C "${INSTALL_ROOT}"
ln -sfn "${INSTALL_ROOT}/HammerDB-${HAMMERDB_VERSION}" "${INSTALL_ROOT}/hammerdb"

echo "build-hammerdb-arm64: installed ${prod_tgz} -> ${INSTALL_ROOT}/hammerdb"
