#!/usr/bin/env bash
# Discover image folders and compute a dependency-ordered build plan.
#
# Each images/<folder> may declare dependencies in a DEPENDS_ON file (one image
# folder name per line). Folders are assigned a topological "level": level 0 has
# no dependencies; a folder's level is max(level of deps) + 1. The workflow builds
# one level at a time (publishing each level before the next starts), so a
# dependency is always available as a base image for its dependents.
#
# Outputs (to GITHUB_OUTPUT or stdout), for L in 0..MAX_LEVELS-1:
#   levelL=<json array of {folder,arch,platform}>   # build matrix for the level
#   folders_levelL=<json array of folder names>      # merge matrix for the level
#
# A folder is discovered if it contains a Dockerfile or a prepare.sh hook.
set -euo pipefail

MAX_LEVELS="${MAX_LEVELS:-4}"
IMAGES_DIR="${IMAGES_DIR:-images}"

cd "$IMAGES_DIR" 2>/dev/null || { echo "resolve-build-plan: run from repo root (no $IMAGES_DIR/)" >&2; exit 1; }

read_list() {
  # Strip comments/blank lines, split on whitespace, one token per line.
  sed 's/#.*//' "$1" 2>/dev/null | tr -s ' \t' '\n' | sed '/^[[:space:]]*$/d'
}

folders=()
for d in */; do
  f="${d%/}"
  if [ -f "$f/Dockerfile" ] || [ -f "$f/prepare.sh" ]; then
    folders+=("$f")
  fi
done

declare -A LEVEL DEPS PLAT
for f in "${folders[@]}"; do
  LEVEL["$f"]=0
  if [ -f "$f/DEPENDS_ON" ]; then
    DEPS["$f"]="$(read_list "$f/DEPENDS_ON" | tr '\n' ' ')"
  else
    DEPS["$f"]=""
  fi
  if [ -f "$f/PLATFORMS" ]; then
    PLAT["$f"]="$(read_list "$f/PLATFORMS" | tr '\n' ' ')"
  else
    PLAT["$f"]="amd64 arm64"
  fi
done

# Validate declared dependencies exist as discovered folders.
for f in "${folders[@]}"; do
  for dep in ${DEPS["$f"]}; do
    if [ -z "${LEVEL[$dep]+x}" ]; then
      echo "::error::image '$f' depends on unknown image '$dep'" >&2
      exit 1
    fi
  done
done

# Relax levels until stable (DAG => converges in <= |folders| passes).
for ((i = 0; i <= ${#folders[@]}; i++)); do
  changed=0
  for f in "${folders[@]}"; do
    for dep in ${DEPS["$f"]}; do
      want=$(( LEVEL["$dep"] + 1 ))
      if [ "$want" -gt "${LEVEL[$f]}" ]; then
        LEVEL["$f"]=$want
        changed=1
      fi
    done
  done
  [ "$changed" -eq 0 ] && break
done

# Detect cycles / excessive depth.
for f in "${folders[@]}"; do
  if [ "${LEVEL[$f]}" -ge "$MAX_LEVELS" ]; then
    echo "::error::dependency depth exceeds MAX_LEVELS=$MAX_LEVELS (image '$f' at level ${LEVEL[$f]}); raise MAX_LEVELS and add a level wave in push.yml" >&2
    exit 1
  fi
done

plan=""
for f in "${folders[@]}"; do
  for a in ${PLAT["$f"]}; do
    plan+="${LEVEL[$f]} $f $a"$'\n'
  done
done

echo "Build plan (level folder arch):" >&2
printf '%s' "$plan" | sort -n >&2

for ((L = 0; L < MAX_LEVELS; L++)); do
  lvl_json=$(printf '%s' "$plan" \
    | awk -v L="$L" '$1==L{print $2" "$3}' \
    | jq -R -s -c 'split("\n")|map(select(length>0))|map(split(" "))|map({folder:.[0],arch:.[1],platform:("linux/"+.[1]),name:(.[0]+" ("+.[1]+")")})')
  fld_json=$(printf '%s' "$plan" \
    | awk -v L="$L" '$1==L{print $2}' \
    | sort -u \
    | jq -R -s -c 'split("\n")|map(select(length>0))')
  echo "level$L=$lvl_json"
  echo "folders_level$L=$fld_json"
done
