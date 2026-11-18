#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # globals are shared between the sourced files
# S3 access through `aws s3api` (control bucket) and the R2 endpoint (headscale bucket).
# Conditional writes: aws-cli >= 2.27 `put-object --if-match / --if-none-match` (the Dockerfile
# pins aws-cli and asserts the flags exist). Return codes of the helpers:
S3_NOTFOUND=44
S3_PRECONDITION=12

_aws() {
  timeout "${AWS_CALL_TIMEOUT:-12}" aws --cli-connect-timeout 3 --cli-read-timeout 8 "$@"
}

_s3_classify() {
  # $1 = stderr file; prints notfound | precondition | error
  if grep -q -E 'NoSuchKey|Not Found|\(404\)' "$1"; then
    echo notfound
  elif grep -q -E 'PreconditionFailed|ConditionalRequestConflict|\(412\)|\(409\)' "$1"; then
    echo precondition
  else
    echo error
  fi
}

# s3_get <key> <outfile>: key relative to the prefix. Prints the ETag (with quotes).
s3_get() {
  local key=$1 out=$2 tmp err json kind
  tmp="$out.dl.$$"
  err="$out.err.$$"
  if json=$(_aws --region "$AWS_REGION" s3api get-object --bucket "$CONTROL_BUCKET" --key "${PREFIX}${key}" "$tmp" 2>"$err"); then
    rm -f "$err"
    mv -f "$tmp" "$out"
    printf '%s' "$json" | jq -r '.ETag // empty'
    return 0
  fi
  rm -f "$tmp"
  kind=$(_s3_classify "$err")
  rm -f "$err"
  case "$kind" in
    notfound) return "$S3_NOTFOUND" ;;
    *) return 1 ;;
  esac
}

# s3_put <key> <file> [if-match <etag> | if-none-match]: prints the new ETag.
s3_put() {
  local key=$1 file=$2 mode=${3:-} etag=${4:-} err json kind
  local cond=()
  case "$mode" in
    if-match) cond=(--if-match "$etag") ;;
    if-none-match) cond=(--if-none-match '*') ;;
    '') ;;
    *) return 1 ;;
  esac
  err=$(mktemp)
  if json=$(_aws --region "$AWS_REGION" s3api put-object --bucket "$CONTROL_BUCKET" --key "${PREFIX}${key}" \
    --body "$file" "${cond[@]}" 2>"$err"); then
    rm -f "$err"
    printf '%s' "$json" | jq -r '.ETag // empty'
    return 0
  fi
  kind=$(_s3_classify "$err")
  rm -f "$err"
  case "$kind" in
    precondition) return "$S3_PRECONDITION" ;;
    *) return 1 ;;
  esac
}

# s3_list <prefix>: keys (relative to PREFIX), one per line.
s3_list() {
  local json
  json=$(_aws --region "$AWS_REGION" s3api list-objects-v2 --bucket "$CONTROL_BUCKET" --prefix "${PREFIX}$1") || return 1
  printf '%s' "$json" | jq -r --arg p "$PREFIX" '.Contents[]?.Key | ltrimstr($p)'
}

# --- R2 (litestream bucket) with the litestream credentials of the secrets doc ------------------

_r2() {
  local ep ak sk
  ep=$(desired_jq -r '.headscale.litestream.endpoint')
  ak=$(secrets_jq -r '.litestream.access_key_id')
  sk=$(secrets_jq -r '.litestream.secret_access_key')
  AWS_ACCESS_KEY_ID=$ak AWS_SECRET_ACCESS_KEY=$sk AWS_SESSION_TOKEN='' \
    _aws --endpoint-url "$ep" --region auto "$@"
}

# r2_highest_gen: prints the highest N of the gen-N/ prefixes of the headscale bucket (nothing
# when there is none); fails on an error.
r2_highest_gen() {
  local bucket json
  bucket=$(desired_jq -r '.headscale.litestream.bucket')
  json=$(_r2 s3api list-objects-v2 --bucket "$bucket" --delimiter / --prefix gen-) || return 1
  printf '%s' "$json" | jq -r '[.CommonPrefixes[]?.Prefix | capture("^gen-(?<n>[0-9]+)/$").n | tonumber] | max // empty'
}

r2_put_new() {
  local key=$1 file=$2 bucket err
  bucket=$(desired_jq -r '.headscale.litestream.bucket')
  err=$(mktemp)
  if _r2 s3api put-object --bucket "$bucket" --key "$key" --body "$file" --if-none-match '*' >/dev/null 2>"$err"; then
    rm -f "$err"
    return 0
  fi
  if grep -q -E 'PreconditionFailed|\(412\)' "$err"; then
    rm -f "$err"
    return "$S3_PRECONDITION"
  fi
  rm -f "$err"
  return 1
}
