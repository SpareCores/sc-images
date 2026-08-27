#!/bin/sh

# Our custom-built OpenSSL (see Dockerfile) lives in /usr/local/lib; not the
# image-wide default so it doesn't shadow the system libssl python3 needs.
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64"

CMD="nice -n -20 openssl speed -mr -mlock -elapsed -multi $(nproc)"

# run each algos in a separate run, so we can parse the output easier, by matching +DT/+DTP and +F
# lines together
for algo in \
  "-evp blake2b512" \
  "-evp sha256" \
  "-evp sha512" \
  "-evp sha3-256" \
  "-evp sha3-512" \
  "-evp shake128" \
  "-evp shake256" \
  "-evp aes-256-cbc" \
  "-evp aria-256-cbc" \
  "-evp camellia-256-cbc" \
  "-evp sm4-cbc"
do
  # print a delimiter, so we can break up the output
  echo "ALGO: $algo ----------------------------------------"
  ${CMD} $algo 2>&1
done

