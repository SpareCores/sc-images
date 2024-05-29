#!/bin/sh

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
  "-evp sm4-cbc" \
  rsa2048 ECP-384 ed25519 X25519 X448
do
  ${CMD} $algo 2>&1
done

