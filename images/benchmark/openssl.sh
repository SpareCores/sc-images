#!/bin/sh

CMD="nice -n -20 openssl speed -mr -mlock -elapsed -multi $(nproc)"

# https://github.com/openssl/openssl/issues/22545
echo blake2b512 sha256 sha512 \
        sha3-256 sha3-512 shake128 shake256 \
        aes-256-cbc aria-256-cbc \
        camellia-256-cbc sm4-cbc | \
xargs -n 1 ${CMD} -evp 2>&1

${CMD} rsa2048 ECP-384 X25519 X448 2>&1
