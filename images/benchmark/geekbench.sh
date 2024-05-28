#!/bin/sh

# decrypt secrets
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 openssl aes-256-cbc -d -pass env:BENCHMARK_SECRETS_PASSPHRASE \
  -pbkdf2 -iter 100000 -in /secrets.enc -out - | tar zxpf - -C /

arch="$(uname -m)"

if [ "${arch}" = aarch64 ]; then
  # http://support.primatelabs.com/discussions/geekbench/83083-geekbench-6-preview-license
  # preview doesn't support pro mode
  /usr/local/geekbench-${arch}/geekbench6
else
  # write the JSON output to stderr, so we can gather it separately from the stdout
  /usr/local/geekbench-${arch}/geekbench6 --export-json /dev/stderr --no-upload
fi