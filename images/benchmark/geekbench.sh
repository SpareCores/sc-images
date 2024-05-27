#!/bin/sh

# decrypt secrets
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 openssl aes-256-cbc -d -pass env:BENCHMARK_SECRETS_PASSPHRASE \
  -pbkdf2 -iter 100000 -in /secrets.enc -out - | tar zxpf - -C /

# write the JSON output to stderr, so we can gather it separately from the stdout
/usr/local/geekbench-$(uname -m)/geekbench6 --export-json /dev/stderr --no-upload