#!/bin/sh

# Check if kvm-ok is installed
if ! command -v kvm-ok > /dev/null 2>&1; then
    echo '{"kvm":false,"kvm_error":"kvm-ok command not found"}'
    exit 1
fi

# Run kvm-ok and capture both stdout and stderr
output=$(kvm-ok 2>&1)
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo '{"kvm":true}'
else
    # Replace newlines with spaces and escape quotes
    sanitized_output=$(printf '%s' "$output" | tr '\n' ' ' | sed 's/"/\\"/g')
    echo "{\"kvm\":false,\"kvm_error\":\"$sanitized_output\"}"
fi