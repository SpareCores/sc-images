#!/bin/bash

BINARY_PATH="/app/llama-bench"
LIBS_DIR="/app/libs"
mkdir -p "$LIBS_DIR"

collect_libs() {
    local BINARY="$1"
    local BINARY_DIR=$(dirname "$BINARY")
    # look up all linked libs
    local LIBS=$(ldd "$BINARY" | awk '{
        # Check if the line has a path (e.g., libxyz.so => /path/to/libxyz.so)
        if ($2 == "=>") {
            print $3  # Full path
        } else if ($1 !~ /^\//) {
            print $1  # Library name (without path)
        }
    }')
    for LIB in $LIBS; do
        # skip virtual files
        if [[ "$LIB" == "linux-vdso.so.1" ]]; then
            continue
        fi
        # look up symlink reference (if any)
        REAL_LIB=$(readlink -f "$LIB" || echo "$LIB")
        # skip already handled files and symlinks
        if [[ -n "$REAL_LIB" && ! -f "$LIBS_DIR/$(basename "$REAL_LIB")" && ! -f "$LIBS_DIR/$(basename "$LIB")" ]]; then
            echo "Extracting $(basename $LIB) ..."
            cp "$REAL_LIB" "$LIBS_DIR/$(basename $LIB)"
            if [[ "$LIB" != /* ]]; then
                REAL_LIB="$BINARY_DIR/$LIB"
            fi
            collect_libs "$REAL_LIB"
        fi
    done
}

collect_libs "$BINARY_PATH"
