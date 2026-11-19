#!/usr/bin/env bats
# The vendored copy of sc-infra's interlock files must match the hashes recorded next to it.

@test "lib/shared matches SHA256SUMS" {
  cd "$BATS_TEST_DIRNAME/../../lib/shared"
  sha256sum -c SHA256SUMS
}
