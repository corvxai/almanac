#!/usr/bin/env sh
# Build arcratio Docker images for an explicit target CPU architecture.
#
# WHY THIS EXISTS
# ---------------
# The validator spawns the `agent-runner` image as *sibling containers* at
# runtime (via /var/run/docker.sock). A container image is architecture-
# specific: an image built on an Apple Silicon (arm64) laptop will NOT run on
# an amd64 Linux server, and vice versa. If the agent-runner image arch does
# not match the host the validator runs on, every agent execution fails to
# start. So always build agent-runner for the architecture of the machine that
# will RUN the validator — not necessarily the machine you build on.
#
# This wrapper makes the target platform explicit and warns when you are
# cross-building (emulated via QEMU/buildx), which is the silent footgun.
#
# USAGE
# -----
#   scripts/build_images.sh                      # agent-runner, linux/amd64 (default)
#   PLATFORM=linux/arm64 scripts/build_images.sh # build for arm64 hosts
#   scripts/build_images.sh validator            # build the validator image
#   scripts/build_images.sh all                  # build both
#
# Windows users: use scripts/build_images.ps1 (PowerShell) instead.
set -eu

PLATFORM="${PLATFORM:-linux/amd64}"
TARGET="${1:-agent-runner}"

host_arch="$(uname -m 2>/dev/null || echo unknown)"
case "$host_arch" in
  x86_64|amd64)   host_plat="linux/amd64" ;;
  aarch64|arm64)  host_plat="linux/arm64" ;;
  *)              host_plat="unknown($host_arch)" ;;
esac

echo "Host arch            : $host_arch (native: $host_plat)"
echo "Target build platform: $PLATFORM"

if [ "$PLATFORM" != "$host_plat" ]; then
  echo "WARNING: cross-building for $PLATFORM on a $host_plat host."
  echo "         This uses QEMU emulation (slower) and needs buildx + binfmt."
  echo "         Docker Desktop bundles these. On a bare Linux host run once:"
  echo "           docker run --privileged --rm tonistiigi/binfmt --install all"
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "ERROR: 'docker buildx' is required (Docker 19.03+ / Docker Desktop)." >&2
  exit 1
fi

build() {
  _dockerfile="$1"; _tag="$2"
  echo ">>> Building $_tag from $_dockerfile for $PLATFORM"
  # --load imports the result into the local daemon so the validator can find
  # the tag. --load supports exactly one --platform, which is why PLATFORM is
  # singular here.
  docker buildx build \
    --platform "$PLATFORM" \
    --file "$_dockerfile" \
    --tag "$_tag" \
    --load \
    .
}

case "$TARGET" in
  agent-runner) build docker/agent-runner.Dockerfile arcratio/agent-runner:latest ;;
  validator)    build docker/validator.Dockerfile    arcratio/validator:latest ;;
  all)
    build docker/agent-runner.Dockerfile arcratio/agent-runner:latest
    build docker/validator.Dockerfile    arcratio/validator:latest
    ;;
  *)
    echo "ERROR: unknown target '$TARGET' (use: agent-runner | validator | all)" >&2
    exit 2
    ;;
esac

echo "Done. Built $TARGET for $PLATFORM."
