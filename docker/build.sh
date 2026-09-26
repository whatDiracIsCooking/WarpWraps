#!/usr/bin/env bash
# Build one of docker/Dockerfile.{base,cuda,hip,combined}, and everything it
# inherits from, in order.
#
#   docker/build.sh base          <PROJECT_NAME>:base
#   docker/build.sh cuda          base, then cuda   -> :cuda and :latest
#   docker/build.sh hip           base, then hip    -> :hip
#   docker/build.sh combined      base, cuda, then combined -> :combined
#
# Anything after the target is passed through to EVERY `docker build` in the
# chain, which is what you want for --no-cache, --progress=plain or --pull:
#
#   docker/build.sh cuda --no-cache
#
# Run it from anywhere; the build context is always the repo root, because the
# Dockerfiles spell their COPY and bind-mount paths relative to it.
#
# WHY THIS SCRIPT EXISTS
# The four Dockerfiles chain by TAG, not by stage (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"): a child's `FROM ${PARENT_IMAGE}` needs its parent
# already built and tagged, so something has to walk the chain in order. This is
# that something -- the single-Dockerfile `docker build --target cuda` that used
# to do it does not exist any more.
#
# It deliberately does NOT skip a step whose tag already exists. A cached
# `docker build` of an unchanged image is a second or two, and skipping on tag
# presence would silently hand a child a STALE parent after an edit to
# Dockerfile.base -- exactly the failure the single file could not have.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# PROJECT_NAME, and nothing else read here. Sourcing it is what keeps the image
# names in one place: devtools/config.sh already has to agree with the
# .devcontainer/*.json files (doctor.sh checks that), and a third spelling in
# here would be a third thing to drift.
# shellcheck source=../devtools/config.sh
. "$REPO_ROOT/devtools/config.sh"

export DOCKER_BUILDKIT=1

die() { echo "build.sh: $*" >&2; exit 1; }

TARGETS="base cuda hip combined"

target=${1:-}
[ -n "$target" ] || die "usage: docker/build.sh <${TARGETS// /|}> [docker build flags...]"
shift
# shellcheck disable=SC2076  # the literal spaces are the point: whole-word match
[[ " $TARGETS " == *" $target "* ]] || die "unknown target '$target' (want: $TARGETS)"

# --- the diamond, as three tables -------------------------------------------
# Kept as case statements rather than associative arrays so the file stays
# readable as the graph it describes.

parent_of() {
  case $1 in
    base)     echo "" ;;
    cuda|hip) echo "base" ;;
    combined) echo "cuda" ;;
  esac
}

# The tags each image gets. `cuda` gets :latest as well, because that is what
# docker/compose.yaml and README.md name (GPUMOD_IMAGE defaults to it) -- the
# CUDA image is the default backend.
tags_of() {
  case $1 in
    base)     echo "$PROJECT_NAME:base" ;;
    cuda)     echo "$PROJECT_NAME:cuda $PROJECT_NAME:latest" ;;
    hip)      echo "$PROJECT_NAME:hip" ;;
    combined) echo "$PROJECT_NAME:combined" ;;
  esac
}

# Which build args each Dockerfile declares. Only names set in the environment
# are forwarded, so the DEFAULTS STAY IN THE DOCKERFILES and this script holds
# no version numbers of its own. Forwarding an arg to a file that does not
# declare it would earn a "build-args were not consumed" warning on every run,
# which is why these are per-file rather than one list.
build_args_of() {
  case $1 in
    base)         echo "UBUNTU_TAG LLVM_VERSION CMAKE_VERSION CMAKE_MAJOR_MINOR NINJA_VERSION" ;;
    cuda)         echo "CUDA_VERSION CUDA_ARCH" ;;
    hip|combined) echo "ROCM_VERSION GPU_TARGETS" ;;
  esac
}

# --- resolve the chain, root first ------------------------------------------
chain=()
step=$target
while [ -n "$step" ]; do
  chain=("$step" "${chain[@]}")
  step=$(parent_of "$step")
done

echo "build.sh: $target <- ${chain[*]}  (project '$PROJECT_NAME', context $REPO_ROOT)"

for step in "${chain[@]}"; do
  read -r -a tags <<<"$(tags_of "$step")"
  cmd=(docker build -f "$REPO_ROOT/docker/Dockerfile.$step")
  for tag in "${tags[@]}"; do cmd+=(-t "$tag"); done

  # Always explicit, never left to the Dockerfile's default: a renamed project
  # must chain to ITS OWN parent, not to gpumod's.
  cmd+=(--build-arg "PROJECT_NAME=$PROJECT_NAME")
  parent=$(parent_of "$step")
  if [ -n "$parent" ]; then
    read -r -a parent_tags <<<"$(tags_of "$parent")"
    cmd+=(--build-arg "PARENT_IMAGE=${parent_tags[0]}")
  fi

  for name in $(build_args_of "$step"); do
    # Set-but-empty still counts: it is how you ask for an empty value.
    [ -n "${!name+set}" ] && cmd+=(--build-arg "$name=${!name}")
  done

  cmd+=("$@" "$REPO_ROOT")

  echo "build.sh: + ${cmd[*]}"
  [ -n "${BUILD_DRY_RUN:-}" ] || "${cmd[@]}"
done

echo "build.sh: done -- $(for step in "${chain[@]}"; do tags_of "$step"; done | tr '\n' ' ')"
