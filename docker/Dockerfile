# syntax=docker/dockerfile:1.7
# ============================================================================
# The GPU image, and the only one that can build the C++ side of this repo.
# ============================================================================
# One file, four selectable targets. `base` is vendor-neutral -- the C++23
# toolchain and nothing else -- and each GPU target adds one SDK on top:
#
#   docker build --target base     -f docker/Dockerfile -t gpumod:base .
#   docker build --target cuda     -f docker/Dockerfile -t gpumod:latest .
#   docker build --target hip      -f docker/Dockerfile -t gpumod:hip .
#   docker build --target combined -f docker/Dockerfile -t gpumod:combined .
#
# RUN THOSE FROM THE REPO ROOT. The trailing `.` is the build context and it is
# the repo root, not this directory: the stages COPY pyproject.toml and uv.lock
# and bind-mount docker/install-{cuda,rocm}.sh, every one of those paths spelled
# relative to that root. `docker build -f Dockerfile .` from inside docker/ dies
# on the first COPY.
#
# `cuda` is the default target (it is last in file order only by accident of
# `combined` inheriting it, so ALWAYS pass --target explicitly) and is what
# .devcontainer/cuda/devcontainer.json and docker/compose.yaml build.
#
# WHY THE TOOLKIT COMES FROM APT, NOT FROM nvidia/cuda:
# a shared toolchain `base` cannot be both nvidia/cuda and rocm/dev-ubuntu at
# once, and duplicating the LLVM/CMake stages per vendor is how they drift. The
# nvidia/cuda images are Ubuntu plus the same apt packages, so the only thing
# lost is the NVIDIA_* runtime env they set -- which the `cuda` stage sets back
# explicitly. ROCm is ~30GB installed, which is the whole reason `hip` is a
# separate target rather than everything living in one image.
#
# Pick CUDA_VERSION against the HOST DRIVER, not against the newest CUDA: a
# driver whose ceiling is below the image's CUDA version makes the container
# refuse to start under `--gpus all` with
#   nvidia-container-cli: requirement error: unsatisfied condition: cuda>=X.Y
# `nvidia-smi` reports the driver's ceiling as "CUDA Version:". Note also that
# CUDA 13 dropped Maxwell/Pascal/Volta -- its nvcc floor is compute_75, so a
# pre-Turing card needs a 12.x toolkit, which also means dropping the `volta`
# preset from CMakePresets.json.
#
# Build one stage in isolation to debug it:
#   docker build --target=compilers -f docker/Dockerfile -t dbg:compilers .
#
# ---------------------------------------------------------------------------
# WHY THE VERSIONS BELOW ARE PINNED HERE AND NOT DISCOVERED
# ---------------------------------------------------------------------------
# `import std` needs three things to agree: clang, libc++'s module manifest,
# and CMake's experimental import-std UUID (set in CMakeLists.txt, and specific
# to the CMake version). Floating any one of them turns a working build into
# "module 'std' not found" with no other symptom. Bump them together, and
# re-check the UUID in the new CMake's Help/dev/experimental.rst.
ARG UBUNTU_TAG=24.04
ARG LLVM_VERSION=20
ARG CMAKE_VERSION=4.2.3
ARG CMAKE_MAJOR_MINOR=4.2
ARG NINJA_VERSION=1.13.1
ARG CUDA_VERSION=13-0
ARG ROCM_VERSION=7.2.4

# ============================================================================
# Stage 1: base-system -- OS packages every later stage shares
# ============================================================================
FROM ubuntu:${UBUNTU_TAG} AS base-system

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates ccache curl git gnupg jq less \
      lcov patch pkg-config shellcheck software-properties-common ssh tree \
      unzip valgrind vim-tiny wget zlib1g-dev \
    && git --version && wget --version | head -1

# ============================================================================
# Stage 2: downloads -- CMake and Ninja tarballs (independent of base-system,
# so BuildKit runs it concurrently)
# ============================================================================
FROM ubuntu:${UBUNTU_TAG} AS downloads

ARG CMAKE_VERSION
ARG NINJA_VERSION
# Supplied automatically by BuildKit: "amd64" or "arm64".
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upstream asset names differ per architecture, so they are resolved here and
# staged under arch-independent filenames for the next stage to consume.
RUN --mount=type=cache,target=/download-cache \
    set -e; \
    case "$TARGETARCH" in \
      amd64) cmake_arch=x86_64;  ninja_asset=ninja-linux.zip ;; \
      arm64) cmake_arch=aarch64; ninja_asset=ninja-linux-aarch64.zip ;; \
      *) echo "unsupported TARGETARCH: '$TARGETARCH'" >&2; exit 1 ;; \
    esac; \
    cmake_asset="cmake-${CMAKE_VERSION}-linux-${cmake_arch}.sh"; \
    mkdir -p /downloads; \
    [ -f "/download-cache/${cmake_asset}" ] || \
      wget -q -O "/download-cache/${cmake_asset}" \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${cmake_asset}"; \
    [ -f "/download-cache/ninja-${NINJA_VERSION}-${ninja_asset}" ] || \
      wget -q -O "/download-cache/ninja-${NINJA_VERSION}-${ninja_asset}" \
        "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/${ninja_asset}"; \
    cp "/download-cache/${cmake_asset}" /downloads/cmake-installer.sh; \
    cp "/download-cache/ninja-${NINJA_VERSION}-${ninja_asset}" /downloads/ninja.zip; \
    chmod +x /downloads/cmake-installer.sh

# ============================================================================
# Stage 3: compilers -- LLVM/clang with libc++ and its module manifest
# ============================================================================
FROM base-system AS compilers

ARG LLVM_VERSION

# libc++-dev is the load-bearing one: it ships libc++.modules.json, which is
# what CMakeLists.txt globs /usr/lib/llvm-*/lib for to resolve `import std`.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    wget -qO /etc/apt/trusted.gpg.d/llvm-snapshot.asc https://apt.llvm.org/llvm-snapshot.gpg.key && \
    add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" && \
    apt-get update && apt-get install -y --no-install-recommends --fix-missing \
      llvm-${LLVM_VERSION} \
      clang-${LLVM_VERSION} \
      clang-tidy-${LLVM_VERSION} \
      clang-format-${LLVM_VERSION} \
      clangd-${LLVM_VERSION} \
      clang-tools-${LLVM_VERSION} \
      libclang-rt-${LLVM_VERSION}-dev \
      libc++-${LLVM_VERSION}-dev \
      libc++abi-${LLVM_VERSION}-dev \
      cppcheck

# The versioned binaries live in /usr/lib/llvm-N/bin, which is not on PATH.
# clang-scan-deps is included: CMake's module scanner, and a FATAL_ERROR in
# CMakeLists.txt when it is missing. llvm-objdump and llvm-cxxfilt are here for
# test/extension/build_time's blas dispatch check, which disassembles the
# instantiations and demangles what they call.
RUN for tool in clang clang++ clang-tidy clang-format clangd clang-scan-deps \
                llvm-objdump llvm-cxxfilt; do \
      update-alternatives --install /usr/bin/$tool $tool /usr/bin/${tool}-${LLVM_VERSION} 100; \
    done && \
    clang --version | grep -q "${LLVM_VERSION}\." && \
    clang-scan-deps --version && \
    llvm-objdump --version | head -1 && \
    test -f /usr/lib/llvm-${LLVM_VERSION}/lib/libc++.modules.json

# ============================================================================
# Stage 4: build-tools -- CMake and Ninja onto a base-system layer
# ============================================================================
FROM base-system AS build-tools

ARG CMAKE_VERSION

COPY --link --from=downloads /downloads/cmake-installer.sh /tmp/
COPY --link --from=downloads /downloads/ninja.zip /tmp/

RUN cd /tmp && ./cmake-installer.sh --skip-license --prefix=/usr/local && rm cmake-installer.sh && \
    unzip -q ninja.zip && mv ninja /usr/local/bin/ && chmod +x /usr/local/bin/ninja && rm ninja.zip && \
    cmake --version | grep -q "${CMAKE_VERSION}" && ninja --version

# ============================================================================
# Stage 5: base -- the complete, vendor-neutral toolchain
# ============================================================================
# Everything the build needs EXCEPT a GPU SDK. Selectable on its own
# (`--target base`) to check the toolchain half in isolation, and the shared
# parent of `cuda` and `hip` so the two cannot drift apart.
#
# Note there are no prebuilt C++ dependencies here any more: the project fetches
# GoogleTest from source at configure time and needs nothing else, so the /opt
# trees the template used to bake are gone along with the demo code that used
# them.
FROM compilers AS base

ARG CMAKE_MAJOR_MINOR
ARG LLVM_VERSION

# This label is what scopes `devtools/worktree.sh gc` to this project's build
# images, so it cannot delete another repo's. Must match PROJECT_NAME in
# devtools/config.sh -- devtools/doctor.sh warns when they drift.
ARG PROJECT_NAME=gpumod
LABEL devcontainer.project="${PROJECT_NAME}"

COPY --link --from=build-tools /usr/local/bin/cmake /usr/local/bin/cmake
COPY --link --from=build-tools /usr/local/bin/ctest /usr/local/bin/ctest
COPY --link --from=build-tools /usr/local/bin/cpack /usr/local/bin/cpack
COPY --link --from=build-tools /usr/local/bin/ninja /usr/local/bin/ninja
# CMake installs to share/cmake-<MAJOR>.<MINOR>, not -<MAJOR>.<MINOR>.<PATCH>.
COPY --link --from=build-tools /usr/local/share/cmake-${CMAKE_MAJOR_MINOR} /usr/local/share/cmake-${CMAKE_MAJOR_MINOR}

# --- Python -----------------------------------------------------------------
# One uv-managed venv at /opt/venv, installed `--frozen` from the committed
# lock, so the container and the host cannot drift. This is why there is no pip install list in here -- add a dependency
# to pyproject.toml, run `uv lock`, commit both.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock /tmp/uvroot/
RUN cd /tmp/uvroot && uv sync --frozen --no-install-project && rm -rf /tmp/uvroot && \
    chown -R 1000:1000 /opt/venv

# Ubuntu 24.04 ships an `ubuntu` user at uid 1000, which matches a typical
# single-user host, so files written into the bind mount stay owned by you.
# .devcontainer/*.json set `remoteUser: ubuntu`; docker/compose.yaml instead
# passes HOST_UID:HOST_GID, which is why /opt/venv is group-readable rather
# than owned by a user this image had to invent.
RUN install -d -o 1000 -g 1000 \
        /workspace \
        /tmp/pytest-of-ubuntu \
        /home/ubuntu/.ccache \
        /home/ubuntu/.claude/plugins \
        /home/ubuntu/.claude/backups

ENV CCACHE_DIR=/home/ubuntu/.ccache \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH}

WORKDIR /workspace

RUN clang --version | head -1 && \
    cmake --version | head -1 && \
    ninja --version && \
    python3 --version && \
    test -f /usr/lib/llvm-${LLVM_VERSION}/lib/libc++.modules.json

USER ubuntu
CMD ["/bin/bash"]

# ============================================================================
# Stage 6: cuda -- base + the CUDA toolkit (the default backend)
# ============================================================================
FROM base AS cuda

ARG CUDA_VERSION
ARG TARGETARCH
USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-cuda.sh,target=/tmp/install-cuda.sh \
    CUDA_VERSION="${CUDA_VERSION}" TARGETARCH="${TARGETARCH}" bash /tmp/install-cuda.sh

# Set by the nvidia/cuda base images, and required by the NVIDIA container
# runtime to inject the driver under `--gpus all`. Building on plain Ubuntu
# means setting them here instead.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Version-independent symlink, so a toolkit bump (13.0 -> 13.1) does not leave
# these pointing at a directory that no longer exists.
ENV CUDAToolkit_ROOT=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# The target architecture, as a build arg so a cluster build can widen it:
#   --build-arg CUDA_ARCH="80;86;90"
# 86 is Ampere consumer (RTX 30xx). CUDAARCHS is what CMake reads when a preset
# does not set CMAKE_CUDA_ARCHITECTURES; the `default` preset sets it to
# `native`, which compiles only for the card in the build machine -- use a
# pinned or `portable` preset when the build host and run host differ.
ARG CUDA_ARCH=86
ENV CUDA_ARCH=${CUDA_ARCH} \
    CUDAARCHS=${CUDA_ARCH}

# Nsight Systems CLI, for profiling from inside the container. Non-fatal: the
# repository is occasionally unavailable, and nothing else depends on nsys.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    repo="https://developer.download.nvidia.com/devtools/repos/ubuntu2404/${TARGETARCH}" && \
    ( wget -qO /etc/apt/trusted.gpg.d/nvidia-devtools.asc "${repo}/nvidia.pub" && \
      echo "deb ${repo}/ /" > /etc/apt/sources.list.d/nvidia-devtools.list && \
      apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli ) \
    || echo "WARN: nsight-systems-cli unavailable; nsys will be missing (doctor.sh reports it)"

USER ubuntu
RUN nvcc --version | grep release

# ============================================================================
# Stage 7: hip -- base + ROCm, with no CUDA toolkit at all
# ============================================================================
# A HIP build needs no CUDA, which is the point of splitting this from `cuda`:
# ROCm alone is ~30GB, and neither backend should pay for the other.
FROM base AS hip

ARG ROCM_VERSION
USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" bash /tmp/install-rocm.sh

# The LLVM target for the AMD card this image is built for. Exported as
# GPU_TARGETS / AMDGPU_TARGETS so a HIP CMake build has a default to read.
# gfx1200 is RDNA4 (Radeon RX 9060 XT); override for another card:
#   --build-arg GPU_TARGETS=gfx1100
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}

USER ubuntu
RUN hipconfig --version

# ============================================================================
# Stage 8: combined -- both SDKs in one image
# ============================================================================
# For switching backends without switching containers. It is the sum of the two
# above (~40GB); prefer `cuda` or `hip` unless you actually need both.
FROM cuda AS combined

ARG ROCM_VERSION
USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" bash /tmp/install-rocm.sh

ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}

USER ubuntu
RUN nvcc --version | grep release && hipconfig --version
