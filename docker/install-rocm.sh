#!/usr/bin/env bash
# Install the ROCm/HIP SDK from AMD's apt repository.
#
# Called by docker/Dockerfile.hip, and again by docker/Dockerfile.combined on
# top of the CUDA image. A script rather than an inline RUN so the block is
# written once -- Dockerfiles have no include, and `combined` cannot inherit
# from two parents, so it reuses THIS instead of the hip image (see that file's
# header).
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION must be set, e.g. 7.2.4}"

# A signed-by keyring, not `apt-key add`: apt-key is deprecated in 24.04 and
# removed in 25.04.
#
# Only repo.radeon.com/rocm is added, NOT repo.radeon.com/amdgpu. That second
# repo exists to ship the DKMS kernel driver and AMD's libdrm fork, and neither
# belongs in a container -- the kernel driver comes from the host.
#
# The pin is load-bearing, not decoration. repo.radeon.com republishes its own
# builds of packages that also exist in Ubuntu; at equal priority apt is free to
# prefer either, and which one it picks can change between base-image refreshes.
# Priority 600 makes it deterministic -- ROCm's repo wins for anything it
# publishes. `o=repo.radeon.com` is the Origin its Release file declares.
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
chmod 0644 /etc/apt/keyrings/rocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  > /etc/apt/preferences.d/rocm-pin-600

# rocm-hip-sdk is AMD's meta-package for "the HIP SDK" -- one name that tracks
# whatever AMD decides belongs in it (hipBLAS, hipSOLVER, hipFFT, hipRAND,
# hipSPARSE, rocm-smi, hipcc, rocm-cmake and the rest of the surface src/hip
# wraps).
#
# The four after it are the ones it does NOT pull, checked against its resolved
# dependency closure rather than assumed:
#   amd-smi-lib     the modern half of the nvml analogue (rocm-smi-lib, the
#                   older half, DOES come with the SDK via rocm-hip-libraries)
#   roctracer-dev   the cupti analogue
#   rocprofiler-sdk rocprofv3, its successor
#   rocm-gdb        parity with cuda-gdb
apt-get update
apt-get install -y --no-install-recommends \
  rocm-hip-sdk \
  amd-smi-lib \
  roctracer-dev \
  rocprofiler-sdk \
  rocm-gdb

# ROCm unpacks to /opt/rocm-<version> and rocm-core provides the /opt/rocm
# symlink. Nothing in the deb set registers that prefix with the dynamic loader,
# so without this every hipcc-linked binary would need an explicit RPATH.
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
ldconfig
test -L /opt/rocm

# Device access. /dev/kfd (the compute node) and /dev/dri/renderD* are
# root:render; /dev/dri/card* is root:video. The groups have to exist INSIDE the
# container for a non-root user to open those nodes once they are passed in, and
# `render` is not present in the stock Ubuntu image.
#
# gid 110 is Ubuntu 24.04's default for render, and the gid HERE is what a
# `--group-add render` by NAME would resolve to -- which is why nothing passes
# the name: the bind-mounted nodes keep the HOST's gid. Pass `--group-add
# <host-gid>` to docker run instead, which is what the devcontainer runArgs and
# cpp-tier.sh --rocm both do. Never rebuild the image to match a host.
(getent group render >/dev/null || groupadd -g 110 render || groupadd render)
(getent group video >/dev/null || groupadd video)
usermod -aG render,video ubuntu
