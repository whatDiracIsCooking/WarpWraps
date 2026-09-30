#!/usr/bin/env bash
# Regenerate every committed vendor manifest under vendor/ from the pinned SDK
# headers, one file per wrapped library. This is the SINGLE source of the
# harvest invocations: the initial commit and the CI regenerate-and-diff job
# (.github/workflows/ci.yml) both run this script, so a manifest can only drift
# if a header changed under a fixed pin -- which is exactly what CI catches.
#
# Each library is a row in one of the two tables below:
#
#     <out-basename> | <prefix> | <header-relative-to-include-root>
#
# The prefix is passed to devtools/vendor_harvest.py explicitly rather than
# auto-detected: many vendor headers are umbrellas that DECLARE nothing
# themselves (auto-detect sees no prefix), and several share a leading token
# (cublasLt / cublasXt both lead with `cublas`; cuComplex / the driver both lead
# with `cu`) that auto-detect resolves ambiguously. Pinning the prefix makes the
# harvested surface a stated choice, not a guess -- see vendor_harvest.py's
# header for what a manifest is and is not.
#
# INTENTIONALLY ABSENT (a manifest here would be noise, not a spec):
#   * The type-wrapper modules -- cuda_fp16/bf16/fp8/fp4/fp6, cuComplex, and
#     their HIP twins hip_fp16/bf16/fp8/fp4/fp6, hip_complex. Their surface is a
#     few `__half` / `__nv_*` / `__hip_*` struct types spelled with a leading
#     underscore, which leading_prefix() (a-z / A-Z runs only) cannot key on.
#   * cufile. Its functions are spelled `cuFile*`, whose leading token collapses
#     to `cu` -- and cufile.h transitively pulls in the whole CUDA driver, so
#     `cu` harvests all of cuda.h too. There is no single prefix that isolates
#     the ~40 cuFile entry points, so no meaningful manifest.
#
# KNOWN SUPERSETS (deterministic, still a valid diff artifact, just broad):
#   * cublasLt / cublasXt harvest with prefix `cublas` because their functions'
#     leading token is `cublas` (leading_prefix stops at the capital L/X). Both
#     headers transitively include the full base cublas_v2 API, so each manifest
#     is base-cublas + its own ~50 Lt/Xt entry points -- a superset of
#     cublas_v2.json, not just the variant surface.
#   * cusolverMg / cusolverSp harvest with prefix `cusolver` and likewise carry
#     the shared cusolver enums/types alongside their own functions.
# See the PR that introduced this script (#115).
#
# Usage:
#   devtools/vendor_manifests.sh cuda   # write vendor/cuda-13.0.x/*.json
#   devtools/vendor_manifests.sh hip    # write vendor/rocm-7.2.4/*.json
#   devtools/vendor_manifests.sh both   # both (needs the combined image)
#
# It writes into the tree under vendor/. To check for drift without touching the
# tree, CI runs it into a temp dir and diffs -- see the workflow.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARVEST="$HERE/devtools/vendor_harvest.py"
# The vendor headers under $AGG are repo-local, so they are named to the
# harvester by a repo-RELATIVE path (and the harvest runs from $HERE): the tool
# records that path verbatim in the manifest's "command" / "header", and an
# absolute one would bake in the checkout location -- differing between a dev
# tree and CI's /__w/... and making the diff job fail on the path alone. The SDK
# headers stay absolute; /opt/... and /usr/... are the same everywhere.
AGG="devtools/harvest_aggregates"
cd "$HERE"

# --- pins (the config tuple's SDK coordinate) --------------------------------
CUDA_SDK="CUDA 13.0.x"          # recorded in every CUDA manifest's config tuple
CUDA_DIR="cuda-13.0.x"          # vendor/<this>/
ROCM_SDK="ROCm 7.2.4"
ROCM_DIR="rocm-7.2.4"
ROCM_ARCH="gfx942"              # the arch GPU_TARGETS pins (docker/install-rocm-ds.sh)

CUDA_INC="${CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
# NCCL / cuTENSOR / nvCOMP arrive by apt into /usr/include, not the toolkit tree.
CUDA_EXTRA_INC="${CUDA_EXTRA_INCLUDE_DIR:-/usr/include}"
ROCM_INC="${ROCM_INCLUDE_DIR:-/opt/rocm/include}"
# hipCOMP is built from source into /opt/rocm-ds (docker/install-rocm-ds.sh).
ROCM_DS_INC="${ROCM_DS_INCLUDE_DIR:-/opt/rocm-ds/include}"

# --- the .so per surface (the linkable half of the manifest) ------------------
# Each run passes the shared object the wrapper links against so vendor_harvest
# can reconcile the declared surface against `nm -D --defined-only` and record
# both the linkable set and the declared-but-not-linkable diff (the thing the
# test suite's WWR_DECLARED_CHECK sites encode by hand). The manifest records the
# .so's SONAME (stable per pin), not this path, so it stays reproducible.
#
# WHERE THEY LIVE: CUDA toolkit libraries under $CUDA_LIB; the three apt/wheel
# libraries (cuTENSOR, nvCOMP) at the fixed /opt/nvidia symlinks install-cuda.sh
# publishes and NCCL at its Debian path; the driver (cuda.h) and NVML link
# against the TOOLKIT STUBS ($CUDA_STUB), whose exported names are exactly what a
# link resolves and which -- unlike the real driver .so -- ship with the toolkit
# and so are deterministic in the -ci image. HIP libraries under $ROCM_LIB, plus
# hipCOMP from the /opt/rocm-ds source build.
CUDA_LIB="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
CUDA_STUB="${CUDA_STUB_DIR:-$CUDA_LIB/stubs}"
CUTENSOR_LIB="${CUTENSOR_LIB_DIR:-/opt/nvidia/cutensor/lib}"
NVCOMP_LIB="${NVCOMP_LIB_DIR:-/opt/nvidia/nvcomp/lib}"
NCCL_LIB="${NCCL_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
ROCM_LIB="${ROCM_LIB_DIR:-/opt/rocm/lib}"
ROCM_DS_LIB="${ROCM_DS_LIB_DIR:-/opt/rocm-ds/lib}"

# run_cuda <out> <prefix> <header> <lib-arg...>
# The trailing lib-arg is EITHER `--lib <path>` or `--lib-none <reason>`; passed
# through verbatim so each row states the linkable source (or why there is none).
run_cuda() {
  local out="$1" prefix="$2" header="$3"; shift 3
  echo "  cuda: $out (prefix $prefix)"
  python3 "$HARVEST" --backend CUDA --sdk-version "$CUDA_SDK" \
    --prefix "$prefix" -I "$CUDA_INC" -I "$CUDA_EXTRA_INC" \
    "$@" "$header" -o "$OUTDIR/$out.json"
}

run_hip() {
  local out="$1" prefix="$2" header="$3"; shift 3
  echo "  hip: $out (prefix $prefix)"
  python3 "$HARVEST" --backend HIP --sdk-version "$ROCM_SDK" --arch "$ROCM_ARCH" \
    --prefix "$prefix" -D __HIP_PLATFORM_AMD__ \
    -I "$ROCM_INC" -I "$ROCM_DS_INC" \
    "$@" "$header" -o "$OUTDIR/$out.json"
}

harvest_cuda() {
  OUTDIR="${VENDOR_ROOT:-$HERE/vendor}/$CUDA_DIR"
  mkdir -p "$OUTDIR"
  echo "== CUDA ($CUDA_SDK) -> $OUTDIR =="
  # out-basename        prefix       header                       linkable source
  # cublasLt/cublasXt link libcublasLt/libcublas (their manifests are supersets
  # that also carry base cublas -- so the base symbols link against libcublas,
  # not the variant .so; the variant-own symbols land in the named .so). We name
  # the variant's own .so; base-cublas names that are not in it (there are none
  # in practice, they forward) would surface as declared_not_linkable if so.
  run_cuda cublas_v2     cublas       "$CUDA_INC/cublas_v2.h"    --lib "$CUDA_LIB/libcublas.so"
  run_cuda cublasLt      cublas       "$CUDA_INC/cublasLt.h"     --lib "$CUDA_LIB/libcublasLt.so"
  run_cuda cublasXt      cublas       "$CUDA_INC/cublasXt.h"     --lib "$CUDA_LIB/libcublas.so"
  run_cuda cufft         cufft        "$CUDA_INC/cufft.h"        --lib "$CUDA_LIB/libcufft.so"
  run_cuda cufftXt       cufft        "$CUDA_INC/cufftXt.h"      --lib "$CUDA_LIB/libcufft.so"
  run_cuda curand        curand       "$CUDA_INC/curand.h"       --lib "$CUDA_LIB/libcurand.so"
  # curand_kernel.h re-declares the host generator API (in libcurand) plus the
  # __device__ curand()/curand_uniform()/distribution templates and the
  # curandState*_t device RNG-state types, which live in no host .so; those
  # device-only functions correctly surface as declared_not_linkable. Mirrors
  # the HIP hiprand_kernel row -- src/cuda/curand.cppm's GMF includes both
  # <curand.h> and <curand_kernel.h>. Unlike hiprand_kernel.h it needs no
  # <cstdio> pre-include, so it harvests from the bare vendor header.
  run_cuda curand_kernel curand       "$CUDA_INC/curand_kernel.h" --lib "$CUDA_LIB/libcurand.so"
  run_cuda cusolverDn    cusolver     "$CUDA_INC/cusolverDn.h"   --lib "$CUDA_LIB/libcusolver.so"
  run_cuda cusolverMg    cusolver     "$CUDA_INC/cusolverMg.h"   --lib "$CUDA_LIB/libcusolverMg.so"
  run_cuda cusolverSp    cusolver     "$CUDA_INC/cusolverSp.h"   --lib "$CUDA_LIB/libcusolver.so"
  run_cuda cusparse      cusparse     "$CUDA_INC/cusparse.h"     --lib "$CUDA_LIB/libcusparse.so"
  run_cuda cutensor      cutensor     "$CUDA_EXTRA_INC/cutensor.h" --lib "$CUTENSOR_LIB/libcutensor.so"
  run_cuda nccl          nccl         "$CUDA_EXTRA_INC/nccl.h"   --lib "$NCCL_LIB/libnccl.so"
  run_cuda nvFatbin      nv           "$CUDA_INC/nvFatbin.h"     --lib "$CUDA_LIB/libnvfatbin.so"
  run_cuda nvJitLink     nv           "$CUDA_INC/nvJitLink.h"    --lib "$CUDA_LIB/libnvJitLink.so"
  # nvtx3 is header-only: its entry points are static-inline, no .so exports them.
  run_cuda nvToolsExt    nvtx         "$CUDA_INC/nvtx3/nvToolsExt.h" --lib-none "nvtx3 is header-only (static-inline entry points); no shared library exports these names"
  run_cuda nvjpeg        nvjpeg       "$CUDA_INC/nvjpeg.h"       --lib "$CUDA_LIB/libnvjpeg.so"
  # NVML and the driver link against the toolkit STUBS: the real libnvidia-ml /
  # libcuda are driver artifacts whose version tracks the host GPU driver, not
  # the pinned toolkit, so they are not reproducible in the -ci image; the stubs
  # (shipped with the toolkit) export the same names a link resolves.
  run_cuda nvml          nvml         "$CUDA_INC/nvml.h"         --lib "$CUDA_STUB/libnvidia-ml.so"
  run_cuda nvrtc         nvrtc        "$CUDA_INC/nvrtc.h"        --lib "$CUDA_LIB/libnvrtc.so"
  run_cuda cuda          cu           "$CUDA_INC/cuda.h"         --lib "$CUDA_STUB/libcuda.so"
  run_cuda cuda_runtime_api cuda      "$CUDA_INC/cuda_runtime_api.h" --lib "$CUDA_LIB/libcudart.so"
  run_cuda cupti         cupti        "$CUDA_INC/cupti.h"        --lib "$CUDA_LIB/libcupti.so"
  run_cuda nvcomp        nvcomp       "$AGG/nvcomp.h"            --lib "$NVCOMP_LIB/libnvcomp.so"
}

harvest_hip() {
  OUTDIR="${VENDOR_ROOT:-$HERE/vendor}/$ROCM_DIR"
  mkdir -p "$OUTDIR"
  echo "== HIP ($ROCM_SDK) -> $OUTDIR =="
  # out-basename         prefix      header                              linkable source
  run_hip hip_runtime_api hip        "$ROCM_INC/hip/hip_runtime_api.h"   --lib "$ROCM_LIB/libamdhip64.so"
  run_hip hipblas         hipblas    "$ROCM_INC/hipblas/hipblas.h"       --lib "$ROCM_LIB/libhipblas.so"
  run_hip hipblaslt       hipblas    "$ROCM_INC/hipblaslt/hipblaslt.h"   --lib "$ROCM_LIB/libhipblaslt.so"
  run_hip hipsolver       hipsolver  "$ROCM_INC/hipsolver/hipsolver.h"   --lib "$ROCM_LIB/libhipsolver.so"
  run_hip hipsparse       hipsparse  "$ROCM_INC/hipsparse/hipsparse.h"   --lib "$ROCM_LIB/libhipsparse.so"
  run_hip hipfft          hipfft     "$ROCM_INC/hipfft/hipfft.h"         --lib "$ROCM_LIB/libhipfft.so"
  run_hip hipfftXt        hipfft     "$ROCM_INC/hipfft/hipfftXt.h"       --lib "$ROCM_LIB/libhipfft.so"
  run_hip hiprand         hiprand    "$ROCM_INC/hiprand/hiprand.h"       --lib "$ROCM_LIB/libhiprand.so"
  # hiprand_kernel.h re-declares the host generator API (in libhiprand) plus the
  # __device__ hiprand()/hiprand4()/distribution templates, which live in no host
  # .so; those device-only names correctly surface as declared_not_linkable.
  run_hip hiprand_kernel  hiprand    "$AGG/hiprand_kernel.h"             --lib "$ROCM_LIB/libhiprand.so"
  run_hip hiprtc          hiprtc     "$ROCM_INC/hip/hiprtc.h"            --lib "$ROCM_LIB/libhiprtc.so"
  run_hip hiptensor       hiptensor  "$ROCM_INC/hiptensor/hiptensor.h"   --lib "$ROCM_LIB/libhiptensor.so"
  run_hip rccl            nccl       "$ROCM_INC/rccl/rccl.h"             --lib "$ROCM_LIB/librccl.so"
  run_hip hipcomp         hipcomp    "$AGG/hipcomp.h"                    --lib "$ROCM_DS_LIB/libhipcomp.so"
  run_hip rocm_smi        rsmi       "$ROCM_INC/rocm_smi/rocm_smi.h"     --lib "$ROCM_LIB/librocm_smi64.so"
  run_hip amd_smi         amdsmi     "$ROCM_INC/amd_smi/amdsmi.h"        --lib "$ROCM_LIB/libamd_smi.so"
  run_hip roctracer       roctracer  "$ROCM_INC/roctracer/roctracer.h"   --lib "$ROCM_LIB/libroctracer64.so"
  run_hip roctx           roctx      "$ROCM_INC/roctracer/roctx.h"       --lib "$ROCM_LIB/libroctx64.so"
}

case "${1:-}" in
  cuda) harvest_cuda ;;
  hip)  harvest_hip ;;
  both) harvest_cuda; harvest_hip ;;
  *) echo "usage: $0 <cuda|hip|both>" >&2; exit 2 ;;
esac
