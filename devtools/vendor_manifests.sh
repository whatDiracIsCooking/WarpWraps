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
AGG="$HERE/devtools/harvest_aggregates"

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

run_cuda() {
  local out="$1" prefix="$2" header="$3"
  echo "  cuda: $out (prefix $prefix)"
  python3 "$HARVEST" --backend CUDA --sdk-version "$CUDA_SDK" \
    --prefix "$prefix" -I "$CUDA_INC" -I "$CUDA_EXTRA_INC" \
    "$header" -o "$OUTDIR/$out.json"
}

run_hip() {
  local out="$1" prefix="$2" header="$3"
  echo "  hip: $out (prefix $prefix)"
  python3 "$HARVEST" --backend HIP --sdk-version "$ROCM_SDK" --arch "$ROCM_ARCH" \
    --prefix "$prefix" -D __HIP_PLATFORM_AMD__ \
    -I "$ROCM_INC" -I "$ROCM_DS_INC" \
    "$header" -o "$OUTDIR/$out.json"
}

harvest_cuda() {
  OUTDIR="${VENDOR_ROOT:-$HERE/vendor}/$CUDA_DIR"
  mkdir -p "$OUTDIR"
  echo "== CUDA ($CUDA_SDK) -> $OUTDIR =="
  # out-basename        prefix       header
  run_cuda cublas_v2     cublas       "$CUDA_INC/cublas_v2.h"
  run_cuda cublasLt      cublas       "$CUDA_INC/cublasLt.h"
  run_cuda cublasXt      cublas       "$CUDA_INC/cublasXt.h"
  run_cuda cufft         cufft        "$CUDA_INC/cufft.h"
  run_cuda cufftXt       cufft        "$CUDA_INC/cufftXt.h"
  run_cuda curand        curand       "$CUDA_INC/curand.h"
  run_cuda cusolverDn    cusolver     "$CUDA_INC/cusolverDn.h"
  run_cuda cusolverMg    cusolver     "$CUDA_INC/cusolverMg.h"
  run_cuda cusolverSp    cusolver     "$CUDA_INC/cusolverSp.h"
  run_cuda cusparse      cusparse     "$CUDA_INC/cusparse.h"
  run_cuda cutensor      cutensor     "$CUDA_EXTRA_INC/cutensor.h"
  run_cuda nccl          nccl         "$CUDA_EXTRA_INC/nccl.h"
  run_cuda nvFatbin      nv           "$CUDA_INC/nvFatbin.h"
  run_cuda nvJitLink     nv           "$CUDA_INC/nvJitLink.h"
  run_cuda nvToolsExt    nvtx         "$CUDA_INC/nvtx3/nvToolsExt.h"
  run_cuda nvjpeg        nvjpeg       "$CUDA_INC/nvjpeg.h"
  run_cuda nvml          nvml         "$CUDA_INC/nvml.h"
  run_cuda nvrtc         nvrtc        "$CUDA_INC/nvrtc.h"
  run_cuda cuda          cu           "$CUDA_INC/cuda.h"
  run_cuda cuda_runtime_api cuda      "$CUDA_INC/cuda_runtime_api.h"
  run_cuda cupti         cupti        "$CUDA_INC/cupti.h"
  run_cuda nvcomp        nvcomp       "$AGG/nvcomp.h"
}

harvest_hip() {
  OUTDIR="${VENDOR_ROOT:-$HERE/vendor}/$ROCM_DIR"
  mkdir -p "$OUTDIR"
  echo "== HIP ($ROCM_SDK) -> $OUTDIR =="
  run_hip hip_runtime_api hip        "$ROCM_INC/hip/hip_runtime_api.h"
  run_hip hipblas         hipblas    "$ROCM_INC/hipblas/hipblas.h"
  run_hip hipblaslt       hipblas    "$ROCM_INC/hipblaslt/hipblaslt.h"
  run_hip hipsolver       hipsolver  "$ROCM_INC/hipsolver/hipsolver.h"
  run_hip hipsparse       hipsparse  "$ROCM_INC/hipsparse/hipsparse.h"
  run_hip hipfft          hipfft     "$ROCM_INC/hipfft/hipfft.h"
  run_hip hipfftXt        hipfft     "$ROCM_INC/hipfft/hipfftXt.h"
  run_hip hiprand         hiprand    "$ROCM_INC/hiprand/hiprand.h"
  run_hip hiprand_kernel  hiprand    "$AGG/hiprand_kernel.h"
  run_hip hiprtc          hiprtc     "$ROCM_INC/hip/hiprtc.h"
  run_hip hiptensor       hiptensor  "$ROCM_INC/hiptensor/hiptensor.h"
  run_hip rccl            nccl       "$ROCM_INC/rccl/rccl.h"
  run_hip hipcomp         hipcomp    "$AGG/hipcomp.h"
  run_hip rocm_smi        rsmi       "$ROCM_INC/rocm_smi/rocm_smi.h"
  run_hip amd_smi         amdsmi     "$ROCM_INC/amd_smi/amdsmi.h"
  run_hip roctracer       roctracer  "$ROCM_INC/roctracer/roctracer.h"
  run_hip roctx           roctx      "$ROCM_INC/roctracer/roctx.h"
}

case "${1:-}" in
  cuda) harvest_cuda ;;
  hip)  harvest_hip ;;
  both) harvest_cuda; harvest_hip ;;
  *) echo "usage: $0 <cuda|hip|both>" >&2; exit 2 ;;
esac
