"""The `fft` (cuFFT / hipFFT) surface as a FRAGMENT-SOURCED spec.

The blas machine (spec/library.py) over detail/fft_names.h: plan create/destroy,
the plan-estimation / make-plan / get-size family, SetStream / SetWorkArea /
SetAutoAllocation, GetVersion, and the six Exec transforms. It REGISTERS the
vendor-free types fft forces:

  * SCALAR ALIASES  cufftHandle is `typedef int` (a plan id, NOT an opaque
    pointer handle), and cufftReal / cufftDoubleReal are `typedef float / double`.
    Folded to the UNDERLYING SCALAR across the ABI (cufftHandle -> int,
    cufftHandle* -> int*, cufftReal* -> float*) via Registry.scalar_aliases, not
    erased to void*.
  * POINTEE PODS  cufftComplex (= cuComplex) -> wwr_complex64*, cufftDoubleComplex
    -> wwr_complex128* (the shared common.h PODs)
  * VALUE ENUMS  Result (the status) and Type (the transform-kind argument)
  * HANDLE  cudaStream_t -> wwr_stream (shared, the only opaque one)

GetStatusName / GetStatusString are hand-written inline switches (not
WWR_FUNCTION_RAW): the parser now SEES them, but they wrap backend-divergent
enumerator sets with no sound 1:1 neutral map, so they are listed in
explicit_skips (as the solver pair is) and reached through the raw module, not
the generic path. Everything fft executes runs on the device, so the thunks are
link-checked on no-GPU CI.
"""

from __future__ import annotations

from captypes import Handle, PointeePOD, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/fft_names.h"
PREFIX = "wwrfft"
CAPI_PREFIX = "wwr_fft_"

CUDA_MANIFEST = "cuda-13.0.x/cufft.json"
HIP_MANIFEST = "rocm-10.0.0/hipfft.json"
MODULES = ["wwr.fft", "wwr.complex"]

HANDLES = [
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
]

# cufftHandle is a plan id (typedef int); cufftReal / cufftDoubleReal are the
# element scalars -- all carried as their underlying C scalar, not void*.
SCALAR_ALIASES = {
    "cufftHandle": "int", "hipfftHandle": "int",
    "cufftReal": "float", "hipfftReal": "float",
    "cufftDoubleReal": "double", "hipfftDoubleReal": "double",
}

POINTEE_PODS = [
    PointeePOD(vendor_types=("cufftComplex", "hipfftComplex"),
               capi_pod="wwr_complex64"),
    PointeePOD(vendor_types=("cufftDoubleComplex", "hipfftDoubleComplex"),
               capi_pod="wwr_complex128"),
]

RESULT = ValueEnum(
    vendor_types=("cufftResult", "cufftResult_t", "hipfftResult", "hipfftResult_t"),
    capi_type="wwr_fft_result",
    wwr_type="wwrfftResult_t",
    fallback="WWR_FFT_RESULT_UNKNOWN",
    entries=(
        ("WWR_FFT_SUCCESS", "WWRFFT_SUCCESS"),
        ("WWR_FFT_INVALID_PLAN", "WWRFFT_INVALID_PLAN"),
        ("WWR_FFT_ALLOC_FAILED", "WWRFFT_ALLOC_FAILED"),
        ("WWR_FFT_INVALID_TYPE", "WWRFFT_INVALID_TYPE"),
        ("WWR_FFT_INVALID_VALUE", "WWRFFT_INVALID_VALUE"),
        ("WWR_FFT_INTERNAL_ERROR", "WWRFFT_INTERNAL_ERROR"),
        ("WWR_FFT_EXEC_FAILED", "WWRFFT_EXEC_FAILED"),
        ("WWR_FFT_SETUP_FAILED", "WWRFFT_SETUP_FAILED"),
        ("WWR_FFT_INVALID_SIZE", "WWRFFT_INVALID_SIZE"),
        ("WWR_FFT_UNALIGNED_DATA", "WWRFFT_UNALIGNED_DATA"),
        ("WWR_FFT_INVALID_DEVICE", "WWRFFT_INVALID_DEVICE"),
        ("WWR_FFT_NO_WORKSPACE", "WWRFFT_NO_WORKSPACE"),
        ("WWR_FFT_NOT_IMPLEMENTED", "WWRFFT_NOT_IMPLEMENTED"),
        ("WWR_FFT_NOT_SUPPORTED", "WWRFFT_NOT_SUPPORTED"),
    ),
)

TYPE = ValueEnum(
    vendor_types=("cufftType", "cufftType_t", "hipfftType", "hipfftType_t"),
    capi_type="wwr_fft_type",
    wwr_type="wwrfftType_t",
    fallback="WWR_FFT_R2C",
    entries=(
        ("WWR_FFT_R2C", "WWRFFT_R2C"),
        ("WWR_FFT_C2R", "WWRFFT_C2R"),
        ("WWR_FFT_C2C", "WWRFFT_C2C"),
        ("WWR_FFT_D2Z", "WWRFFT_D2Z"),
        ("WWR_FFT_Z2D", "WWRFFT_Z2D"),
        ("WWR_FFT_Z2Z", "WWRFFT_Z2Z"),
    ),
)

VALUE_ENUMS = [RESULT, TYPE]

# The two hand-written status-string switches: visible to the parser, but a
# backend-divergent enumerator mapping the generic path cannot re-wrap soundly
# (the solver pair's reason, §6). Skipped explicitly so they are LOGGED, never
# silently wrapped or dropped.
EXPLICIT_SKIPS = [
    ("wwrfftGetStatusName",
     "hand-written inline switch over backend-divergent enumerator sets, "
     "not a WWR_FUNCTION_RAW fragment entry"),
    ("wwrfftGetStatusString",
     "hand-written inline switch over backend-divergent enumerator sets, "
     "not a WWR_FUNCTION_RAW fragment entry"),
]


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="fft",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    value_enums=VALUE_ENUMS,
    scalar_aliases=SCALAR_ALIASES,
    explicit_skips=EXPLICIT_SKIPS,
    capi_name=capi_name,
)
