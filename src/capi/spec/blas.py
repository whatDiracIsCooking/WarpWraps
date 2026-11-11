"""The `blas` (cuBLAS / hipBLAS) surface as a FRAGMENT-SOURCED spec.

The shape spec/runtime_status.py models, scaled to the first LARGE library: the
whole curated wwrblas* surface (handle/stream/pointer-mode calls, the status
strings, all three BLAS levels, the batched / strided-batched / Ex extensions).
Like the status slice it carries NO signatures and NO vendor names -- it names
the fragment (detail/blas_names.h) and REGISTERS the vendor-free types blas
forces, which the status slice did not exercise:

  * HANDLES  wwrblasHandle_t / cudaStream_t -> void* (wwr_blas_handle / wwr_stream)
  * VALUE ENUMS  Operation / FillMode / DiagType / SideMode / PointerMode /
    compute / datatype / algo, plus the status enum -- each a stable capi enum
    with fixed §6 values the thunk maps capi<->neutral at the boundary
  * POINTEE PODS  cuComplex* -> wwr_complex64*, cuDoubleComplex* -> wwr_complex128*
    (reusing complex.py's PODs), __half* -> a new wwr_half POD

generate.py resolves every fragment function against the manifest pair, classifies
each parameter with these registrations, and emits the header / thunk / ffi.rs.
A function whose signature the classifier cannot represent is SKIPPED with a
logged reason (SKIP, below) -- never silently dropped.

Everything blas executes runs on the DEVICE (a handle binds a device; a GEMM
launches), so the thunks are link-checked on no-GPU CI and run-checked on a GPU
box (the wwr.capi.blas_smoke ctest, REQUIRES_GPU). See the README and
architecture.md §25.
"""

from __future__ import annotations

from captypes import Handle, PointeePOD, ValueEnum
from library import LibrarySpec

# The fragment this surface sources its names from (relative to src/).
FRAGMENT = "detail/blas_names.h"

# A capi POD for the vendor half type (__half / hipblasHalf), 16 opaque bits.
# Rust sees a transparent u16; the thunk reinterpret_casts to the neutral
# wwr::wwrHalf the Hgemm forwarders name (layout-identical, §3/§12). complex.py's
# wwr_complex64 / wwr_complex128 are reused as the complex PODs.
HALF_POD = {"capi": "wwr_half", "wwr": "wwr::wwrHalf",
            "fields": [("uint16_t", "bits")]}

# Opaque handles -> void*. The two the surface names: the library handle and the
# stream a SetStream/GetStream passes.
HANDLES = [
    Handle(vendor_types=("cublasHandle_t", "hipblasHandle_t"),
           capi_type="wwr_blas_handle"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"),
           capi_type="wwr_stream"),
]

# Pointee PODs: a complex / half POINTER keeps a typed capi POD, not void*.
POINTEE_PODS = [
    PointeePOD(vendor_types=("cuComplex", "hipComplex", "hipblasComplex"),
               capi_pod="wwr_complex64"),
    PointeePOD(vendor_types=("cuDoubleComplex", "hipDoubleComplex",
                             "hipblasDoubleComplex"),
               capi_pod="wwr_complex128"),
    PointeePOD(vendor_types=("__half", "hipblasHalf"), capi_pod="wwr_half"),
]

# The stable value enums (§6). Each `entries` pair is (capi_enumerator, neutral
# wwr constant); the neutral constant is itself a fragment VALUE entry
# (WWR_VALUE_RAW), so this lists the MAPPING, not a second copy of the names. The
# stable capi values are assigned fixed ascending integers at emit time and are a
# frozen ABI contract; the raw backend value (which DIFFERS per backend -- OP_N
# is 0 on cuBLAS, 111 on hipBLAS) never crosses the boundary.
STATUS = ValueEnum(
    vendor_types=("cublasStatus_t", "hipblasStatus_t"),
    capi_type="wwr_blas_status",
    wwr_type="wwrblasStatus_t",
    fallback="WWR_BLAS_STATUS_UNKNOWN",
    entries=(
        ("WWR_BLAS_STATUS_SUCCESS", "WWRBLAS_STATUS_SUCCESS"),
        ("WWR_BLAS_STATUS_NOT_INITIALIZED", "WWRBLAS_STATUS_NOT_INITIALIZED"),
        ("WWR_BLAS_STATUS_ALLOC_FAILED", "WWRBLAS_STATUS_ALLOC_FAILED"),
        ("WWR_BLAS_STATUS_ARCH_MISMATCH", "WWRBLAS_STATUS_ARCH_MISMATCH"),
        ("WWR_BLAS_STATUS_EXECUTION_FAILED", "WWRBLAS_STATUS_EXECUTION_FAILED"),
        ("WWR_BLAS_STATUS_INTERNAL_ERROR", "WWRBLAS_STATUS_INTERNAL_ERROR"),
        ("WWR_BLAS_STATUS_INVALID_VALUE", "WWRBLAS_STATUS_INVALID_VALUE"),
        ("WWR_BLAS_STATUS_MAPPING_ERROR", "WWRBLAS_STATUS_MAPPING_ERROR"),
        ("WWR_BLAS_STATUS_NOT_SUPPORTED", "WWRBLAS_STATUS_NOT_SUPPORTED"),
    ),
)

OPERATION = ValueEnum(
    vendor_types=("cublasOperation_t", "hipblasOperation_t"),
    capi_type="wwr_blas_operation",
    wwr_type="wwrblasOperation_t",
    fallback="WWR_BLAS_OP_N",
    entries=(
        ("WWR_BLAS_OP_N", "WWRBLAS_OP_N"),
        ("WWR_BLAS_OP_T", "WWRBLAS_OP_T"),
        ("WWR_BLAS_OP_C", "WWRBLAS_OP_C"),
    ),
)

FILL_MODE = ValueEnum(
    vendor_types=("cublasFillMode_t", "hipblasFillMode_t"),
    capi_type="wwr_blas_fill_mode",
    wwr_type="wwrblasFillMode_t",
    fallback="WWR_BLAS_FILL_MODE_LOWER",
    entries=(
        ("WWR_BLAS_FILL_MODE_LOWER", "WWRBLAS_FILL_MODE_LOWER"),
        ("WWR_BLAS_FILL_MODE_UPPER", "WWRBLAS_FILL_MODE_UPPER"),
    ),
)

DIAG_TYPE = ValueEnum(
    vendor_types=("cublasDiagType_t", "hipblasDiagType_t"),
    capi_type="wwr_blas_diag_type",
    wwr_type="wwrblasDiagType_t",
    fallback="WWR_BLAS_DIAG_NON_UNIT",
    entries=(
        ("WWR_BLAS_DIAG_NON_UNIT", "WWRBLAS_DIAG_NON_UNIT"),
        ("WWR_BLAS_DIAG_UNIT", "WWRBLAS_DIAG_UNIT"),
    ),
)

SIDE_MODE = ValueEnum(
    vendor_types=("cublasSideMode_t", "hipblasSideMode_t"),
    capi_type="wwr_blas_side_mode",
    wwr_type="wwrblasSideMode_t",
    fallback="WWR_BLAS_SIDE_LEFT",
    entries=(
        ("WWR_BLAS_SIDE_LEFT", "WWRBLAS_SIDE_LEFT"),
        ("WWR_BLAS_SIDE_RIGHT", "WWRBLAS_SIDE_RIGHT"),
    ),
)

POINTER_MODE = ValueEnum(
    vendor_types=("cublasPointerMode_t", "hipblasPointerMode_t"),
    capi_type="wwr_blas_pointer_mode",
    wwr_type="wwrblasPointerMode_t",
    fallback="WWR_BLAS_POINTER_MODE_HOST",
    entries=(
        ("WWR_BLAS_POINTER_MODE_HOST", "WWRBLAS_POINTER_MODE_HOST"),
        ("WWR_BLAS_POINTER_MODE_DEVICE", "WWRBLAS_POINTER_MODE_DEVICE"),
    ),
)

# cudaDataType / hipDataType. Its constants (CUDA_R_16F / HIP_R_16F) live in the
# runtime library, NOT the cublas/hipblas manifest, so generate.py cannot
# cross-check their VALUE against the blas manifest -- the thunk never needs it
# (it compares the neutral wwr::WWRBLAS_R_* constants, which the fragment carries).
DATA_TYPE = ValueEnum(
    vendor_types=("cudaDataType", "hipDataType", "cudaDataType_t"),
    capi_type="wwr_blas_data_type",
    wwr_type="wwrblasDataType_t",
    fallback="WWR_BLAS_R_32F",
    entries=(
        ("WWR_BLAS_R_16F", "WWRBLAS_R_16F"),
        ("WWR_BLAS_R_32F", "WWRBLAS_R_32F"),
        ("WWR_BLAS_R_64F", "WWRBLAS_R_64F"),
    ),
)

COMPUTE_TYPE = ValueEnum(
    vendor_types=("cublasComputeType_t", "hipblasComputeType_t"),
    capi_type="wwr_blas_compute_type",
    wwr_type="wwrblasComputeType_t",
    fallback="WWR_BLAS_COMPUTE_32F",
    entries=(
        ("WWR_BLAS_COMPUTE_16F", "WWRBLAS_COMPUTE_16F"),
        ("WWR_BLAS_COMPUTE_32F", "WWRBLAS_COMPUTE_32F"),
        ("WWR_BLAS_COMPUTE_64F", "WWRBLAS_COMPUTE_64F"),
    ),
)

GEMM_ALGO = ValueEnum(
    vendor_types=("cublasGemmAlgo_t", "hipblasGemmAlgo_t"),
    capi_type="wwr_blas_gemm_algo",
    wwr_type="wwrblasGemmAlgo_t",
    fallback="WWR_BLAS_GEMM_DEFAULT",
    entries=(
        ("WWR_BLAS_GEMM_DEFAULT", "WWRBLAS_GEMM_DEFAULT"),
    ),
)

VALUE_ENUMS = [STATUS, OPERATION, FILL_MODE, DIAG_TYPE, SIDE_MODE, POINTER_MODE,
               DATA_TYPE, COMPUTE_TYPE, GEMM_ALGO]

# The capi name for a fragment function: strip the wwr prefix, snake_case the
# stem, and re-prefix wwr_blas_. wwrblasSgemm -> wwr_blas_sgemm, wwrblasGemmEx_64
# -> wwr_blas_gemm_ex_64. Deterministic and collision-free over the surface.
PREFIX = "wwrblas"


def capi_name(wwr_name: str) -> str:
    stem = wwr_name[len(PREFIX):]
    out = []
    for i, ch in enumerate(stem):
        prev_lower = i > 0 and not stem[i - 1].isupper()
        next_lower = i + 1 < len(stem) and stem[i + 1].islower()
        if ch.isupper() and i > 0 and (prev_lower or next_lower):
            out.append("_")
        out.append(ch.lower())
    return "wwr_blas_" + "".join(out).lstrip("_")


# Enum constants a ValueEnum references that are NOT in the blas manifest (they
# live in another vendor library). generate.py skips the manifest VALUE
# cross-check for these -- the thunk only needs the neutral wwr:: constant, which
# the fragment carries. Keyed by the vendor constant name in each manifest.
ENUM_CONSTANTS_NOT_IN_MANIFEST = {
    "CUDA_R_16F", "CUDA_R_32F", "CUDA_R_64F",
    "HIP_R_16F", "HIP_R_32F", "HIP_R_64F",
}

# blas as a LibrarySpec: the same bundle the four milestone-#6 surfaces declare,
# so generate.py's one generic library path emits blas exactly as before. Its
# modules are wwr.blas + wwr.fp16 (the half forwarders), which transitively carry
# cuBLAS / hipBLAS -- so the blas unit's .so NEEDs the vendor BLAS library.
SPEC = LibrarySpec(
    name="blas",
    fragment=FRAGMENT,
    capi_prefix="wwr_blas_",
    cuda_manifest="cuda-13.0.x/cublas_v2.json",
    hip_manifest="rocm-10.0.0/hipblas.json",
    modules=["wwr.blas", "wwr.fp16"],
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    pods=[HALF_POD],
    value_enums=VALUE_ENUMS,
    enum_constants_not_in_manifest=ENUM_CONSTANTS_NOT_IN_MANIFEST,
    scalar_aliases={},
    explicit_skips=[],
    capi_name=capi_name,
)
