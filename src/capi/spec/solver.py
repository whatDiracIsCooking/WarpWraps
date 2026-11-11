"""The `solver` (cuSOLVER Dn / hipSOLVER) surface as a FRAGMENT-SOURCED spec.

The dense linear-algebra library and the LARGEST of the four milestone-#6
surfaces (detail/solver_names.h: the Dn handle / params / Jacobi-info types, the
status + eig-mode/type/range + data-type constants, and 234 functions -- the
legacy linear-solver, legacy eigen/SVD, and modern X-prefixed APIs). Only the
`Dn` (dense) surface: hipSOLVER has no Sp/Mg counterpart, so the fragment names
none and this spec wraps exactly what the fragment lists. Same SHAPE as blas
(spec/blas.py) -- a status enum returned by value, opaque handles, by-value value
enums, complex POINTER parameters -- so it registers through the same captypes
machinery and the generic fragment-sourced library path emits it.

  * HANDLES  cusolverDnHandle_t, cusolverDnParams_t (the modern X-API params),
    the two Jacobi-info objects (syevjInfo_t / gesvdjInfo_t -- note the CUDA
    spellings carry NO cusolver prefix) and cudaStream_t -> void*.
  * POINTEE PODS  cuComplex * -> wwr_complex64 *, cuDoubleComplex * ->
    wwr_complex128 * (the matrix / work-buffer pointers of the C/Z routines),
    reusing complex.py's shared PODs.
  * VALUE ENUMS  cusolverStatus_t (returned by every function), the eig
    mode/type/range enums, and the classic-BLAS fill-mode / side-mode / operation
    enums cuSOLVER inherits for uplo / side / trans -- all stable capi enums the
    thunk maps capi<->neutral (the fill/side/op raw values DIVERGE across
    backends, mapped to the WWRSOLVER_* constants the fragment carries).
  * SCALAR ALIASES  cusolver_int_t -> int (cuSOLVER's own `typedef int`), and
    cudaDataType -> int (identical enumerator values across backends, so a raw
    int is backend-neutral and carries every data type the X-API accepts).

The two hand-written status-string switches (wwrsolverGetStatusName /
wwrsolverGetStatusString) are NOT WWR_FUNCTION_RAW macro entries -- they are
`inline const char *` functions over backend-divergent enumerator SETS -- so the
fragment parser never sees them and they are SKIPPED explicitly (below), never
silently. See spec/blas.py and docs/architecture.md section 25.
"""

from __future__ import annotations

from captypes import Handle, PointeePOD, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/solver_names.h"
PREFIX = "wwrsolver"
CAPI_PREFIX = "wwr_solver_"

CUDA_MANIFEST = "cuda-13.0.x/cusolverDn.json"
HIP_MANIFEST = "rocm-10.0.0/hipsolver.json"
MODULES = ["wwr.solver"]

HANDLES = [
    Handle(vendor_types=("cusolverDnHandle_t", "hipsolverDnHandle_t",
                         "hipsolverHandle_t"),
           capi_type="wwr_solver_handle"),
    Handle(vendor_types=("cusolverDnParams_t", "hipsolverDnParams_t"),
           capi_type="wwr_solver_params"),
    Handle(vendor_types=("syevjInfo_t", "hipsolverSyevjInfo_t"),
           capi_type="wwr_solver_syevj_info"),
    Handle(vendor_types=("gesvdjInfo_t", "hipsolverGesvdjInfo_t"),
           capi_type="wwr_solver_gesvdj_info"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"),
           capi_type="wwr_stream"),
]

POINTEE_PODS = [
    PointeePOD(vendor_types=("cuComplex", "hipComplex", "hipFloatComplex"),
               capi_pod="wwr_complex64"),
    PointeePOD(vendor_types=("cuDoubleComplex", "hipDoubleComplex"),
               capi_pod="wwr_complex128"),
]

PODS: list = []

STATUS = ValueEnum(
    vendor_types=("cusolverStatus_t", "hipsolverStatus_t"),
    capi_type="wwr_solver_status",
    wwr_type="wwrsolverStatus_t",
    fallback="WWR_SOLVER_STATUS_UNKNOWN",
    entries=(
        ("WWR_SOLVER_STATUS_SUCCESS", "WWRSOLVER_STATUS_SUCCESS"),
    ),
)

EIG_MODE = ValueEnum(
    vendor_types=("cusolverEigMode_t", "hipsolverEigMode_t"),
    capi_type="wwr_solver_eig_mode",
    wwr_type="wwrsolverEigMode_t",
    fallback="WWR_SOLVER_EIG_MODE_NOVECTOR",
    entries=(
        ("WWR_SOLVER_EIG_MODE_NOVECTOR", "WWRSOLVER_EIG_MODE_NOVECTOR"),
        ("WWR_SOLVER_EIG_MODE_VECTOR", "WWRSOLVER_EIG_MODE_VECTOR"),
    ),
)

EIG_TYPE = ValueEnum(
    vendor_types=("cusolverEigType_t", "hipsolverEigType_t"),
    capi_type="wwr_solver_eig_type",
    wwr_type="wwrsolverEigType_t",
    fallback="WWR_SOLVER_EIG_TYPE_1",
    entries=(
        ("WWR_SOLVER_EIG_TYPE_1", "WWRSOLVER_EIG_TYPE_1"),
        ("WWR_SOLVER_EIG_TYPE_2", "WWRSOLVER_EIG_TYPE_2"),
        ("WWR_SOLVER_EIG_TYPE_3", "WWRSOLVER_EIG_TYPE_3"),
    ),
)

EIG_RANGE = ValueEnum(
    vendor_types=("cusolverEigRange_t", "hipsolverEigRange_t"),
    capi_type="wwr_solver_eig_range",
    wwr_type="wwrsolverEigRange_t",
    fallback="WWR_SOLVER_EIG_RANGE_ALL",
    entries=(
        ("WWR_SOLVER_EIG_RANGE_ALL", "WWRSOLVER_EIG_RANGE_ALL"),
        ("WWR_SOLVER_EIG_RANGE_V", "WWRSOLVER_EIG_RANGE_V"),
        ("WWR_SOLVER_EIG_RANGE_I", "WWRSOLVER_EIG_RANGE_I"),
    ),
)

FILL_MODE = ValueEnum(
    vendor_types=("cublasFillMode_t", "hipblasFillMode_t"),
    capi_type="wwr_solver_fill_mode",
    wwr_type="wwrsolverFillMode_t",
    fallback="WWR_SOLVER_FILL_MODE_LOWER",
    entries=(
        ("WWR_SOLVER_FILL_MODE_LOWER", "WWRSOLVER_FILL_MODE_LOWER"),
        ("WWR_SOLVER_FILL_MODE_UPPER", "WWRSOLVER_FILL_MODE_UPPER"),
    ),
)

SIDE_MODE = ValueEnum(
    vendor_types=("cublasSideMode_t", "hipblasSideMode_t"),
    capi_type="wwr_solver_side_mode",
    wwr_type="wwrsolverSideMode_t",
    fallback="WWR_SOLVER_SIDE_LEFT",
    entries=(
        ("WWR_SOLVER_SIDE_LEFT", "WWRSOLVER_SIDE_LEFT"),
        ("WWR_SOLVER_SIDE_RIGHT", "WWRSOLVER_SIDE_RIGHT"),
    ),
)

OPERATION = ValueEnum(
    vendor_types=("cublasOperation_t", "hipblasOperation_t"),
    capi_type="wwr_solver_operation",
    wwr_type="wwrsolverOperation_t",
    fallback="WWR_SOLVER_OP_N",
    entries=(
        ("WWR_SOLVER_OP_N", "WWRSOLVER_OP_N"),
        ("WWR_SOLVER_OP_T", "WWRSOLVER_OP_T"),
        ("WWR_SOLVER_OP_C", "WWRSOLVER_OP_C"),
    ),
)

VALUE_ENUMS = [STATUS, EIG_MODE, EIG_TYPE, EIG_RANGE, FILL_MODE, SIDE_MODE,
               OPERATION]

# The classic-BLAS fill/side/op constants (and the data-type constants) live in a
# NEIGHBOURING vendor library's header (cublas_api.h / hipblas-common.h and
# library_types.h), not the cusolverDn/hipsolver manifest, so they are exempt from
# the manifest value cross-check: the thunk compares the neutral wwr:: constant
# the solver vendor header pulls in, and never needs the raw manifest value.
ENUM_CONSTANTS_NOT_IN_MANIFEST = {
    "CUBLAS_FILL_MODE_LOWER", "CUBLAS_FILL_MODE_UPPER",
    "CUBLAS_SIDE_LEFT", "CUBLAS_SIDE_RIGHT",
    "CUBLAS_OP_N", "CUBLAS_OP_T", "CUBLAS_OP_C",
    "HIPBLAS_FILL_MODE_LOWER", "HIPBLAS_FILL_MODE_UPPER",
    "HIPBLAS_SIDE_LEFT", "HIPBLAS_SIDE_RIGHT",
    "HIPBLAS_OP_N", "HIPBLAS_OP_T", "HIPBLAS_OP_C",
}

# cusolver_int_t is cuSOLVER's own `typedef int`; cudaDataType / hipDataType share
# their enumerator integer values across backends (library_types.h), so both fold
# to a plain `int` -- no vendor name leaks and the value is backend-neutral.
SCALAR_ALIASES = {
    "cusolver_int_t": "int",
    "cudaDataType": "int",
    "cudaDataType_t": "int",
}

# The two status-string switches are hand-written `inline const char *` functions
# in the fragment, NOT WWR_FUNCTION_RAW macro entries (the parser does not see
# them), and they wrap backend-divergent enumerator SETS with no 1:1 neutral
# mapping -- so they are skipped explicitly, as the blas GetStatusName/String are.
EXPLICIT_SKIPS = [
    ("wwrsolverGetStatusName",
     "hand-written inline switch over backend-divergent enumerator sets, "
     "not a WWR_FUNCTION_RAW fragment entry"),
    ("wwrsolverGetStatusString",
     "hand-written inline switch over backend-divergent enumerator sets, "
     "not a WWR_FUNCTION_RAW fragment entry"),
]


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="solver",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    pods=PODS,
    value_enums=VALUE_ENUMS,
    enum_constants_not_in_manifest=ENUM_CONSTANTS_NOT_IN_MANIFEST,
    scalar_aliases=SCALAR_ALIASES,
    explicit_skips=EXPLICIT_SKIPS,
    capi_name=capi_name,
)
