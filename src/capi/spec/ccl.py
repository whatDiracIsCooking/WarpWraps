"""The `ccl` (NCCL / RCCL) collective-communications surface, FRAGMENT-SOURCED.

The multi-GPU collectives library, and the one surface whose fragment IS its
module: wwr.ccl defines its neutral names directly in src/ccl.cppm with the plain
three-argument WWR_TYPE / WWR_VALUE / WWR_FUNCTION macros (over an IMPORTED raw
vendor module, wwr.cuda.nccl / wwr.hip.rccl), not in a separate
detail/*_names.h against the vendor header. NCCL and RCCL are source-compatible,
so every name spells cuda == hip. The generator sources this surface straight
from ccl.cppm (fragments.py learned the plain macro form), so that .cppm stays the
single source -- no duplicate detail/ fragment to drift (architecture.md §25).

It registers the vendor-free types ccl forces, resolved from the manifests:

  * HANDLES  ncclComm_t -> wwr_ccl_comm, ncclWindow_t -> wwr_ccl_window (opaque
    vendor POINTER typedefs -> void*), plus the shared cudaStream_t ->
    wwr_stream the collective calls take.
  * VALUE ENUMS  the result status (returned by nearly every call, mapped to a
    stable wwr_ccl_result), and the data-type / reduction-op / scalar-residence
    enums passed by value -- each a stable §6 capi enum the thunk maps
    capi<->neutral at the boundary.

ncclUniqueId / ncclConfig_t / ncclSimInfo_t are by-value / by-pointer structs.
A POINTER to one erases to void* (so ncclGetUniqueId / the config-taking inits
wrap). ncclCommInitRank / ncclCommInitRankConfig take ncclUniqueId BY VALUE -- a
128-byte opaque blob with no vendor-free POD the generic path builds -- so they
are SKIPPED with a logged reason (the generator's `struct passed by value`
report), never silently dropped. The pointer-form communicator inits
(ncclCommInitRankScalable, which takes ncclUniqueId*) still wrap.

Everything ccl executes runs on the DEVICE / across the fabric, so the thunks are
link-checked on no-GPU CI and run-checked on a multi-GPU box. See spec/blas.py
for the ValueEnum / Handle rationale and docs/architecture.md section 25.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum
from library import LibrarySpec, snake_capi_name

# The fragment this surface sources its names from: its own module interface.
FRAGMENT = "ccl.cppm"
PREFIX = "wwrccl"
CAPI_PREFIX = "wwr_ccl_"

CUDA_MANIFEST = "cuda-13.0.x/nccl.json"
HIP_MANIFEST = "rocm-10.0.0/rccl.json"
MODULES = ["wwr.ccl"]

# Opaque handles -> void*. The communicator and the window are vendor POINTER
# typedefs on both backends; wwr_stream is the shared stream handle the
# collective calls take (declared once in common.h).
HANDLES = [
    Handle(vendor_types=("ncclComm_t",), capi_type="wwr_ccl_comm"),
    Handle(vendor_types=("ncclWindow_t",), capi_type="wwr_ccl_window"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
]

POINTEE_PODS: list = []
PODS: list = []

# The result enum (wwrcclResult_t): returned by nearly every function and taken
# by value by GetErrorString. Each pair is (capi_enumerator, neutral wwr constant
# from the fragment); the stable capi values are fixed §6 integers.
RESULT = ValueEnum(
    vendor_types=("ncclResult_t",),
    capi_type="wwr_ccl_result",
    wwr_type="wwrcclResult_t",
    fallback="WWR_CCL_RESULT_INTERNAL_ERROR",
    entries=(
        ("WWR_CCL_RESULT_SUCCESS", "WWRCCL_SUCCESS"),
        ("WWR_CCL_RESULT_UNHANDLED_CUDA_ERROR", "WWRCCL_UNHANDLED_CUDA_ERROR"),
        ("WWR_CCL_RESULT_SYSTEM_ERROR", "WWRCCL_SYSTEM_ERROR"),
        ("WWR_CCL_RESULT_INTERNAL_ERROR", "WWRCCL_INTERNAL_ERROR"),
        ("WWR_CCL_RESULT_INVALID_ARGUMENT", "WWRCCL_INVALID_ARGUMENT"),
        ("WWR_CCL_RESULT_INVALID_USAGE", "WWRCCL_INVALID_USAGE"),
        ("WWR_CCL_RESULT_REMOTE_ERROR", "WWRCCL_REMOTE_ERROR"),
        ("WWR_CCL_RESULT_IN_PROGRESS", "WWRCCL_IN_PROGRESS"),
    ),
)

# The data type passed by value to every collective (count-of-elements typing).
DATA_TYPE = ValueEnum(
    vendor_types=("ncclDataType_t",),
    capi_type="wwr_ccl_data_type",
    wwr_type="wwrcclDataType_t",
    fallback="WWR_CCL_FLOAT32",
    entries=(
        ("WWR_CCL_INT8", "WWRCCL_INT8"),
        ("WWR_CCL_UINT8", "WWRCCL_UINT8"),
        ("WWR_CCL_INT32", "WWRCCL_INT32"),
        ("WWR_CCL_UINT32", "WWRCCL_UINT32"),
        ("WWR_CCL_INT64", "WWRCCL_INT64"),
        ("WWR_CCL_UINT64", "WWRCCL_UINT64"),
        ("WWR_CCL_FLOAT16", "WWRCCL_FLOAT16"),
        ("WWR_CCL_FLOAT32", "WWRCCL_FLOAT32"),
        ("WWR_CCL_FLOAT64", "WWRCCL_FLOAT64"),
        ("WWR_CCL_BFLOAT16", "WWRCCL_BFLOAT16"),
        ("WWR_CCL_FLOAT8E4M3", "WWRCCL_FLOAT8E4M3"),
        ("WWR_CCL_FLOAT8E5M2", "WWRCCL_FLOAT8E5M2"),
    ),
)

# The reduction operator passed by value to Reduce / AllReduce / ReduceScatter.
# The built-in ops only -- a custom op from RedOpCreatePreMulSum is created
# through the handle-less ncclRedOp_t out-param path (wrapped separately).
RED_OP = ValueEnum(
    vendor_types=("ncclRedOp_t",),
    capi_type="wwr_ccl_red_op",
    wwr_type="wwrcclRedOp_t",
    fallback="WWR_CCL_SUM",
    entries=(
        ("WWR_CCL_SUM", "WWRCCL_SUM"),
        ("WWR_CCL_PROD", "WWRCCL_PROD"),
        ("WWR_CCL_MAX", "WWRCCL_MAX"),
        ("WWR_CCL_MIN", "WWRCCL_MIN"),
        ("WWR_CCL_AVG", "WWRCCL_AVG"),
    ),
)

# The scalar residence passed by value to RedOpCreatePreMulSum (where the custom
# reduction's scalar lives -- device memory or an immediate host value).
SCALAR_RESIDENCE = ValueEnum(
    vendor_types=("ncclScalarResidence_t",),
    capi_type="wwr_ccl_scalar_residence",
    wwr_type="wwrcclScalarResidence_t",
    fallback="WWR_CCL_SCALAR_HOST_IMMEDIATE",
    entries=(
        ("WWR_CCL_SCALAR_DEVICE", "WWRCCL_SCALAR_DEVICE"),
        ("WWR_CCL_SCALAR_HOST_IMMEDIATE", "WWRCCL_SCALAR_HOST_IMMEDIATE"),
    ),
)

VALUE_ENUMS = [RESULT, DATA_TYPE, RED_OP, SCALAR_RESIDENCE]

# Every mapped enum constant is a fragment VALUE entry present in both manifests,
# so none is exempt from the manifest value cross-check.
ENUM_CONSTANTS_NOT_IN_MANIFEST: set[str] = set()

# No vendor scalar typedefs to normalise.
SCALAR_ALIASES: dict[str, str] = {}

# The two by-value-ncclUniqueId communicator inits are left to the generator's
# automatic SKIP (a struct passed by value, no vendor-free POD), so no explicit
# skip is listed here -- the generator reports them.
EXPLICIT_SKIPS: list[tuple[str, str]] = []


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="ccl",
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
