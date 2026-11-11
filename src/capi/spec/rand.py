"""The `rand` (cuRAND / hipRAND) HOST surface as a FRAGMENT-SOURCED spec.

The host RNG library (detail/rand_names.h: the generator/status/rng-type/ordering
types, the status + rng-type + ordering + direction-vector-set constants, 25
functions). Same SHAPE as blas (spec/blas.py): a status enum returned by value,
opaque handles, by-value value enums -- so it registers through the same captypes
machinery and the generic fragment-sourced library path emits it.

  * HANDLES  curandGenerator_t -> wwr_rand_generator (void*),
    curandDiscreteDistribution_t -> wwr_rand_discrete_distribution (void*),
    cudaStream_t -> wwr_stream (the stream a SetStream passes -- NOT in the rand
    fragment's maps, so registered here with its backend pair)
  * VALUE ENUMS  curandStatus_t (returned by every function), curandRngType_t,
    curandOrdering_t, curandDirectionVectorSet_t (by-value parameters) -- each a
    stable capi enum the thunk maps capi<->neutral at the boundary

The two direction-vector TABLE types (curandDirectionVectors32_t /
curandDirectionVectors64_t) appear only as an opaque OUT double-pointer in
GetDirectionVectors32/64; the generic pointer arm erases a vendor pointee to
void*, which is exactly right for a library-owned read-only table a caller only
hands back. No complex, no half. See spec/blas.py and docs/architecture.md §25.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/rand_names.h"
PREFIX = "wwrrand"
CAPI_PREFIX = "wwr_rand_"

CUDA_MANIFEST = "cuda-13.0.x/curand.json"
HIP_MANIFEST = "rocm-10.0.0/hiprand.json"
MODULES = ["wwr.rand"]

HANDLES = [
    Handle(vendor_types=("curandGenerator_t", "hiprandGenerator_t"),
           capi_type="wwr_rand_generator"),
    Handle(vendor_types=("curandDiscreteDistribution_t",
                         "hiprandDiscreteDistribution_t"),
           capi_type="wwr_rand_discrete_distribution"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"),
           capi_type="wwr_stream"),
]

POINTEE_PODS: list = []
PODS: list = []

STATUS = ValueEnum(
    vendor_types=("curandStatus_t", "hiprandStatus_t"),
    capi_type="wwr_rand_status",
    wwr_type="wwrrandStatus_t",
    fallback="WWR_RAND_STATUS_INTERNAL_ERROR",
    entries=(
        ("WWR_RAND_STATUS_SUCCESS", "WWRRAND_STATUS_SUCCESS"),
        ("WWR_RAND_STATUS_VERSION_MISMATCH", "WWRRAND_STATUS_VERSION_MISMATCH"),
        ("WWR_RAND_STATUS_NOT_INITIALIZED", "WWRRAND_STATUS_NOT_INITIALIZED"),
        ("WWR_RAND_STATUS_ALLOCATION_FAILED", "WWRRAND_STATUS_ALLOCATION_FAILED"),
        ("WWR_RAND_STATUS_TYPE_ERROR", "WWRRAND_STATUS_TYPE_ERROR"),
        ("WWR_RAND_STATUS_OUT_OF_RANGE", "WWRRAND_STATUS_OUT_OF_RANGE"),
        ("WWR_RAND_STATUS_LENGTH_NOT_MULTIPLE",
         "WWRRAND_STATUS_LENGTH_NOT_MULTIPLE"),
        ("WWR_RAND_STATUS_DOUBLE_PRECISION_REQUIRED",
         "WWRRAND_STATUS_DOUBLE_PRECISION_REQUIRED"),
        ("WWR_RAND_STATUS_LAUNCH_FAILURE", "WWRRAND_STATUS_LAUNCH_FAILURE"),
        ("WWR_RAND_STATUS_PREEXISTING_FAILURE",
         "WWRRAND_STATUS_PREEXISTING_FAILURE"),
        ("WWR_RAND_STATUS_INITIALIZATION_FAILED",
         "WWRRAND_STATUS_INITIALIZATION_FAILED"),
        ("WWR_RAND_STATUS_ARCH_MISMATCH", "WWRRAND_STATUS_ARCH_MISMATCH"),
        ("WWR_RAND_STATUS_INTERNAL_ERROR", "WWRRAND_STATUS_INTERNAL_ERROR"),
    ),
)

RNG_TYPE = ValueEnum(
    vendor_types=("curandRngType_t", "hiprandRngType_t"),
    capi_type="wwr_rand_rng_type",
    wwr_type="wwrrandRngType_t",
    fallback="WWR_RAND_RNG_PSEUDO_DEFAULT",
    entries=(
        ("WWR_RAND_RNG_TEST", "WWRRAND_RNG_TEST"),
        ("WWR_RAND_RNG_PSEUDO_DEFAULT", "WWRRAND_RNG_PSEUDO_DEFAULT"),
        ("WWR_RAND_RNG_PSEUDO_XORWOW", "WWRRAND_RNG_PSEUDO_XORWOW"),
        ("WWR_RAND_RNG_PSEUDO_MRG32K3A", "WWRRAND_RNG_PSEUDO_MRG32K3A"),
        ("WWR_RAND_RNG_PSEUDO_MTGP32", "WWRRAND_RNG_PSEUDO_MTGP32"),
        ("WWR_RAND_RNG_PSEUDO_MT19937", "WWRRAND_RNG_PSEUDO_MT19937"),
        ("WWR_RAND_RNG_PSEUDO_PHILOX4_32_10",
         "WWRRAND_RNG_PSEUDO_PHILOX4_32_10"),
        ("WWR_RAND_RNG_QUASI_DEFAULT", "WWRRAND_RNG_QUASI_DEFAULT"),
        ("WWR_RAND_RNG_QUASI_SOBOL32", "WWRRAND_RNG_QUASI_SOBOL32"),
        ("WWR_RAND_RNG_QUASI_SCRAMBLED_SOBOL32",
         "WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL32"),
        ("WWR_RAND_RNG_QUASI_SOBOL64", "WWRRAND_RNG_QUASI_SOBOL64"),
        ("WWR_RAND_RNG_QUASI_SCRAMBLED_SOBOL64",
         "WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL64"),
    ),
)

ORDERING = ValueEnum(
    vendor_types=("curandOrdering_t", "hiprandOrdering_t"),
    capi_type="wwr_rand_ordering",
    wwr_type="wwrrandOrdering_t",
    fallback="WWR_RAND_ORDERING_PSEUDO_DEFAULT",
    entries=(
        ("WWR_RAND_ORDERING_PSEUDO_BEST", "WWRRAND_ORDERING_PSEUDO_BEST"),
        ("WWR_RAND_ORDERING_PSEUDO_DEFAULT", "WWRRAND_ORDERING_PSEUDO_DEFAULT"),
        ("WWR_RAND_ORDERING_PSEUDO_SEEDED", "WWRRAND_ORDERING_PSEUDO_SEEDED"),
        ("WWR_RAND_ORDERING_PSEUDO_LEGACY", "WWRRAND_ORDERING_PSEUDO_LEGACY"),
        ("WWR_RAND_ORDERING_PSEUDO_DYNAMIC", "WWRRAND_ORDERING_PSEUDO_DYNAMIC"),
        ("WWR_RAND_ORDERING_QUASI_DEFAULT", "WWRRAND_ORDERING_QUASI_DEFAULT"),
    ),
)

DIRECTION_VECTOR_SET = ValueEnum(
    vendor_types=("curandDirectionVectorSet_t", "hiprandDirectionVectorSet_t"),
    capi_type="wwr_rand_direction_vector_set",
    wwr_type="wwrrandDirectionVectorSet_t",
    fallback="WWR_RAND_DIRECTION_VECTORS_32_JOEKUO6",
    entries=(
        ("WWR_RAND_DIRECTION_VECTORS_32_JOEKUO6",
         "WWRRAND_DIRECTION_VECTORS_32_JOEKUO6"),
        ("WWR_RAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
         "WWRRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6"),
        ("WWR_RAND_DIRECTION_VECTORS_64_JOEKUO6",
         "WWRRAND_DIRECTION_VECTORS_64_JOEKUO6"),
        ("WWR_RAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
         "WWRRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6"),
    ),
)

VALUE_ENUMS = [STATUS, RNG_TYPE, ORDERING, DIRECTION_VECTOR_SET]

ENUM_CONSTANTS_NOT_IN_MANIFEST: set[str] = set()
SCALAR_ALIASES: dict[str, str] = {}

# No hand-written fragment forwarders are skipped. On CUDA the fragment provides
# two `#if WWR_SELECTED_CUDA` inline const_cast forwarders for the scramble
# constants, but the fragment ALSO carries the plain WWR_FUNCTION_RAW form (the
# #else branch) for wwrrandGetScrambleConstants32/64, which the parser DOES see;
# they resolve against curandGetScrambleConstants32/64 and are wrapped. All 25.
EXPLICIT_SKIPS: list[tuple[str, str]] = []


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="rand",
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
