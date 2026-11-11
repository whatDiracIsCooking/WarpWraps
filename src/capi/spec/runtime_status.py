"""The `runtime_api` surface as a FRAGMENT-SOURCED surface.

This is the thin per-surface definition the fragment-sourcing path needs. It
carries NO signatures and NO vendor names: it names

  * the fragment to source from (`detail/runtime_api_names.h`),
  * the opaque handles the runtime surface forces (stream / event / mem-pool /
    graph / array / library / user-object -> `void*`), and
  * the one thing a name cannot carry -- the stable `wwr_status` enum map (§6).

Unlike the vendor-library surfaces (spec/blas.py and the milestone-#6 four), this
one does NOT enumerate its functions: #348 wraps the WHOLE host-callable runtime,
so the generator sweeps EVERY `WWR_RT_FUNCTION` / `WWR_FUNCTION_RAW` entry in the
fragment and classifies each against the manifests. A function the classifier
cannot represent -- an enum passed BY VALUE whose neutral enumerator constants the
fragment does not carry, a by-value descriptor struct (`cudaMemcpy3DParms`,
`cudaExtent`), a function-pointer parameter (`cudaStreamCallback_t` /
`cudaHostFn_t`), a kernel `void**` launch, or a type whose ABI diverges between
backends (`cudaTextureObject_t` is a 64-bit integer on CUDA but a pointer on HIP)
-- is SKIPPED with a logged reason, never silently dropped (the milestone's NO
SILENT DROPS rule). The status slice that first shipped here (#336: version /
device-count / set-device) is a strict subset of this sweep, and its three capi
names are unchanged (the snake transform is stable), so the ABI does not break.

Everything runtime_api executes that touches a device runs on the DEVICE at call
time, but the surface itself is host-callable (cudart / amdhip do not probe a GPU
on load), so the thunks are link-checked on no-GPU CI and the host-safe slice
(version / device count) runs there too.

`complex` is NOT sourced this way: its thunks forward into wwr-AUTHORED inline
wrappers (`wwr::wwrCaddf`), which are not vendor functions and so appear in no
vendor manifest. It stays a documented hand-written special case in
`spec/complex.py`; see the generator and the README.
"""

from __future__ import annotations

from captypes import EnumMap, Handle

# The fragment this surface sources its names from (relative to src/).
FRAGMENT = "detail/runtime_api_names.h"

# The wwr-name prefix stripped before the snake_case capi transform, and the capi
# prefix re-glued on. `wwrRuntimeGetVersion` -> `wwr_runtime_get_version`,
# `wwrStreamCreateWithFlags` -> `wwr_stream_create_with_flags`. The three status
# slice names (#336) transform to exactly what shipped, so the ABI is unchanged.
PREFIX = "wwr"
CAPI_PREFIX = "wwr_"

# The opaque runtime handles -> `void*`. Each is a vendor POINTER typedef on BOTH
# backends (verified against the manifests), so erasing it to a vendor-free
# `void*` is sound -- the thunk C-style-casts at the boundary. The texture /
# surface object types are DELIBERATELY ABSENT: `cudaTextureObject_t` /
# `cudaSurfaceObject_t` are a 64-bit integer on CUDA but a pointer on HIP (the
# BaseHandle two-liveness-paths divergence, see detail/runtime_api_names.h), so
# they are not one vendor-free handle, and the few functions that take them are
# skipped by the generator.
HANDLES = [
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
    Handle(vendor_types=("cudaEvent_t", "hipEvent_t"), capi_type="wwr_event"),
    Handle(vendor_types=("cudaMemPool_t", "hipMemPool_t"),
           capi_type="wwr_mem_pool"),
    Handle(vendor_types=("cudaGraph_t", "hipGraph_t"), capi_type="wwr_graph"),
    Handle(vendor_types=("cudaGraphExec_t", "hipGraphExec_t"),
           capi_type="wwr_graph_exec"),
    Handle(vendor_types=("cudaArray_t", "hipArray_t"), capi_type="wwr_array"),
    Handle(vendor_types=("cudaArray_const_t", "hipArray_const_t"),
           capi_type="wwr_array_const"),
    Handle(vendor_types=("cudaMipmappedArray_t", "hipMipmappedArray_t"),
           capi_type="wwr_mipmapped_array"),
    Handle(vendor_types=("cudaMipmappedArray_const_t",
                         "hipMipmappedArray_const_t"),
           capi_type="wwr_mipmapped_array_const"),
    Handle(vendor_types=("cudaLibrary_t", "hipLibrary_t"),
           capi_type="wwr_library"),
    Handle(vendor_types=("cudaUserObject_t", "hipUserObject_t"),
           capi_type="wwr_user_object"),
]

# The stable status enum (§6). The vendor return type cudaError_t / hipError_t is
# classified as this EnumMap: a fixed-value capi enum with SUCCESS first (== 0),
# the curated error set, UNKNOWN last, and a `_map` that compares the NEUTRAL
# wwr:: constants. The values are backend-independent and frozen (architecture.md
# §21); each `entries` pair is (capi_enumerator, neutral wwr constant), and both
# the wwr constant and the SUCCESS/error vendor names are themselves fragment
# entries (WWR_RT_VALUE(Success), WWR_RT_VALUE(ErrorInvalidValue), ...), so this
# lists the MAPPING, not a second copy of the names.
STATUS = EnumMap(
    vendor_types=("cudaError_t", "hipError_t"),
    capi_type="wwr_status",
    success="WWR_STATUS_SUCCESS",
    unknown="WWR_STATUS_UNKNOWN",
    entries=(
        ("WWR_STATUS_INVALID_VALUE", "wwrErrorInvalidValue"),
        ("WWR_STATUS_INVALID_DEVICE", "wwrErrorInvalidDevice"),
        ("WWR_STATUS_NO_DEVICE", "wwrErrorNoDevice"),
        ("WWR_STATUS_NOT_READY", "wwrErrorNotReady"),
        ("WWR_STATUS_MEMORY_ALLOCATION", "wwrErrorMemoryAllocation"),
        ("WWR_STATUS_NOT_SUPPORTED", "wwrErrorNotSupported"),
    ),
)
