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

from captypes import EnumMap, Handle, ValueEnum

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
# The NEUTRAL `wwr*_t` spelling is listed beside each vendor pair because the
# hand-written forwarders (wwrMallocAsync, wwrGraphInstantiate, ...) are written
# in neutral types -- the classifier resolves their parsed signature against this
# same registry, not a vendor manifest decl. A vendor-keyed macro function and a
# neutral-keyed hand-written function both land on the one vendor-free capi_type.
HANDLES = [
    Handle(vendor_types=("cudaStream_t", "hipStream_t", "wwrStream_t"),
           capi_type="wwr_stream"),
    Handle(vendor_types=("cudaEvent_t", "hipEvent_t", "wwrEvent_t"),
           capi_type="wwr_event"),
    Handle(vendor_types=("cudaMemPool_t", "hipMemPool_t", "wwrMemPool_t"),
           capi_type="wwr_mem_pool"),
    Handle(vendor_types=("cudaGraph_t", "hipGraph_t", "wwrGraph_t"),
           capi_type="wwr_graph"),
    Handle(vendor_types=("cudaGraphExec_t", "hipGraphExec_t", "wwrGraphExec_t"),
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
    vendor_types=("cudaError_t", "hipError_t", "wwrError_t"),
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

# The by-value VALUE ENUMS the runtime surface forwards (#373, "bucket C"). Each
# is the §6 shape blas's Operation / FillMode already use: a stable capi enum
# with fixed ascending values the thunk maps capi<->neutral at the boundary,
# comparing ONLY the neutral wwr:: constants, so the one source compiles on both
# backends and the raw backend value (which can DIFFER -- DeviceP2PAttr is 1/2/3
# on CUDA, 0/1/2 on HIP) never crosses the boundary. Unlike the status EnumMap
# above (returned one way, neutral -> stable), these cross as a parameter
# (capi -> neutral), so they are a ValueEnum, not an EnumMap.
#
# Each entries pair is (capi_enumerator, neutral wwr constant); every neutral
# constant is a WWR_RT_VALUE entry the fragment already carries (so the generator
# cross-checks its value against both manifests), and the neutral wwr_type is a
# WWR_RT_TYPE the fragment declares. A vendor enum whose neutral TYPE or whose
# enumerator set the fragment does NOT carry (cudaDeviceAttr -- no fragment
# constants; cudaLaunchAttributeID -- its functions also take the
# cudaLaunchAttributeValue union, inexpressible) stays a logged skip (bucket D).

MEMCPY_KIND = ValueEnum(
    vendor_types=("cudaMemcpyKind", "hipMemcpyKind", "wwrMemcpyKind"),
    capi_type="wwr_memcpy_kind",
    wwr_type="wwrMemcpyKind",
    fallback="WWR_MEMCPY_DEFAULT",
    entries=(
        ("WWR_MEMCPY_HOST_TO_HOST", "wwrMemcpyHostToHost"),
        ("WWR_MEMCPY_HOST_TO_DEVICE", "wwrMemcpyHostToDevice"),
        ("WWR_MEMCPY_DEVICE_TO_HOST", "wwrMemcpyDeviceToHost"),
        ("WWR_MEMCPY_DEVICE_TO_DEVICE", "wwrMemcpyDeviceToDevice"),
        ("WWR_MEMCPY_DEFAULT", "wwrMemcpyDefault"),
    ),
)

LIMIT = ValueEnum(
    vendor_types=("cudaLimit", "hipLimit_t", "wwrLimit"),
    capi_type="wwr_limit",
    wwr_type="wwrLimit",
    fallback="WWR_LIMIT_STACK_SIZE",
    entries=(
        ("WWR_LIMIT_STACK_SIZE", "wwrLimitStackSize"),
        ("WWR_LIMIT_PRINTF_FIFO_SIZE", "wwrLimitPrintfFifoSize"),
        ("WWR_LIMIT_MALLOC_HEAP_SIZE", "wwrLimitMallocHeapSize"),
    ),
)

FUNC_CACHE = ValueEnum(
    vendor_types=("cudaFuncCache", "hipFuncCache_t", "wwrFuncCache"),
    capi_type="wwr_func_cache",
    wwr_type="wwrFuncCache",
    fallback="WWR_FUNC_CACHE_PREFER_NONE",
    entries=(
        ("WWR_FUNC_CACHE_PREFER_NONE", "wwrFuncCachePreferNone"),
        ("WWR_FUNC_CACHE_PREFER_SHARED", "wwrFuncCachePreferShared"),
        ("WWR_FUNC_CACHE_PREFER_L1", "wwrFuncCachePreferL1"),
        ("WWR_FUNC_CACHE_PREFER_EQUAL", "wwrFuncCachePreferEqual"),
    ),
)

FUNC_ATTRIBUTE = ValueEnum(
    vendor_types=("cudaFuncAttribute", "hipFuncAttribute", "wwrFuncAttribute"),
    capi_type="wwr_func_attribute",
    wwr_type="wwrFuncAttribute",
    fallback="WWR_FUNC_ATTRIBUTE_MAX",
    entries=(
        ("WWR_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE",
         "wwrFuncAttributeMaxDynamicSharedMemorySize"),
        ("WWR_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
         "wwrFuncAttributePreferredSharedMemoryCarveout"),
        ("WWR_FUNC_ATTRIBUTE_MAX", "wwrFuncAttributeMax"),
    ),
)

MEM_POOL_ATTR = ValueEnum(
    vendor_types=("cudaMemPoolAttr", "hipMemPoolAttr", "wwrMemPoolAttr"),
    capi_type="wwr_mem_pool_attr",
    wwr_type="wwrMemPoolAttr",
    fallback="WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD",
    entries=(
        ("WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD",
         "wwrMemPoolAttrReleaseThreshold"),
        ("WWR_MEM_POOL_ATTR_RESERVED_MEM_CURRENT",
         "wwrMemPoolAttrReservedMemCurrent"),
        ("WWR_MEM_POOL_ATTR_RESERVED_MEM_HIGH",
         "wwrMemPoolAttrReservedMemHigh"),
        ("WWR_MEM_POOL_ATTR_USED_MEM_CURRENT", "wwrMemPoolAttrUsedMemCurrent"),
        ("WWR_MEM_POOL_ATTR_USED_MEM_HIGH", "wwrMemPoolAttrUsedMemHigh"),
    ),
)

MEM_RANGE_ATTRIBUTE = ValueEnum(
    vendor_types=("cudaMemRangeAttribute", "hipMemRangeAttribute",
                  "wwrMemRangeAttribute"),
    capi_type="wwr_mem_range_attribute",
    wwr_type="wwrMemRangeAttribute",
    fallback="WWR_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
    entries=(
        ("WWR_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
         "wwrMemRangeAttributeReadMostly"),
        ("WWR_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
         "wwrMemRangeAttributePreferredLocation"),
        ("WWR_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
         "wwrMemRangeAttributeAccessedBy"),
        ("WWR_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
         "wwrMemRangeAttributeLastPrefetchLocation"),
    ),
)

MEM_ALLOCATION_HANDLE_TYPE = ValueEnum(
    vendor_types=("cudaMemAllocationHandleType", "hipMemAllocationHandleType",
                  "wwrMemAllocationHandleType"),
    capi_type="wwr_mem_allocation_handle_type",
    wwr_type="wwrMemAllocationHandleType",
    fallback="WWR_MEM_HANDLE_TYPE_NONE",
    entries=(
        ("WWR_MEM_HANDLE_TYPE_NONE", "wwrMemHandleTypeNone"),
        ("WWR_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR",
         "wwrMemHandleTypePosixFileDescriptor"),
        ("WWR_MEM_HANDLE_TYPE_WIN32", "wwrMemHandleTypeWin32"),
        ("WWR_MEM_HANDLE_TYPE_WIN32_KMT", "wwrMemHandleTypeWin32Kmt"),
    ),
)

STREAM_CAPTURE_MODE = ValueEnum(
    vendor_types=("cudaStreamCaptureMode", "hipStreamCaptureMode",
                  "wwrStreamCaptureMode"),
    capi_type="wwr_stream_capture_mode",
    wwr_type="wwrStreamCaptureMode",
    fallback="WWR_STREAM_CAPTURE_MODE_GLOBAL",
    entries=(
        ("WWR_STREAM_CAPTURE_MODE_GLOBAL", "wwrStreamCaptureModeGlobal"),
        ("WWR_STREAM_CAPTURE_MODE_THREAD_LOCAL",
         "wwrStreamCaptureModeThreadLocal"),
        ("WWR_STREAM_CAPTURE_MODE_RELAXED", "wwrStreamCaptureModeRelaxed"),
    ),
)

DEVICE_P2P_ATTR = ValueEnum(
    vendor_types=("cudaDeviceP2PAttr", "hipDeviceP2PAttr", "wwrDeviceP2PAttr"),
    capi_type="wwr_device_p2p_attr",
    wwr_type="wwrDeviceP2PAttr",
    fallback="WWR_DEV_P2P_ATTR_PERFORMANCE_RANK",
    entries=(
        ("WWR_DEV_P2P_ATTR_PERFORMANCE_RANK", "wwrDevP2PAttrPerformanceRank"),
        ("WWR_DEV_P2P_ATTR_ACCESS_SUPPORTED", "wwrDevP2PAttrAccessSupported"),
        ("WWR_DEV_P2P_ATTR_NATIVE_ATOMIC_SUPPORTED",
         "wwrDevP2PAttrNativeAtomicSupported"),
    ),
)

VALUE_ENUMS = [MEMCPY_KIND, LIMIT, FUNC_CACHE, FUNC_ATTRIBUTE, MEM_POOL_ATTR,
               MEM_RANGE_ATTRIBUTE, MEM_ALLOCATION_HANDLE_TYPE,
               STREAM_CAPTURE_MODE, DEVICE_P2P_ATTR]
