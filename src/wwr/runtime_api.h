/**
 * @file wwr/runtime_api.h
 * @brief The backend-neutral runtime API as a single, self-contained HOST
 *        #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.runtime_api;`: a plain .cpp that
 * #includes this one header gets the same wwr* runtime surface the module
 * exports -- wwrError_t, wwrMalloc, wwrStreamCreate, wwrSuccess, ... -- bound to
 * the selected backend's ::cuda* / ::hip* entry points.
 *
 * ONE file, deliberately. The wwr.runtime_api MODULE is split (a vendor-include
 * + flag-dance part in its global module fragment, the name bindings in its
 * exported purview) because C++ modules force that: names declared in a GMF are
 * not exported, and the dance's internal-linkage flag constants cannot be
 * exported at all. A plain header has no such split to make -- like the vendor
 * headers it wraps (cuda_runtime_api.h / hip_runtime_api.h are monolithic), it
 * does the vendor include, the allocation-flag dance and the wwr* bindings top to
 * bottom. The cost is that the name list here and the module's
 * runtime_api_surface.h are two copies kept in step by hand -- the price of a
 * standalone, module-independent header path.
 *
 * Backend selection is the one thing it shares: selected_backend.h, the
 * project-wide switch (not a surface fragment -- the primitive every path uses).
 * The include root, the WWR_GPU_BACKEND_* define it reads in a host compile, and
 * the runtime link library arrive by linking wwr::runtime_api::host. HOST only: a
 * device .cu/.cuh gets this surface from runtime.h's device section (which binds
 * the forwarding-template form the full vendor runtime header forces). See
 * src/runtime_api.cppm, src/runtime.h and docs/architecture.md, section 3.
 */

#pragma once

// HOST only -- see the file header. A device pass gets the surface from runtime.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/runtime_api.h is the HOST #include path; a device .cu/.cuh gets the runtime surface from runtime.h. See src/README.md."
#endif

#include <cstddef> // std::size_t for the surface's hand-written forwarders

// ---------------------------------------------------------------------------
// Vendor include + allocation-flag dance (selected_backend.h + the vendor api
// header, with the flag macros #undef'd to equal-valued global constexprs so the
// ::-prefixed bindings below resolve). Same content the module's
// runtime_api.h fragment carries -- restated here, not shared, per the file header.
// ---------------------------------------------------------------------------
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime_api.h>

// Validate each allocation flag's macro value before #undef-ing it, so a vendor
// value drift is caught here rather than silently baked into the module's
// surface. See this file's header for why these -- and only these -- need it.
static_assert(cudaHostAllocDefault == 0x00, "cudaHostAllocDefault value mismatch");
static_assert(cudaHostAllocMapped == 0x02, "cudaHostAllocMapped value mismatch");
static_assert(cudaHostAllocWriteCombined == 0x04, "cudaHostAllocWriteCombined value mismatch");
static_assert(cudaEventDefault == 0x00, "cudaEventDefault value mismatch");
static_assert(cudaEventBlockingSync == 0x01, "cudaEventBlockingSync value mismatch");
static_assert(cudaEventDisableTiming == 0x02, "cudaEventDisableTiming value mismatch");
static_assert(cudaStreamDefault == 0x00, "cudaStreamDefault value mismatch");
static_assert(cudaStreamNonBlocking == 0x01, "cudaStreamNonBlocking value mismatch");
static_assert(cudaMemAttachGlobal == 0x01, "cudaMemAttachGlobal value mismatch");
static_assert(cudaMemAttachHost == 0x02, "cudaMemAttachHost value mismatch");
static_assert(cudaArrayDefault == 0x00, "cudaArrayDefault value mismatch");
static_assert(cudaArraySurfaceLoadStore == 0x02, "cudaArraySurfaceLoadStore value mismatch");

#undef cudaHostAllocDefault
#undef cudaHostAllocMapped
#undef cudaHostAllocWriteCombined
#undef cudaEventDefault
#undef cudaEventBlockingSync
#undef cudaEventDisableTiming
#undef cudaStreamDefault
#undef cudaStreamNonBlocking
#undef cudaMemAttachGlobal
#undef cudaMemAttachHost
#undef cudaArrayDefault
#undef cudaArraySurfaceLoadStore

// Global scope, not namespace wwr: runtime_api.cppm's WWR_VALUE_RAW bindings
// name ::cuda<Flag>, so the replacements must sit where that resolves. constexpr
// gives them internal linkage -- private to this one TU, never a module export.
constexpr unsigned int cudaHostAllocDefault = 0x00;
constexpr unsigned int cudaHostAllocMapped = 0x02;
constexpr unsigned int cudaHostAllocWriteCombined = 0x04;
constexpr unsigned int cudaEventDefault = 0x00;
constexpr unsigned int cudaEventBlockingSync = 0x01;
constexpr unsigned int cudaEventDisableTiming = 0x02;
constexpr unsigned int cudaStreamDefault = 0x00;
constexpr unsigned int cudaStreamNonBlocking = 0x01;
constexpr unsigned int cudaMemAttachGlobal = 0x01;
constexpr unsigned int cudaMemAttachHost = 0x02;
constexpr unsigned int cudaArrayDefault = 0x00;
constexpr unsigned int cudaArraySurfaceLoadStore = 0x02;

#else

// __HIP_DISABLE_CPP_FUNCTIONS__ before the vendor header, as the raw
// wwr.hip.hip_runtime_api module's global module fragment does: it keeps the
// HIP headers from pulling in the C++ helper overloads a module cannot export.
#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
#include <hip/hip_runtime_api.h>

// The HIP counterparts of the CUDA block above. HIP spells the pinned-host
// family hipHostMalloc* (its hipHostAlloc* names are equal-valued aliases);
// wwrHostAlloc* in runtime_api.cppm bind to these. Values verified against
// hip/hip_runtime_api.h independently of CUDA's, though they happen to agree.
static_assert(hipHostMallocDefault == 0x0, "hipHostMallocDefault value mismatch");
static_assert(hipHostMallocMapped == 0x2, "hipHostMallocMapped value mismatch");
static_assert(hipHostMallocWriteCombined == 0x4, "hipHostMallocWriteCombined value mismatch");
static_assert(hipEventDefault == 0x0, "hipEventDefault value mismatch");
static_assert(hipEventBlockingSync == 0x1, "hipEventBlockingSync value mismatch");
static_assert(hipEventDisableTiming == 0x2, "hipEventDisableTiming value mismatch");
static_assert(hipStreamDefault == 0x00, "hipStreamDefault value mismatch");
static_assert(hipStreamNonBlocking == 0x01, "hipStreamNonBlocking value mismatch");
static_assert(hipMemAttachGlobal == 0x01, "hipMemAttachGlobal value mismatch");
static_assert(hipMemAttachHost == 0x02, "hipMemAttachHost value mismatch");
static_assert(hipArrayDefault == 0x00, "hipArrayDefault value mismatch");
static_assert(hipArraySurfaceLoadStore == 0x02, "hipArraySurfaceLoadStore value mismatch");

#undef hipHostMallocDefault
#undef hipHostMallocMapped
#undef hipHostMallocWriteCombined
#undef hipEventDefault
#undef hipEventBlockingSync
#undef hipEventDisableTiming
#undef hipStreamDefault
#undef hipStreamNonBlocking
#undef hipMemAttachGlobal
#undef hipMemAttachHost
#undef hipArrayDefault
#undef hipArraySurfaceLoadStore

constexpr unsigned int hipHostMallocDefault = 0x0;
constexpr unsigned int hipHostMallocMapped = 0x2;
constexpr unsigned int hipHostMallocWriteCombined = 0x4;
constexpr unsigned int hipEventDefault = 0x0;
constexpr unsigned int hipEventBlockingSync = 0x1;
constexpr unsigned int hipEventDisableTiming = 0x2;
constexpr unsigned int hipStreamDefault = 0x00;
constexpr unsigned int hipStreamNonBlocking = 0x01;
constexpr unsigned int hipMemAttachGlobal = 0x01;
constexpr unsigned int hipMemAttachHost = 0x02;
constexpr unsigned int hipArrayDefault = 0x00;
constexpr unsigned int hipArraySurfaceLoadStore = 0x02;

#endif

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- valid because only the MINIMAL api
// header above is in scope, whose entry points are not overloaded. Keyed on
// WWR_SELECTED_*, set by selected_backend.h above. #undef'd at the end so none
// leaks into the including TU.
// ---------------------------------------------------------------------------
#if defined(WWR_SELECTED_CUDA)
#define WWR_SELECT_RAW(cuda_name, hip_name) ::cuda_name
#else
#define WWR_SELECT_RAW(cuda_name, hip_name) ::hip_name
#endif
#define WWR_TYPE_RAW(wwr_name, cuda_name, hip_name)                                                 \
  using wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_VALUE_RAW(wwr_name, cuda_name, hip_name)                                                \
  inline constexpr auto wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_FUNCTION_RAW(wwr_name, cuda_name, hip_name)                                             \
  inline constexpr auto &wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_RT_TYPE(x) WWR_TYPE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_VALUE(x) WWR_VALUE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_FUNCTION(x) WWR_FUNCTION_RAW(wwr##x, cuda##x, hip##x)

// ---------------------------------------------------------------------------
// The wwr* runtime surface -- types, constants, functions and the hand-written
// forwarders. Restated here (the module keeps its own copy in
// runtime_api_surface.h); keep the two in step.
// ---------------------------------------------------------------------------
namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_RT_TYPE(Error_t)

// Device properties struct. Spelled out rather than WWR_RT_TYPE(DeviceProp)
// because the HIP name differs beyond the cuda/hip prefix: hip_runtime_api.h
// version-renames hipDeviceProp_t to hipDeviceProp_tR0600 (hipDeviceProp_t is
// its own alias macro), so the real versioned struct is what we bind to.
WWR_TYPE_RAW(wwrDeviceProp, cudaDeviceProp, hipDeviceProp_tR0600)

// wwrStream_t is ::cudaStream_t / ::hipStream_t here (from the vendor header),
// the same vendor handle runtime.h's always-on section names for device and GMF
// code -- a redefinition to that same alias, which is well-formed.
WWR_RT_TYPE(Stream_t)
WWR_RT_TYPE(StreamCaptureMode)
WWR_RT_TYPE(Event_t)
WWR_RT_TYPE(MemPool_t)
WWR_RT_TYPE(MemPoolProps)
WWR_RT_TYPE(MemcpyKind)
WWR_RT_TYPE(Graph_t)
WWR_RT_TYPE(GraphExec_t)

// Texture and surface objects. On CUDA the object handles are 64-bit integers
// with no reserved invalid value (uint64); on HIP they are pointers. The RAII
// wrappers in wwr.extension.{texture,surface} ride that difference through
// BaseHandle's two liveness paths. wwrArray_t is the array backing a surface
// (and a valid texture source); the descriptor structs and wwrChannelFormatDesc
// are what the create calls take.
WWR_RT_TYPE(TextureObject_t)
WWR_RT_TYPE(SurfaceObject_t)
WWR_RT_TYPE(ResourceDesc)
WWR_RT_TYPE(TextureDesc)
WWR_RT_TYPE(ResourceViewDesc)
WWR_RT_TYPE(ChannelFormatDesc)
WWR_RT_TYPE(ChannelFormatKind)
WWR_RT_TYPE(ResourceType)
WWR_RT_TYPE(Array_t)

// ========================================================================
// Constants
// ========================================================================

WWR_RT_VALUE(Success)
WWR_RT_VALUE(ErrorInvalidValue)

WWR_RT_VALUE(StreamDefault)
WWR_RT_VALUE(StreamNonBlocking)

WWR_RT_VALUE(StreamCaptureModeGlobal)
WWR_RT_VALUE(StreamCaptureModeThreadLocal)
WWR_RT_VALUE(StreamCaptureModeRelaxed)

WWR_RT_VALUE(EventDefault)
WWR_RT_VALUE(EventBlockingSync)
WWR_RT_VALUE(EventDisableTiming)

WWR_RT_VALUE(MemAllocationTypePinned)
WWR_RT_VALUE(MemHandleTypeNone)
WWR_RT_VALUE(MemLocationTypeDevice)
WWR_RT_VALUE(MemPoolAttrReleaseThreshold)

WWR_RT_VALUE(MemcpyHostToDevice)
WWR_RT_VALUE(MemcpyDeviceToHost)
WWR_RT_VALUE(MemcpyDeviceToDevice)
WWR_RT_VALUE(MemcpyDefault)

// Pinned host allocation flags. hip_runtime_api.h spells these hipHostMalloc*
// (its hipHostAlloc* macros are equal-valued aliases); runtime_api.h captures
// the hipHostMalloc* family as the global constexprs these bind to.
WWR_VALUE_RAW(wwrHostAllocDefault, cudaHostAllocDefault, hipHostMallocDefault)
WWR_VALUE_RAW(wwrHostAllocMapped, cudaHostAllocMapped, hipHostMallocMapped)
WWR_VALUE_RAW(wwrHostAllocWriteCombined, cudaHostAllocWriteCombined, hipHostMallocWriteCombined)

// Managed memory attach flags
WWR_RT_VALUE(MemAttachGlobal)
WWR_RT_VALUE(MemAttachHost)

// Channel-format kinds -- the element interpretation in a wwrChannelFormatDesc
WWR_RT_VALUE(ChannelFormatKindSigned)
WWR_RT_VALUE(ChannelFormatKindUnsigned)
WWR_RT_VALUE(ChannelFormatKindFloat)

// Resource kinds -- which arm of a wwrResourceDesc is populated
WWR_RT_VALUE(ResourceTypeArray)
WWR_RT_VALUE(ResourceTypeLinear)

// Array allocation flags. wwrArraySurfaceLoadStore must be set when allocating
// the array a surface object binds to (see the vendor modules' constexpr
// wrappers -- the backends spell these as macros).
WWR_RT_VALUE(ArrayDefault)
WWR_RT_VALUE(ArraySurfaceLoadStore)

// ========================================================================
// Functions
// ========================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwr*
// below is a deliberate constexpr reference to the selected backend's entry
// point (via WWR_FUNCTION_RAW or a hand-written overload binding). A reference
// to a vendor function has no const form, so the check cannot be satisfied
// without abandoning the alias pattern -- see backend.h.

// Errors
WWR_RT_FUNCTION(GetErrorName)
WWR_RT_FUNCTION(GetErrorString)
WWR_RT_FUNCTION(GetLastError)

// Device
WWR_RT_FUNCTION(GetDevice)
WWR_RT_FUNCTION(SetDevice)
// Like wwrDeviceProp above, HIP version-renames the entry point
// (hipGetDeviceProperties -> hipGetDevicePropertiesR0600), so spell it out.
WWR_FUNCTION_RAW(wwrGetDeviceProperties, cudaGetDeviceProperties, hipGetDevicePropertiesR0600)

// Streams
WWR_RT_FUNCTION(StreamCreate)
WWR_RT_FUNCTION(StreamCreateWithFlags)
WWR_RT_FUNCTION(StreamCreateWithPriority)
WWR_RT_FUNCTION(StreamGetFlags)
WWR_RT_FUNCTION(StreamWaitEvent)
WWR_RT_FUNCTION(StreamSynchronize)
WWR_RT_FUNCTION(StreamDestroy)
WWR_RT_FUNCTION(StreamBeginCapture)
WWR_RT_FUNCTION(StreamEndCapture)

// Events
WWR_RT_FUNCTION(EventCreate)
WWR_RT_FUNCTION(EventCreateWithFlags)
WWR_RT_FUNCTION(EventRecord)
WWR_RT_FUNCTION(EventRecordWithFlags)
WWR_RT_FUNCTION(EventSynchronize)
WWR_RT_FUNCTION(EventQuery)
WWR_RT_FUNCTION(EventDestroy)

// Stream-ordered memory pools
WWR_RT_FUNCTION(MemPoolCreate)
WWR_RT_FUNCTION(MemPoolSetAttribute)
WWR_RT_FUNCTION(MemPoolDestroy)

// Graphs
WWR_RT_FUNCTION(GraphCreate)
WWR_RT_FUNCTION(GraphDestroy)
WWR_RT_FUNCTION(GraphExecDestroy)
WWR_RT_FUNCTION(GraphLaunch)
WWR_RT_FUNCTION(GraphUpload)

// wwrGraphInstantiate is a hand-written forwarding function, not a
// WWR_RT_FUNCTION reference, because the backends' plain *Instantiate entry
// points disagree on signature beyond the cuda/hip prefix:
//   CUDA 12+: cudaGraphInstantiate(GraphExec_t*, Graph_t, unsigned long long flags)
//   HIP:      hipGraphInstantiate(GraphExec_t*, Graph_t,
//                                 GraphNode_t* errorNode, char* logBuffer, size_t)
// Their flags-taking spellings agree -- cudaGraphInstantiate itself on CUDA and
// hipGraphInstantiateWithFlags on HIP both take
// (GraphExec_t*, Graph_t, unsigned long long) -- so forward to whichever carries
// that signature. A WWR_FUNCTION_RAW reference cannot do this: it can bind
// neither two differently-named backend functions nor a default argument. See
// docs/architecture.md section 4.
inline wwrError_t wwrGraphInstantiate(wwrGraphExec_t *exec, wwrGraph_t graph,
                                      unsigned long long flags = 0) {
  return WWR_SELECT_RAW(cudaGraphInstantiate, hipGraphInstantiateWithFlags)(exec, graph, flags);
}

// Device memory
//
// hip_runtime_api.h overloads hipMalloc with a template<class T>
// hipMalloc(T**, size_t), so a plain function reference cannot name it; bind
// the void** overload explicitly (on both backends, for one signature).
inline constexpr wwrError_t (&wwrMalloc)(void **, std::size_t) = WWR_SELECT_RAW(cudaMalloc,
                                                                               hipMalloc);
WWR_RT_FUNCTION(Free)
WWR_RT_FUNCTION(Memcpy)
WWR_RT_FUNCTION(Memset)
WWR_RT_FUNCTION(DeviceSynchronize)

// Stream-ordered device memory
//
// wwrMallocAsync / wwrMallocFromPoolAsync are hand-written forwarders, not
// WWR_RT_FUNCTION reference bindings, because hipMallocAsync /
// hipMallocFromPoolAsync are overload sets: each carries a template<class T>
// (T**, ...) convenience overload that -- unlike hipMalloc's, which sits in a
// __HIP_DISABLE_CPP_FUNCTIONS__-guarded block -- is in a plain `#if __cplusplus`
// block the disable macro does NOT reach, so the template survives beside the
// extern "C" void** entry point. A reference cannot cleanly alias one member of
// that set (the two live in different namespaces here than in the raw module, so
// an address-identity alias does not even hold), so forward instead -- the
// wwrGraphInstantiate shape. The void** argument makes the call pick the C
// overload; cudaMalloc*Async are single functions and resolve the same way.
inline wwrError_t wwrMallocAsync(void **ptr, std::size_t size, wwrStream_t stream) {
  return WWR_SELECT_RAW(cudaMallocAsync, hipMallocAsync)(ptr, size, stream);
}
inline wwrError_t wwrMallocFromPoolAsync(void **ptr, std::size_t size, wwrMemPool_t pool,
                                         wwrStream_t stream) {
  return WWR_SELECT_RAW(cudaMallocFromPoolAsync, hipMallocFromPoolAsync)(ptr, size, pool, stream);
}
WWR_RT_FUNCTION(FreeAsync)
WWR_RT_FUNCTION(MemcpyAsync)
WWR_RT_FUNCTION(MemsetAsync)

// Pinned host memory
//
// hipHostAlloc is an overload set like hipMalloc (plus a template<class T>
// hipHostAlloc(T**, size_t, unsigned)); bind the void** overload. There is no
// wwrMallocHost: hipMallocHost is deprecated, and cudaMallocHost is documented
// as cudaHostAlloc with cudaHostAllocDefault, which is what callers write.
inline constexpr wwrError_t (&wwrHostAlloc)(void **, std::size_t,
                                            unsigned int) = WWR_SELECT_RAW(cudaHostAlloc,
                                                                          hipHostAlloc);
WWR_RT_FUNCTION(FreeHost)

// Managed memory -- hipMallocManaged is an overload set too.
inline constexpr wwrError_t (&wwrMallocManaged)(void **, std::size_t,
                                                unsigned int) = WWR_SELECT_RAW(cudaMallocManaged,
                                                                              hipMallocManaged);

// CUDA array allocation -- the backing store a surface object binds to, and a
// valid texture source. Both are plain (non-overloaded) functions, so a
// WWR_FUNCTION reference names them directly; the reference carries no default
// arguments, so callers pass width, height and flags explicitly.
WWR_RT_FUNCTION(MallocArray)
WWR_RT_FUNCTION(FreeArray)

// Texture and surface objects
WWR_RT_FUNCTION(CreateTextureObject)
WWR_RT_FUNCTION(DestroyTextureObject)
WWR_RT_FUNCTION(CreateSurfaceObject)
WWR_RT_FUNCTION(DestroySurfaceObject)
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
#undef WWR_RT_TYPE
#undef WWR_RT_VALUE
#undef WWR_RT_FUNCTION
