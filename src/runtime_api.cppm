/**
 * @file runtime_api.cppm
 * @brief Backend-neutral GPU runtime API: wwr* names for cuda* / hip*
 *
 * Exports wwr-prefixed aliases of the CUDA runtime API (cuda_runtime_api.h)
 * or the HIP runtime API (hip_runtime_api.h), whichever backend this build is
 * configured for. See backend.h for the switch.
 *
 * Only the names src/wrappers uses are listed. Add a name here, once, when
 * code above this layer needs it; a name that differs between the backends
 * beyond the cuda/hip prefix gets its own explicit #if block.
 *
 * Usage:
 *   import wwr.runtime_api;
 *
 *   wwrStream_t stream;
 *   if (wwrStreamCreate(&stream) != wwrSuccess) { ... }
 */

module;

#include "backend.h"

// Runtime API: wwrX -> cudaX / hipX
#define WWR_RT_TYPE(x) WWR_TYPE(wwr##x, cuda##x, hip##x)
#define WWR_RT_VALUE(x) WWR_VALUE(wwr##x, cuda##x, hip##x)
#define WWR_RT_FUNCTION(x) WWR_FUNCTION(wwr##x, cuda##x, hip##x)

export module wwr.runtime_api;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_runtime_api;
#else
import wwr.hip.hip_runtime_api;
#endif
import std;

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_RT_TYPE(Error_t)

// Device properties struct. Spelled out rather than WWR_RT_TYPE(DeviceProp)
// because the HIP name differs beyond the cuda/hip prefix: hip_runtime_api.h
// version-renames hipDeviceProp_t to hipDeviceProp_tR0600, and that macro is
// not in scope here (this unit imports the HIP module, not its header), so the
// versioned spelling is what the module actually exports.
WWR_TYPE(wwrDeviceProp, cudaDeviceProp, hipDeviceProp_tR0600)

WWR_RT_TYPE(Stream_t)
WWR_RT_TYPE(StreamCaptureMode)
WWR_RT_TYPE(Event_t)
WWR_RT_TYPE(MemPool_t)
WWR_RT_TYPE(MemPoolProps)
WWR_RT_TYPE(MemcpyKind)
WWR_RT_TYPE(Graph_t)
WWR_RT_TYPE(GraphExec_t)

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
// (its hipHostAlloc* macros are equal-valued aliases the raw module does not
// re-export).
WWR_VALUE(wwrHostAllocDefault, cudaHostAllocDefault, hipHostMallocDefault)
WWR_VALUE(wwrHostAllocMapped, cudaHostAllocMapped, hipHostMallocMapped)
WWR_VALUE(wwrHostAllocWriteCombined, cudaHostAllocWriteCombined, hipHostMallocWriteCombined)

// Managed memory attach flags
WWR_RT_VALUE(MemAttachGlobal)
WWR_RT_VALUE(MemAttachHost)

// ========================================================================
// Functions
// ========================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwr*
// below is a deliberate constexpr reference to the selected backend's entry
// point (via WWR_FUNCTION or a hand-written overload binding). A reference to
// a vendor function has no const form, so the check cannot be satisfied without
// abandoning the alias pattern -- see backend.h.

// Errors
WWR_RT_FUNCTION(GetErrorName)
WWR_RT_FUNCTION(GetErrorString)
WWR_RT_FUNCTION(GetLastError)

// Device
WWR_RT_FUNCTION(GetDevice)
WWR_RT_FUNCTION(SetDevice)
// Like wwrDeviceProp above, HIP version-renames the entry point
// (hipGetDeviceProperties -> hipGetDevicePropertiesR0600), so spell it out.
WWR_FUNCTION(wwrGetDeviceProperties, cudaGetDeviceProperties, hipGetDevicePropertiesR0600)

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
// that signature. A WWR_FUNCTION reference cannot do this: it can bind neither
// two differently-named backend functions nor a default argument. See
// docs/architecture.md section 4.
inline wwrError_t wwrGraphInstantiate(wwrGraphExec_t *exec, wwrGraph_t graph,
                                      unsigned long long flags = 0) {
  return WWR_SELECT(cudaGraphInstantiate, hipGraphInstantiateWithFlags)(exec, graph, flags);
}

// Device memory
//
// hip_runtime_api.h overloads hipMalloc with a template<class T>
// hipMalloc(T**, size_t), so a plain function reference cannot name it; bind
// the void** overload explicitly (on both backends, for one signature).
inline constexpr wwrError_t (&wwrMalloc)(void **, std::size_t) = WWR_SELECT(cudaMalloc,
                                                                               hipMalloc);
WWR_RT_FUNCTION(Free)
WWR_RT_FUNCTION(Memcpy)
WWR_RT_FUNCTION(Memset)
WWR_RT_FUNCTION(DeviceSynchronize)

// Stream-ordered device memory
WWR_RT_FUNCTION(MallocAsync)
WWR_RT_FUNCTION(MallocFromPoolAsync)
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
                                            unsigned int) = WWR_SELECT(cudaHostAlloc,
                                                                          hipHostAlloc);
WWR_RT_FUNCTION(FreeHost)

// Managed memory -- hipMallocManaged is an overload set too.
inline constexpr wwrError_t (&wwrMallocManaged)(void **, std::size_t,
                                                unsigned int) = WWR_SELECT(cudaMallocManaged,
                                                                              hipMallocManaged);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace wwr
