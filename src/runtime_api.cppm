/**
 * @file runtime_api.cppm
 * @brief Backend-neutral GPU runtime API: gpu* names for cuda* / hip*
 *
 * Exports gpu-prefixed aliases of the CUDA runtime API (cuda_runtime_api.h)
 * or the HIP runtime API (hip_runtime_api.h), whichever backend this build is
 * configured for. See gpu_backend.h for the switch.
 *
 * Only the names src/wrappers uses are listed. Add a name here, once, when
 * code above this layer needs it; a name that differs between the backends
 * beyond the cuda/hip prefix gets its own explicit #if block.
 *
 * Usage:
 *   import gpumod.runtime_api;
 *
 *   gpuStream_t stream;
 *   if (gpuStreamCreate(&stream) != gpuSuccess) { ... }
 */

module;

#include "gpu_backend.h"

// Runtime API: gpuX -> cudaX / hipX
#define GPUMOD_RT_TYPE(x) GPUMOD_TYPE(gpu##x, cuda##x, hip##x)
#define GPUMOD_RT_VALUE(x) GPUMOD_VALUE(gpu##x, cuda##x, hip##x)
#define GPUMOD_RT_FUNCTION(x) GPUMOD_FUNCTION(gpu##x, cuda##x, hip##x)

export module gpumod.runtime_api;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_runtime_api;
#else
import gpumod.hip.hip_runtime_api;
#endif
import std;

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_RT_TYPE(Error_t)

// Device properties struct. Spelled out rather than GPUMOD_RT_TYPE(DeviceProp)
// because the HIP name differs beyond the cuda/hip prefix: hip_runtime_api.h
// version-renames hipDeviceProp_t to hipDeviceProp_tR0600, and that macro is
// not in scope here (this unit imports the HIP module, not its header), so the
// versioned spelling is what the module actually exports.
GPUMOD_TYPE(gpuDeviceProp, cudaDeviceProp, hipDeviceProp_tR0600)

GPUMOD_RT_TYPE(Stream_t)
GPUMOD_RT_TYPE(StreamCaptureMode)
GPUMOD_RT_TYPE(Event_t)
GPUMOD_RT_TYPE(MemPool_t)
GPUMOD_RT_TYPE(MemPoolProps)
GPUMOD_RT_TYPE(MemcpyKind)
GPUMOD_RT_TYPE(Graph_t)
GPUMOD_RT_TYPE(GraphExec_t)

// ========================================================================
// Constants
// ========================================================================

GPUMOD_RT_VALUE(Success)
GPUMOD_RT_VALUE(ErrorInvalidValue)

GPUMOD_RT_VALUE(StreamDefault)
GPUMOD_RT_VALUE(StreamNonBlocking)

GPUMOD_RT_VALUE(StreamCaptureModeGlobal)
GPUMOD_RT_VALUE(StreamCaptureModeThreadLocal)
GPUMOD_RT_VALUE(StreamCaptureModeRelaxed)

GPUMOD_RT_VALUE(EventDefault)
GPUMOD_RT_VALUE(EventBlockingSync)
GPUMOD_RT_VALUE(EventDisableTiming)

GPUMOD_RT_VALUE(MemAllocationTypePinned)
GPUMOD_RT_VALUE(MemHandleTypeNone)
GPUMOD_RT_VALUE(MemLocationTypeDevice)
GPUMOD_RT_VALUE(MemPoolAttrReleaseThreshold)

GPUMOD_RT_VALUE(MemcpyHostToDevice)
GPUMOD_RT_VALUE(MemcpyDeviceToHost)
GPUMOD_RT_VALUE(MemcpyDeviceToDevice)
GPUMOD_RT_VALUE(MemcpyDefault)

// Pinned host allocation flags. hip_runtime_api.h spells these hipHostMalloc*
// (its hipHostAlloc* macros are equal-valued aliases the raw module does not
// re-export).
GPUMOD_VALUE(gpuHostAllocDefault, cudaHostAllocDefault, hipHostMallocDefault)
GPUMOD_VALUE(gpuHostAllocMapped, cudaHostAllocMapped, hipHostMallocMapped)
GPUMOD_VALUE(gpuHostAllocWriteCombined, cudaHostAllocWriteCombined, hipHostMallocWriteCombined)

// Managed memory attach flags
GPUMOD_RT_VALUE(MemAttachGlobal)
GPUMOD_RT_VALUE(MemAttachHost)

// ========================================================================
// Functions
// ========================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each gpu*
// below is a deliberate constexpr reference to the selected backend's entry
// point (via GPUMOD_FUNCTION or a hand-written overload binding). A reference to
// a vendor function has no const form, so the check cannot be satisfied without
// abandoning the alias pattern -- see gpu_backend.h.

// Errors
GPUMOD_RT_FUNCTION(GetErrorName)
GPUMOD_RT_FUNCTION(GetErrorString)
GPUMOD_RT_FUNCTION(GetLastError)

// Device
GPUMOD_RT_FUNCTION(GetDevice)
GPUMOD_RT_FUNCTION(SetDevice)
// Like gpuDeviceProp above, HIP version-renames the entry point
// (hipGetDeviceProperties -> hipGetDevicePropertiesR0600), so spell it out.
GPUMOD_FUNCTION(gpuGetDeviceProperties, cudaGetDeviceProperties, hipGetDevicePropertiesR0600)

// Streams
GPUMOD_RT_FUNCTION(StreamCreate)
GPUMOD_RT_FUNCTION(StreamCreateWithFlags)
GPUMOD_RT_FUNCTION(StreamCreateWithPriority)
GPUMOD_RT_FUNCTION(StreamWaitEvent)
GPUMOD_RT_FUNCTION(StreamSynchronize)
GPUMOD_RT_FUNCTION(StreamDestroy)
GPUMOD_RT_FUNCTION(StreamBeginCapture)
GPUMOD_RT_FUNCTION(StreamEndCapture)

// Events
GPUMOD_RT_FUNCTION(EventCreate)
GPUMOD_RT_FUNCTION(EventCreateWithFlags)
GPUMOD_RT_FUNCTION(EventRecord)
GPUMOD_RT_FUNCTION(EventRecordWithFlags)
GPUMOD_RT_FUNCTION(EventSynchronize)
GPUMOD_RT_FUNCTION(EventQuery)
GPUMOD_RT_FUNCTION(EventDestroy)

// Stream-ordered memory pools
GPUMOD_RT_FUNCTION(MemPoolCreate)
GPUMOD_RT_FUNCTION(MemPoolSetAttribute)
GPUMOD_RT_FUNCTION(MemPoolDestroy)

// Graphs
GPUMOD_RT_FUNCTION(GraphCreate)
GPUMOD_RT_FUNCTION(GraphDestroy)
GPUMOD_RT_FUNCTION(GraphExecDestroy)
GPUMOD_RT_FUNCTION(GraphLaunch)
GPUMOD_RT_FUNCTION(GraphUpload)

// gpuGraphInstantiate is a hand-written forwarding function, not a
// GPUMOD_RT_FUNCTION reference, because the backends' plain *Instantiate entry
// points disagree on signature beyond the cuda/hip prefix:
//   CUDA 12+: cudaGraphInstantiate(GraphExec_t*, Graph_t, unsigned long long flags)
//   HIP:      hipGraphInstantiate(GraphExec_t*, Graph_t,
//                                 GraphNode_t* errorNode, char* logBuffer, size_t)
// Their flags-taking spellings agree -- cudaGraphInstantiate itself on CUDA and
// hipGraphInstantiateWithFlags on HIP both take
// (GraphExec_t*, Graph_t, unsigned long long) -- so forward to whichever carries
// that signature. A GPUMOD_FUNCTION reference cannot do this: it can bind neither
// two differently-named backend functions nor a default argument. See
// docs/architecture.md section 4.
inline gpuError_t gpuGraphInstantiate(gpuGraphExec_t *exec, gpuGraph_t graph,
                                      unsigned long long flags = 0) {
  return GPUMOD_SELECT(cudaGraphInstantiate, hipGraphInstantiateWithFlags)(exec, graph, flags);
}

// Device memory
//
// hip_runtime_api.h overloads hipMalloc with a template<class T>
// hipMalloc(T**, size_t), so a plain function reference cannot name it; bind
// the void** overload explicitly (on both backends, for one signature).
inline constexpr gpuError_t (&gpuMalloc)(void **, std::size_t) = GPUMOD_SELECT(cudaMalloc,
                                                                               hipMalloc);
GPUMOD_RT_FUNCTION(Free)
GPUMOD_RT_FUNCTION(Memcpy)
GPUMOD_RT_FUNCTION(Memset)
GPUMOD_RT_FUNCTION(DeviceSynchronize)

// Stream-ordered device memory
GPUMOD_RT_FUNCTION(MallocAsync)
GPUMOD_RT_FUNCTION(MallocFromPoolAsync)
GPUMOD_RT_FUNCTION(FreeAsync)
GPUMOD_RT_FUNCTION(MemcpyAsync)
GPUMOD_RT_FUNCTION(MemsetAsync)

// Pinned host memory
//
// hipHostAlloc is an overload set like hipMalloc (plus a template<class T>
// hipHostAlloc(T**, size_t, unsigned)); bind the void** overload. There is no
// gpuMallocHost: hipMallocHost is deprecated, and cudaMallocHost is documented
// as cudaHostAlloc with cudaHostAllocDefault, which is what callers write.
inline constexpr gpuError_t (&gpuHostAlloc)(void **, std::size_t,
                                            unsigned int) = GPUMOD_SELECT(cudaHostAlloc,
                                                                          hipHostAlloc);
GPUMOD_RT_FUNCTION(FreeHost)

// Managed memory -- hipMallocManaged is an overload set too.
inline constexpr gpuError_t (&gpuMallocManaged)(void **, std::size_t,
                                                unsigned int) = GPUMOD_SELECT(cudaMallocManaged,
                                                                              hipMallocManaged);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace gpumod
