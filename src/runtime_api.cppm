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

// NB: this GMF must NOT #include runtime.h (or any vendor runtime header). The
// vendor's cuda_runtime_api.h / hip_runtime_api.h #define preprocessor macros
// (cudaStreamDefault, cudaEventDefault, cudaArrayDefault, ...) that collide with
// the cuda##x / hip##x tokens in the WWR_RT_VALUE expansions below. The wwr*
// names come from the import instead -- macros do not cross a module boundary.
// wwrStream_t is exported from here via WWR_RT_TYPE(Stream_t) as ::cudaStream_t /
// ::hipStream_t, the SAME vendor handle runtime.h names for device and GMF code,
// so a stream still crosses the boundary as one type. (This is where runtime
// diverges from complex.h, whose cuComplex.h defines no colliding macros.)

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

// wwrStream_t is ::cudaStream_t / ::hipStream_t here (via the import), the same
// vendor handle runtime.h names for device and GMF code -- see the GMF comment
// above for why this comes from the import and not a shared #include.
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
// (its hipHostAlloc* macros are equal-valued aliases the raw module does not
// re-export).
WWR_VALUE(wwrHostAllocDefault, cudaHostAllocDefault, hipHostMallocDefault)
WWR_VALUE(wwrHostAllocMapped, cudaHostAllocMapped, hipHostMallocMapped)
WWR_VALUE(wwrHostAllocWriteCombined, cudaHostAllocWriteCombined, hipHostMallocWriteCombined)

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
