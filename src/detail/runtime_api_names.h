/**
 * @file detail/runtime_api_names.h
 * @brief The backend-neutral runtime-API surface, as a macro-driven include
 *        fragment shared by every consumption path
 *
 * NOT a standalone header: it is the list of wwr* runtime names (types,
 * constants, functions and the few hand-written forwarders) with NO namespace
 * of its own and NO vendor #include. The includer supplies all of that and
 * pastes this inside its own `namespace wwr` -- so one list binds all three
 * ways the surface is consumed: wwr.runtime_api (the module, host,
 * `export namespace wwr`), runtime.h's device section (a device .cu/.cuh, which
 * cannot import the module), and wwr/runtime_api.h (the non-module #include
 * path, host). Add a runtime name here, once, and every path gains it.
 *
 * Before including, the includer must have, in order:
 *   - the vendor runtime header in scope and its allocation-flag macros run
 *     through the #undef-to-constexpr dance (runtime_api.h) -- the `::cuda*`
 *     flag names the WWR_RT_VALUE expansions below put a `::` in front of must
 *     resolve to the constexpr, never a macro;
 *   - std::size_t available (import std, or <cstddef>);
 *   - WWR_SELECT_RAW(cuda, hip) picking the selected backend's raw name --
 *     keyed on WWR_GPU_BACKEND_* in the module (backend.h) or WWR_SELECTED_* in
 *     a device pass (runtime.h) and the #include path (wwr/runtime_api.h), the
 *     one thing that legitimately differs between the sites -- plus
 *     WWR_TYPE_RAW / WWR_VALUE_RAW / WWR_FUNCTION_RAW and the WWR_RT_TYPE /
 *     WWR_RT_VALUE / WWR_RT_FUNCTION conveniences on top of them.
 *
 * See src/runtime_api.cppm, src/runtime.h, src/wwr/runtime_api.h and
 * docs/architecture.md section 3.
 */

#pragma once

#ifndef WWR_RT_TYPE
#error                                                                                             \
    "detail/runtime_api_names.h is an include fragment, not a standalone header: define WWR_RT_TYPE/VALUE/FUNCTION, the _RAW macros and WWR_SELECT_RAW, ensure the vendor runtime header + flag dance and std::size_t are in scope, and #include it inside namespace wwr. See src/runtime.h, src/runtime_api.cppm and src/wwr/runtime_api.h."
#endif

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

// Execution control -- stream/event/launch/occupancy/func types (#261). Plain
// typedefs, enums, structs and the stream-callback function-pointer typedef; the
// _st tags are aliased alongside their typedefs because both shared names must
// carry a wwr* spelling.
WWR_RT_TYPE(StreamCallback_t)
WWR_RT_TYPE(StreamCaptureStatus)
WWR_RT_TYPE(StreamUpdateCaptureDependenciesFlags)
WWR_RT_TYPE(EventRecordNodeParams)
WWR_RT_TYPE(EventWaitNodeParams)
WWR_RT_TYPE(FuncAttribute)
WWR_RT_TYPE(FuncAttributes)
WWR_RT_TYPE(Function_t)
WWR_RT_TYPE(Kernel_t)
WWR_RT_TYPE(KernelNodeParams)
WWR_RT_TYPE(LaunchAttribute)
WWR_RT_TYPE(LaunchAttribute_st)
WWR_RT_TYPE(LaunchAttributeID)
WWR_RT_TYPE(LaunchAttributeValue)
WWR_RT_TYPE(LaunchConfig_st)
WWR_RT_TYPE(LaunchConfig_t)
WWR_RT_TYPE(LaunchMemSyncDomain)
WWR_RT_TYPE(LaunchMemSyncDomainMap)

// ========================================================================
// Constants
// ========================================================================

WWR_RT_VALUE(Success)
WWR_RT_VALUE(ErrorInvalidValue)

// Status / error codes -- the shared cudaError_t / hipError_t enum constants
// (enum values, not the flag macros below, so WWR_RT_VALUE binds ::cudaError*
// straight from the vendor header, no #undef dance). Each alias is the SELECTED
// backend's constant -- the value a wwr caller compares wwrGetLastError() against;
// cross-backend value divergence is irrelevant to a one-backend build. HIP spells
// the timeout code hipErrorLaunchTimeOut (capital O), so that one is written out.
// Deprecated codes auto-omit; backend-only codes are absent from the intersection.
WWR_RT_VALUE(ErrorAlreadyAcquired)
WWR_RT_VALUE(ErrorAlreadyMapped)
WWR_RT_VALUE(ErrorArrayIsMapped)
WWR_RT_VALUE(ErrorAssert)
WWR_RT_VALUE(ErrorCapturedEvent)
WWR_RT_VALUE(ErrorContextIsDestroyed)
WWR_RT_VALUE(ErrorCooperativeLaunchTooLarge)
WWR_RT_VALUE(ErrorFileNotFound)
WWR_RT_VALUE(ErrorGraphExecUpdateFailure)
WWR_RT_VALUE(ErrorHostMemoryAlreadyRegistered)
WWR_RT_VALUE(ErrorHostMemoryNotRegistered)
WWR_RT_VALUE(ErrorIllegalAddress)
WWR_RT_VALUE(ErrorIllegalState)
WWR_RT_VALUE(ErrorInitializationError)
WWR_RT_VALUE(ErrorInsufficientDriver)
WWR_RT_VALUE(ErrorInvalidChannelDescriptor)
WWR_RT_VALUE(ErrorInvalidConfiguration)
WWR_RT_VALUE(ErrorInvalidDevice)
WWR_RT_VALUE(ErrorInvalidDeviceFunction)
WWR_RT_VALUE(ErrorInvalidDevicePointer)
WWR_RT_VALUE(ErrorInvalidGraphicsContext)
WWR_RT_VALUE(ErrorInvalidMemcpyDirection)
WWR_RT_VALUE(ErrorInvalidPitchValue)
WWR_RT_VALUE(ErrorInvalidResourceHandle)
WWR_RT_VALUE(ErrorInvalidSource)
WWR_RT_VALUE(ErrorInvalidSymbol)
WWR_RT_VALUE(ErrorInvalidTexture)
WWR_RT_VALUE(ErrorLaunchFailure)
WWR_RT_VALUE(ErrorLaunchOutOfResources)
WWR_VALUE_RAW(wwrErrorLaunchTimeout, cudaErrorLaunchTimeout, hipErrorLaunchTimeOut)
WWR_RT_VALUE(ErrorMapBufferObjectFailed)
WWR_RT_VALUE(ErrorMemoryAllocation)
WWR_RT_VALUE(ErrorMissingConfiguration)
WWR_RT_VALUE(ErrorNoDevice)
WWR_RT_VALUE(ErrorNotMapped)
WWR_RT_VALUE(ErrorNotMappedAsArray)
WWR_RT_VALUE(ErrorNotMappedAsPointer)
WWR_RT_VALUE(ErrorNotReady)
WWR_RT_VALUE(ErrorNotSupported)
WWR_RT_VALUE(ErrorOperatingSystem)
WWR_RT_VALUE(ErrorPeerAccessAlreadyEnabled)
WWR_RT_VALUE(ErrorPeerAccessNotEnabled)
WWR_RT_VALUE(ErrorPeerAccessUnsupported)
WWR_RT_VALUE(ErrorPriorLaunchFailure)
WWR_RT_VALUE(ErrorProfilerAlreadyStarted)
WWR_RT_VALUE(ErrorProfilerAlreadyStopped)
WWR_RT_VALUE(ErrorProfilerDisabled)
WWR_RT_VALUE(ErrorProfilerNotInitialized)
WWR_RT_VALUE(ErrorSetOnActiveProcess)
WWR_RT_VALUE(ErrorSharedObjectInitFailed)
WWR_RT_VALUE(ErrorSharedObjectSymbolNotFound)
WWR_RT_VALUE(ErrorStreamCaptureImplicit)
WWR_RT_VALUE(ErrorStreamCaptureInvalidated)
WWR_RT_VALUE(ErrorStreamCaptureIsolation)
WWR_RT_VALUE(ErrorStreamCaptureMerge)
WWR_RT_VALUE(ErrorStreamCaptureUnjoined)
WWR_RT_VALUE(ErrorStreamCaptureUnmatched)
WWR_RT_VALUE(ErrorStreamCaptureUnsupported)
WWR_RT_VALUE(ErrorStreamCaptureWrongThread)
WWR_RT_VALUE(ErrorUnknown)
WWR_RT_VALUE(ErrorUnsupportedLimit)

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

// Execution control -- stream/launch/func enum constants (#261). Enum values,
// not flag macros, so WWR_RT_VALUE binds them straight from the vendor header.
WWR_RT_VALUE(StreamCaptureStatusNone)
WWR_RT_VALUE(StreamCaptureStatusActive)
WWR_RT_VALUE(StreamCaptureStatusInvalidated)
WWR_RT_VALUE(StreamAddCaptureDependencies)
WWR_RT_VALUE(StreamSetCaptureDependencies)
WWR_RT_VALUE(FuncAttributeMax)
WWR_RT_VALUE(FuncAttributeMaxDynamicSharedMemorySize)
WWR_RT_VALUE(FuncAttributePreferredSharedMemoryCarveout)
WWR_RT_VALUE(FuncCachePreferNone)
WWR_RT_VALUE(FuncCachePreferShared)
WWR_RT_VALUE(FuncCachePreferL1)
WWR_RT_VALUE(FuncCachePreferEqual)
WWR_RT_VALUE(LaunchAttributeAccessPolicyWindow)
WWR_RT_VALUE(LaunchAttributeCooperative)
WWR_RT_VALUE(LaunchAttributeSynchronizationPolicy)
WWR_RT_VALUE(LaunchAttributePriority)
WWR_RT_VALUE(LaunchAttributeMemSyncDomain)
WWR_RT_VALUE(LaunchAttributeMemSyncDomainMap)
WWR_RT_VALUE(LaunchMemSyncDomainDefault)
WWR_RT_VALUE(LaunchMemSyncDomainRemote)

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
WWR_RT_FUNCTION(PeekAtLastError)

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

// Execution control -- stream/event/launch/func entry points (#261). Each is a
// single (non-overloaded) function on both backends, so a WWR_FUNCTION reference
// names it directly.
WWR_RT_FUNCTION(StreamQuery)
WWR_RT_FUNCTION(StreamGetPriority)
WWR_RT_FUNCTION(StreamGetId)
WWR_RT_FUNCTION(StreamGetDevice)
WWR_RT_FUNCTION(StreamGetAttribute)
WWR_RT_FUNCTION(StreamSetAttribute)
WWR_RT_FUNCTION(StreamAddCallback)
WWR_RT_FUNCTION(StreamAttachMemAsync)
WWR_RT_FUNCTION(StreamIsCapturing)
WWR_RT_FUNCTION(StreamGetCaptureInfo)
WWR_RT_FUNCTION(StreamUpdateCaptureDependencies)
WWR_RT_FUNCTION(EventElapsedTime)
WWR_RT_FUNCTION(LaunchKernel)
WWR_RT_FUNCTION(LaunchKernelExC)
WWR_RT_FUNCTION(LaunchHostFunc)
WWR_RT_FUNCTION(FuncGetAttributes)
WWR_RT_FUNCTION(FuncSetAttribute)
WWR_RT_FUNCTION(FuncSetCacheConfig)

// Occupancy / cooperative-launch forwarders (#261). On HIP each of these entry
// points is an overload set: a template<class T> convenience overload survives
// beside the extern-C function because it sits OUTSIDE the
// __HIP_DISABLE_CPP_FUNCTIONS__ guard (the wwrMallocAsync situation). A reference
// binding cannot alias one member of an overload set, so forward instead -- the
// concrete argument types pick the C overload; the cuda* entry points are single
// functions and resolve the same way. cudaLaunchCooperativeKernel additionally
// differs in signature (its sharedMem is size_t where HIP takes unsigned int), a
// second reason a reference cannot bind it.
inline wwrError_t wwrOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                               int blockSize,
                                                               std::size_t dynamicSMemSize) {
  return WWR_SELECT_RAW(cudaOccupancyMaxActiveBlocksPerMultiprocessor,
                        hipOccupancyMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize,
                                                                      dynamicSMemSize);
}
inline wwrError_t wwrOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks,
                                                                        const void *func,
                                                                        int blockSize,
                                                                        std::size_t dynamicSMemSize,
                                                                        unsigned int flags) {
  return WWR_SELECT_RAW(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
                        hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
      numBlocks, func, blockSize, dynamicSMemSize, flags);
}
inline wwrError_t wwrLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                             void **args, std::size_t sharedMem,
                                             wwrStream_t stream) {
  return WWR_SELECT_RAW(cudaLaunchCooperativeKernel,
                        hipLaunchCooperativeKernel)(func, gridDim, blockDim, args, sharedMem,
                                                    stream);
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
