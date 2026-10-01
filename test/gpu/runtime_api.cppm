// runtime_api.cppm - Compile-time tests for wwr.runtime_api
//
// Every exported wwr* name is the backend's own entity. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.runtime_api;

import std;
import wwr.runtime_api;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_runtime_api;
#else
import wwr.hip.hip_runtime_api;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrError_t, cudaError_t)
WWR_SAME_TYPE(wwrStream_t, cudaStream_t)
WWR_SAME_TYPE(wwrStreamCaptureMode, cudaStreamCaptureMode)
WWR_SAME_TYPE(wwrEvent_t, cudaEvent_t)
WWR_SAME_TYPE(wwrMemPool_t, cudaMemPool_t)
WWR_SAME_TYPE(wwrMemPoolProps, cudaMemPoolProps)
WWR_SAME_TYPE(wwrGraph_t, cudaGraph_t)
WWR_SAME_TYPE(wwrGraphExec_t, cudaGraphExec_t)
WWR_SAME_TYPE(wwrDeviceProp, cudaDeviceProp)

WWR_SAME_VALUE(wwrSuccess, cudaSuccess)
WWR_SAME_VALUE(wwrStreamDefault, cudaStreamDefault)
WWR_SAME_VALUE(wwrStreamNonBlocking, cudaStreamNonBlocking)
WWR_SAME_VALUE(wwrStreamCaptureModeGlobal, cudaStreamCaptureModeGlobal)
WWR_SAME_VALUE(wwrStreamCaptureModeThreadLocal, cudaStreamCaptureModeThreadLocal)
WWR_SAME_VALUE(wwrStreamCaptureModeRelaxed, cudaStreamCaptureModeRelaxed)
WWR_SAME_VALUE(wwrEventDefault, cudaEventDefault)
WWR_SAME_VALUE(wwrEventBlockingSync, cudaEventBlockingSync)
WWR_SAME_VALUE(wwrEventDisableTiming, cudaEventDisableTiming)
WWR_SAME_VALUE(wwrMemAllocationTypePinned, cudaMemAllocationTypePinned)
WWR_SAME_VALUE(wwrMemHandleTypeNone, cudaMemHandleTypeNone)
WWR_SAME_VALUE(wwrMemLocationTypeDevice, cudaMemLocationTypeDevice)
WWR_SAME_VALUE(wwrMemPoolAttrReleaseThreshold, cudaMemPoolAttrReleaseThreshold)

WWR_SAME_FUNCTION(wwrGetErrorName, cudaGetErrorName)
WWR_SAME_FUNCTION(wwrGetErrorString, cudaGetErrorString)
WWR_SAME_FUNCTION(wwrGetDevice, cudaGetDevice)
WWR_SAME_FUNCTION(wwrSetDevice, cudaSetDevice)
WWR_SAME_FUNCTION(wwrGetDeviceProperties, cudaGetDeviceProperties)
WWR_SAME_FUNCTION(wwrStreamCreate, cudaStreamCreate)
WWR_SAME_FUNCTION(wwrStreamCreateWithFlags, cudaStreamCreateWithFlags)
WWR_SAME_FUNCTION(wwrStreamCreateWithPriority, cudaStreamCreateWithPriority)
WWR_SAME_FUNCTION(wwrStreamGetFlags, cudaStreamGetFlags)
WWR_SAME_FUNCTION(wwrStreamWaitEvent, cudaStreamWaitEvent)
WWR_SAME_FUNCTION(wwrStreamSynchronize, cudaStreamSynchronize)
WWR_SAME_FUNCTION(wwrStreamDestroy, cudaStreamDestroy)
WWR_SAME_FUNCTION(wwrStreamBeginCapture, cudaStreamBeginCapture)
WWR_SAME_FUNCTION(wwrStreamEndCapture, cudaStreamEndCapture)
WWR_SAME_FUNCTION(wwrEventCreate, cudaEventCreate)
WWR_SAME_FUNCTION(wwrEventCreateWithFlags, cudaEventCreateWithFlags)
WWR_SAME_FUNCTION(wwrEventRecord, cudaEventRecord)
WWR_SAME_FUNCTION(wwrEventRecordWithFlags, cudaEventRecordWithFlags)
WWR_SAME_FUNCTION(wwrEventSynchronize, cudaEventSynchronize)
WWR_SAME_FUNCTION(wwrEventQuery, cudaEventQuery)
WWR_SAME_FUNCTION(wwrEventDestroy, cudaEventDestroy)
WWR_SAME_FUNCTION(wwrMemPoolCreate, cudaMemPoolCreate)
WWR_SAME_FUNCTION(wwrMemPoolSetAttribute, cudaMemPoolSetAttribute)
WWR_SAME_FUNCTION(wwrMemPoolDestroy, cudaMemPoolDestroy)
WWR_SAME_FUNCTION(wwrGraphCreate, cudaGraphCreate)
WWR_SAME_FUNCTION(wwrGraphDestroy, cudaGraphDestroy)
WWR_SAME_FUNCTION(wwrGraphExecDestroy, cudaGraphExecDestroy)
WWR_SAME_FUNCTION(wwrGraphLaunch, cudaGraphLaunch)
WWR_SAME_FUNCTION(wwrGraphUpload, cudaGraphUpload)
// wwrGraphInstantiate is a hand-written forwarding function (its backend
// signatures diverge), not a plain alias, so &wwrGraphInstantiate is not any
// single backend symbol -- WWR_SAME_FUNCTION does not apply. It is proven by
// compiling (the forward resolves on each backend) and by GpuGraphExecTests.

WWR_SAME_TYPE(wwrMemcpyKind, cudaMemcpyKind)
WWR_SAME_VALUE(wwrMemcpyHostToDevice, cudaMemcpyHostToDevice)
WWR_SAME_VALUE(wwrMemcpyDeviceToHost, cudaMemcpyDeviceToHost)
WWR_SAME_VALUE(wwrMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice)
WWR_SAME_FUNCTION(wwrMalloc, cudaMalloc)
WWR_SAME_FUNCTION(wwrFree, cudaFree)
WWR_SAME_FUNCTION(wwrMemcpy, cudaMemcpy)
WWR_SAME_FUNCTION(wwrMemset, cudaMemset)
WWR_SAME_FUNCTION(wwrDeviceSynchronize, cudaDeviceSynchronize)

WWR_SAME_VALUE(wwrErrorInvalidValue, cudaErrorInvalidValue)
WWR_SAME_FUNCTION(wwrGetLastError, cudaGetLastError)

WWR_SAME_VALUE(wwrMemcpyDefault, cudaMemcpyDefault)
// wwrMallocAsync / wwrMallocFromPoolAsync are hand-written forwarders, not plain
// aliases: the HIP entry points are overload sets whose template overload the
// disable macro does not suppress, so the wwr* layer forwards rather than binds a
// reference. &wwr* is therefore not a single backend symbol -- WWR_SAME_FUNCTION
// does not apply. Proven by compiling and by GpuMemoryBufferTests / basic.cpp.
WWR_SAME_FUNCTION(wwrFreeAsync, cudaFreeAsync)
WWR_SAME_FUNCTION(wwrMemcpyAsync, cudaMemcpyAsync)
WWR_SAME_FUNCTION(wwrMemsetAsync, cudaMemsetAsync)

WWR_SAME_VALUE(wwrHostAllocDefault, cudaHostAllocDefault)
WWR_SAME_VALUE(wwrHostAllocMapped, cudaHostAllocMapped)
WWR_SAME_VALUE(wwrHostAllocWriteCombined, cudaHostAllocWriteCombined)
WWR_SAME_FUNCTION(wwrHostAlloc, cudaHostAlloc)
WWR_SAME_FUNCTION(wwrFreeHost, cudaFreeHost)

WWR_SAME_VALUE(wwrMemAttachGlobal, cudaMemAttachGlobal)
WWR_SAME_VALUE(wwrMemAttachHost, cudaMemAttachHost)
WWR_SAME_FUNCTION(wwrMallocManaged, cudaMallocManaged)

// Texture and surface objects
WWR_SAME_TYPE(wwrTextureObject_t, cudaTextureObject_t)
WWR_SAME_TYPE(wwrSurfaceObject_t, cudaSurfaceObject_t)
WWR_SAME_TYPE(wwrResourceDesc, cudaResourceDesc)
WWR_SAME_TYPE(wwrTextureDesc, cudaTextureDesc)
WWR_SAME_TYPE(wwrResourceViewDesc, cudaResourceViewDesc)
WWR_SAME_TYPE(wwrChannelFormatDesc, cudaChannelFormatDesc)
WWR_SAME_TYPE(wwrChannelFormatKind, cudaChannelFormatKind)
WWR_SAME_TYPE(wwrResourceType, cudaResourceType)
WWR_SAME_TYPE(wwrArray_t, cudaArray_t)
WWR_SAME_VALUE(wwrChannelFormatKindSigned, cudaChannelFormatKindSigned)
WWR_SAME_VALUE(wwrChannelFormatKindUnsigned, cudaChannelFormatKindUnsigned)
WWR_SAME_VALUE(wwrChannelFormatKindFloat, cudaChannelFormatKindFloat)
WWR_SAME_VALUE(wwrResourceTypeArray, cudaResourceTypeArray)
WWR_SAME_VALUE(wwrResourceTypeLinear, cudaResourceTypeLinear)
WWR_SAME_VALUE(wwrArrayDefault, cudaArrayDefault)
WWR_SAME_VALUE(wwrArraySurfaceLoadStore, cudaArraySurfaceLoadStore)
WWR_SAME_FUNCTION(wwrMallocArray, cudaMallocArray)
WWR_SAME_FUNCTION(wwrFreeArray, cudaFreeArray)
WWR_SAME_FUNCTION(wwrCreateTextureObject, cudaCreateTextureObject)
WWR_SAME_FUNCTION(wwrDestroyTextureObject, cudaDestroyTextureObject)
WWR_SAME_FUNCTION(wwrCreateSurfaceObject, cudaCreateSurfaceObject)
WWR_SAME_FUNCTION(wwrDestroySurfaceObject, cudaDestroySurfaceObject)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrError_t, hipError_t)
WWR_SAME_TYPE(wwrStream_t, hipStream_t)
WWR_SAME_TYPE(wwrStreamCaptureMode, hipStreamCaptureMode)
WWR_SAME_TYPE(wwrEvent_t, hipEvent_t)
WWR_SAME_TYPE(wwrMemPool_t, hipMemPool_t)
WWR_SAME_TYPE(wwrMemPoolProps, hipMemPoolProps)
WWR_SAME_TYPE(wwrGraph_t, hipGraph_t)
WWR_SAME_TYPE(wwrGraphExec_t, hipGraphExec_t)
WWR_SAME_TYPE(wwrDeviceProp, hipDeviceProp_tR0600)

WWR_SAME_VALUE(wwrSuccess, hipSuccess)
WWR_SAME_VALUE(wwrStreamDefault, hipStreamDefault)
WWR_SAME_VALUE(wwrStreamNonBlocking, hipStreamNonBlocking)
WWR_SAME_VALUE(wwrStreamCaptureModeGlobal, hipStreamCaptureModeGlobal)
WWR_SAME_VALUE(wwrStreamCaptureModeThreadLocal, hipStreamCaptureModeThreadLocal)
WWR_SAME_VALUE(wwrStreamCaptureModeRelaxed, hipStreamCaptureModeRelaxed)
WWR_SAME_VALUE(wwrEventDefault, hipEventDefault)
WWR_SAME_VALUE(wwrEventBlockingSync, hipEventBlockingSync)
WWR_SAME_VALUE(wwrEventDisableTiming, hipEventDisableTiming)
WWR_SAME_VALUE(wwrMemAllocationTypePinned, hipMemAllocationTypePinned)
WWR_SAME_VALUE(wwrMemHandleTypeNone, hipMemHandleTypeNone)
WWR_SAME_VALUE(wwrMemLocationTypeDevice, hipMemLocationTypeDevice)
WWR_SAME_VALUE(wwrMemPoolAttrReleaseThreshold, hipMemPoolAttrReleaseThreshold)

WWR_SAME_FUNCTION(wwrGetErrorName, hipGetErrorName)
WWR_SAME_FUNCTION(wwrGetErrorString, hipGetErrorString)
WWR_SAME_FUNCTION(wwrGetDevice, hipGetDevice)
WWR_SAME_FUNCTION(wwrSetDevice, hipSetDevice)
WWR_SAME_FUNCTION(wwrGetDeviceProperties, hipGetDevicePropertiesR0600)
WWR_SAME_FUNCTION(wwrStreamCreate, hipStreamCreate)
WWR_SAME_FUNCTION(wwrStreamCreateWithFlags, hipStreamCreateWithFlags)
WWR_SAME_FUNCTION(wwrStreamCreateWithPriority, hipStreamCreateWithPriority)
WWR_SAME_FUNCTION(wwrStreamGetFlags, hipStreamGetFlags)
WWR_SAME_FUNCTION(wwrStreamWaitEvent, hipStreamWaitEvent)
WWR_SAME_FUNCTION(wwrStreamSynchronize, hipStreamSynchronize)
WWR_SAME_FUNCTION(wwrStreamDestroy, hipStreamDestroy)
WWR_SAME_FUNCTION(wwrStreamBeginCapture, hipStreamBeginCapture)
WWR_SAME_FUNCTION(wwrStreamEndCapture, hipStreamEndCapture)
WWR_SAME_FUNCTION(wwrEventCreate, hipEventCreate)
WWR_SAME_FUNCTION(wwrEventCreateWithFlags, hipEventCreateWithFlags)
WWR_SAME_FUNCTION(wwrEventRecord, hipEventRecord)
WWR_SAME_FUNCTION(wwrEventRecordWithFlags, hipEventRecordWithFlags)
WWR_SAME_FUNCTION(wwrEventSynchronize, hipEventSynchronize)
WWR_SAME_FUNCTION(wwrEventQuery, hipEventQuery)
WWR_SAME_FUNCTION(wwrEventDestroy, hipEventDestroy)
WWR_SAME_FUNCTION(wwrMemPoolCreate, hipMemPoolCreate)
WWR_SAME_FUNCTION(wwrMemPoolSetAttribute, hipMemPoolSetAttribute)
WWR_SAME_FUNCTION(wwrMemPoolDestroy, hipMemPoolDestroy)
WWR_SAME_FUNCTION(wwrGraphCreate, hipGraphCreate)
WWR_SAME_FUNCTION(wwrGraphDestroy, hipGraphDestroy)
WWR_SAME_FUNCTION(wwrGraphExecDestroy, hipGraphExecDestroy)
WWR_SAME_FUNCTION(wwrGraphLaunch, hipGraphLaunch)
WWR_SAME_FUNCTION(wwrGraphUpload, hipGraphUpload)
// wwrGraphInstantiate forwards to hipGraphInstantiateWithFlags here (CUDA maps
// to cudaGraphInstantiate); as a hand-written forwarding function it is not a
// single backend symbol, so WWR_SAME_FUNCTION does not apply. Proven by
// compiling and by GpuGraphExecTests.

WWR_SAME_TYPE(wwrMemcpyKind, hipMemcpyKind)
WWR_SAME_VALUE(wwrMemcpyHostToDevice, hipMemcpyHostToDevice)
WWR_SAME_VALUE(wwrMemcpyDeviceToHost, hipMemcpyDeviceToHost)
WWR_SAME_VALUE(wwrMemcpyDeviceToDevice, hipMemcpyDeviceToDevice)
// hipMalloc is an overload set (plus a template<class T> hipMalloc(T**, size_t));
// wwrMalloc must be its void** overload.
static_assert(&wwrMalloc == static_cast<hipError_t (*)(void **, std::size_t)>(&hipMalloc),
              "wwrMalloc is not hipMalloc(void**, size_t)");
WWR_LINK_CHECK(wwrMalloc)
WWR_SAME_FUNCTION(wwrFree, hipFree)
WWR_SAME_FUNCTION(wwrMemcpy, hipMemcpy)
WWR_SAME_FUNCTION(wwrMemset, hipMemset)
WWR_SAME_FUNCTION(wwrDeviceSynchronize, hipDeviceSynchronize)

WWR_SAME_VALUE(wwrErrorInvalidValue, hipErrorInvalidValue)
WWR_SAME_FUNCTION(wwrGetLastError, hipGetLastError)

WWR_SAME_VALUE(wwrMemcpyDefault, hipMemcpyDefault)
// wwrMallocAsync / wwrMallocFromPoolAsync are hand-written forwarders, not plain
// aliases: hipMallocAsync / hipMallocFromPoolAsync are overload sets whose
// template<class T> (T**, ...) overload __HIP_DISABLE_CPP_FUNCTIONS__ does not
// suppress (unlike hipMalloc's), so the wwr* layer forwards rather than binds a
// reference. &wwr* is therefore not a single backend symbol -- WWR_SAME_FUNCTION
// does not apply. Proven by compiling and by GpuMemoryBufferTests / basic.cpp.
WWR_SAME_FUNCTION(wwrFreeAsync, hipFreeAsync)
WWR_SAME_FUNCTION(wwrMemcpyAsync, hipMemcpyAsync)
WWR_SAME_FUNCTION(wwrMemsetAsync, hipMemsetAsync)

// hip_runtime_api.h's flag macros are hipHostMalloc*; hipHostAlloc* are
// equal-valued aliases.
WWR_SAME_VALUE(wwrHostAllocDefault, hipHostMallocDefault)
WWR_SAME_VALUE(wwrHostAllocMapped, hipHostMallocMapped)
WWR_SAME_VALUE(wwrHostAllocWriteCombined, hipHostMallocWriteCombined)
// hipHostAlloc and hipMallocManaged are overload sets like hipMalloc; the gpu
// names must be their void** overloads.
static_assert(&wwrHostAlloc ==
                  static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(&hipHostAlloc),
              "wwrHostAlloc is not hipHostAlloc(void**, size_t, unsigned)");
WWR_LINK_CHECK(wwrHostAlloc)
WWR_SAME_FUNCTION(wwrFreeHost, hipFreeHost)

WWR_SAME_VALUE(wwrMemAttachGlobal, hipMemAttachGlobal)
WWR_SAME_VALUE(wwrMemAttachHost, hipMemAttachHost)
static_assert(&wwrMallocManaged == static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(
                                       &hipMallocManaged),
              "wwrMallocManaged is not hipMallocManaged(void**, size_t, unsigned)");
WWR_LINK_CHECK(wwrMallocManaged)

// Texture and surface objects
WWR_SAME_TYPE(wwrTextureObject_t, hipTextureObject_t)
WWR_SAME_TYPE(wwrSurfaceObject_t, hipSurfaceObject_t)
WWR_SAME_TYPE(wwrResourceDesc, hipResourceDesc)
WWR_SAME_TYPE(wwrTextureDesc, hipTextureDesc)
WWR_SAME_TYPE(wwrResourceViewDesc, hipResourceViewDesc)
WWR_SAME_TYPE(wwrChannelFormatDesc, hipChannelFormatDesc)
WWR_SAME_TYPE(wwrChannelFormatKind, hipChannelFormatKind)
WWR_SAME_TYPE(wwrResourceType, hipResourceType)
WWR_SAME_TYPE(wwrArray_t, hipArray_t)
WWR_SAME_VALUE(wwrChannelFormatKindSigned, hipChannelFormatKindSigned)
WWR_SAME_VALUE(wwrChannelFormatKindUnsigned, hipChannelFormatKindUnsigned)
WWR_SAME_VALUE(wwrChannelFormatKindFloat, hipChannelFormatKindFloat)
WWR_SAME_VALUE(wwrResourceTypeArray, hipResourceTypeArray)
WWR_SAME_VALUE(wwrResourceTypeLinear, hipResourceTypeLinear)
WWR_SAME_VALUE(wwrArrayDefault, hipArrayDefault)
WWR_SAME_VALUE(wwrArraySurfaceLoadStore, hipArraySurfaceLoadStore)
WWR_SAME_FUNCTION(wwrMallocArray, hipMallocArray)
WWR_SAME_FUNCTION(wwrFreeArray, hipFreeArray)
WWR_SAME_FUNCTION(wwrCreateTextureObject, hipCreateTextureObject)
WWR_SAME_FUNCTION(wwrDestroyTextureObject, hipDestroyTextureObject)
WWR_SAME_FUNCTION(wwrCreateSurfaceObject, hipCreateSurfaceObject)
WWR_SAME_FUNCTION(wwrDestroySurfaceObject, hipDestroySurfaceObject)

#endif

} // namespace wwr::test
