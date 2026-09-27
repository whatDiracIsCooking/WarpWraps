// runtime_api.cppm - Compile-time tests for gpumod.runtime_api
//
// Every exported gpu* name is the backend's own entity. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.runtime_api;

import std;
import gpumod.runtime_api;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_runtime_api;
#else
import gpumod.hip.hip_runtime_api;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(gpuError_t, cudaError_t)
WWR_SAME_TYPE(gpuStream_t, cudaStream_t)
WWR_SAME_TYPE(gpuStreamCaptureMode, cudaStreamCaptureMode)
WWR_SAME_TYPE(gpuEvent_t, cudaEvent_t)
WWR_SAME_TYPE(gpuMemPool_t, cudaMemPool_t)
WWR_SAME_TYPE(gpuMemPoolProps, cudaMemPoolProps)
WWR_SAME_TYPE(gpuGraph_t, cudaGraph_t)
WWR_SAME_TYPE(gpuGraphExec_t, cudaGraphExec_t)
WWR_SAME_TYPE(gpuDeviceProp, cudaDeviceProp)

WWR_SAME_VALUE(gpuSuccess, cudaSuccess)
WWR_SAME_VALUE(gpuStreamDefault, cudaStreamDefault)
WWR_SAME_VALUE(gpuStreamNonBlocking, cudaStreamNonBlocking)
WWR_SAME_VALUE(gpuStreamCaptureModeGlobal, cudaStreamCaptureModeGlobal)
WWR_SAME_VALUE(gpuStreamCaptureModeThreadLocal, cudaStreamCaptureModeThreadLocal)
WWR_SAME_VALUE(gpuStreamCaptureModeRelaxed, cudaStreamCaptureModeRelaxed)
WWR_SAME_VALUE(gpuEventDefault, cudaEventDefault)
WWR_SAME_VALUE(gpuEventBlockingSync, cudaEventBlockingSync)
WWR_SAME_VALUE(gpuEventDisableTiming, cudaEventDisableTiming)
WWR_SAME_VALUE(gpuMemAllocationTypePinned, cudaMemAllocationTypePinned)
WWR_SAME_VALUE(gpuMemHandleTypeNone, cudaMemHandleTypeNone)
WWR_SAME_VALUE(gpuMemLocationTypeDevice, cudaMemLocationTypeDevice)
WWR_SAME_VALUE(gpuMemPoolAttrReleaseThreshold, cudaMemPoolAttrReleaseThreshold)

WWR_SAME_FUNCTION(gpuGetErrorName, cudaGetErrorName)
WWR_SAME_FUNCTION(gpuGetErrorString, cudaGetErrorString)
WWR_SAME_FUNCTION(gpuGetDevice, cudaGetDevice)
WWR_SAME_FUNCTION(gpuSetDevice, cudaSetDevice)
WWR_SAME_FUNCTION(gpuGetDeviceProperties, cudaGetDeviceProperties)
WWR_SAME_FUNCTION(gpuStreamCreate, cudaStreamCreate)
WWR_SAME_FUNCTION(gpuStreamCreateWithFlags, cudaStreamCreateWithFlags)
WWR_SAME_FUNCTION(gpuStreamCreateWithPriority, cudaStreamCreateWithPriority)
WWR_SAME_FUNCTION(gpuStreamGetFlags, cudaStreamGetFlags)
WWR_SAME_FUNCTION(gpuStreamWaitEvent, cudaStreamWaitEvent)
WWR_SAME_FUNCTION(gpuStreamSynchronize, cudaStreamSynchronize)
WWR_SAME_FUNCTION(gpuStreamDestroy, cudaStreamDestroy)
WWR_SAME_FUNCTION(gpuStreamBeginCapture, cudaStreamBeginCapture)
WWR_SAME_FUNCTION(gpuStreamEndCapture, cudaStreamEndCapture)
WWR_SAME_FUNCTION(gpuEventCreate, cudaEventCreate)
WWR_SAME_FUNCTION(gpuEventCreateWithFlags, cudaEventCreateWithFlags)
WWR_SAME_FUNCTION(gpuEventRecord, cudaEventRecord)
WWR_SAME_FUNCTION(gpuEventRecordWithFlags, cudaEventRecordWithFlags)
WWR_SAME_FUNCTION(gpuEventSynchronize, cudaEventSynchronize)
WWR_SAME_FUNCTION(gpuEventQuery, cudaEventQuery)
WWR_SAME_FUNCTION(gpuEventDestroy, cudaEventDestroy)
WWR_SAME_FUNCTION(gpuMemPoolCreate, cudaMemPoolCreate)
WWR_SAME_FUNCTION(gpuMemPoolSetAttribute, cudaMemPoolSetAttribute)
WWR_SAME_FUNCTION(gpuMemPoolDestroy, cudaMemPoolDestroy)
WWR_SAME_FUNCTION(gpuGraphCreate, cudaGraphCreate)
WWR_SAME_FUNCTION(gpuGraphDestroy, cudaGraphDestroy)
WWR_SAME_FUNCTION(gpuGraphExecDestroy, cudaGraphExecDestroy)
WWR_SAME_FUNCTION(gpuGraphLaunch, cudaGraphLaunch)
WWR_SAME_FUNCTION(gpuGraphUpload, cudaGraphUpload)
// gpuGraphInstantiate is a hand-written forwarding function (its backend
// signatures diverge), not a plain alias, so &gpuGraphInstantiate is not any
// single backend symbol -- WWR_SAME_FUNCTION does not apply. It is proven by
// compiling (the forward resolves on each backend) and by GpuGraphExecTests.

WWR_SAME_TYPE(gpuMemcpyKind, cudaMemcpyKind)
WWR_SAME_VALUE(gpuMemcpyHostToDevice, cudaMemcpyHostToDevice)
WWR_SAME_VALUE(gpuMemcpyDeviceToHost, cudaMemcpyDeviceToHost)
WWR_SAME_VALUE(gpuMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice)
WWR_SAME_FUNCTION(gpuMalloc, cudaMalloc)
WWR_SAME_FUNCTION(gpuFree, cudaFree)
WWR_SAME_FUNCTION(gpuMemcpy, cudaMemcpy)
WWR_SAME_FUNCTION(gpuMemset, cudaMemset)
WWR_SAME_FUNCTION(gpuDeviceSynchronize, cudaDeviceSynchronize)

WWR_SAME_VALUE(gpuErrorInvalidValue, cudaErrorInvalidValue)
WWR_SAME_FUNCTION(gpuGetLastError, cudaGetLastError)

WWR_SAME_VALUE(gpuMemcpyDefault, cudaMemcpyDefault)
WWR_SAME_FUNCTION(gpuMallocAsync, cudaMallocAsync)
WWR_SAME_FUNCTION(gpuMallocFromPoolAsync, cudaMallocFromPoolAsync)
WWR_SAME_FUNCTION(gpuFreeAsync, cudaFreeAsync)
WWR_SAME_FUNCTION(gpuMemcpyAsync, cudaMemcpyAsync)
WWR_SAME_FUNCTION(gpuMemsetAsync, cudaMemsetAsync)

WWR_SAME_VALUE(gpuHostAllocDefault, cudaHostAllocDefault)
WWR_SAME_VALUE(gpuHostAllocMapped, cudaHostAllocMapped)
WWR_SAME_VALUE(gpuHostAllocWriteCombined, cudaHostAllocWriteCombined)
WWR_SAME_FUNCTION(gpuHostAlloc, cudaHostAlloc)
WWR_SAME_FUNCTION(gpuFreeHost, cudaFreeHost)

WWR_SAME_VALUE(gpuMemAttachGlobal, cudaMemAttachGlobal)
WWR_SAME_VALUE(gpuMemAttachHost, cudaMemAttachHost)
WWR_SAME_FUNCTION(gpuMallocManaged, cudaMallocManaged)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gpuError_t, hipError_t)
WWR_SAME_TYPE(gpuStream_t, hipStream_t)
WWR_SAME_TYPE(gpuStreamCaptureMode, hipStreamCaptureMode)
WWR_SAME_TYPE(gpuEvent_t, hipEvent_t)
WWR_SAME_TYPE(gpuMemPool_t, hipMemPool_t)
WWR_SAME_TYPE(gpuMemPoolProps, hipMemPoolProps)
WWR_SAME_TYPE(gpuGraph_t, hipGraph_t)
WWR_SAME_TYPE(gpuGraphExec_t, hipGraphExec_t)
WWR_SAME_TYPE(gpuDeviceProp, hipDeviceProp_tR0600)

WWR_SAME_VALUE(gpuSuccess, hipSuccess)
WWR_SAME_VALUE(gpuStreamDefault, hipStreamDefault)
WWR_SAME_VALUE(gpuStreamNonBlocking, hipStreamNonBlocking)
WWR_SAME_VALUE(gpuStreamCaptureModeGlobal, hipStreamCaptureModeGlobal)
WWR_SAME_VALUE(gpuStreamCaptureModeThreadLocal, hipStreamCaptureModeThreadLocal)
WWR_SAME_VALUE(gpuStreamCaptureModeRelaxed, hipStreamCaptureModeRelaxed)
WWR_SAME_VALUE(gpuEventDefault, hipEventDefault)
WWR_SAME_VALUE(gpuEventBlockingSync, hipEventBlockingSync)
WWR_SAME_VALUE(gpuEventDisableTiming, hipEventDisableTiming)
WWR_SAME_VALUE(gpuMemAllocationTypePinned, hipMemAllocationTypePinned)
WWR_SAME_VALUE(gpuMemHandleTypeNone, hipMemHandleTypeNone)
WWR_SAME_VALUE(gpuMemLocationTypeDevice, hipMemLocationTypeDevice)
WWR_SAME_VALUE(gpuMemPoolAttrReleaseThreshold, hipMemPoolAttrReleaseThreshold)

WWR_SAME_FUNCTION(gpuGetErrorName, hipGetErrorName)
WWR_SAME_FUNCTION(gpuGetErrorString, hipGetErrorString)
WWR_SAME_FUNCTION(gpuGetDevice, hipGetDevice)
WWR_SAME_FUNCTION(gpuSetDevice, hipSetDevice)
WWR_SAME_FUNCTION(gpuGetDeviceProperties, hipGetDevicePropertiesR0600)
WWR_SAME_FUNCTION(gpuStreamCreate, hipStreamCreate)
WWR_SAME_FUNCTION(gpuStreamCreateWithFlags, hipStreamCreateWithFlags)
WWR_SAME_FUNCTION(gpuStreamCreateWithPriority, hipStreamCreateWithPriority)
WWR_SAME_FUNCTION(gpuStreamGetFlags, hipStreamGetFlags)
WWR_SAME_FUNCTION(gpuStreamWaitEvent, hipStreamWaitEvent)
WWR_SAME_FUNCTION(gpuStreamSynchronize, hipStreamSynchronize)
WWR_SAME_FUNCTION(gpuStreamDestroy, hipStreamDestroy)
WWR_SAME_FUNCTION(gpuStreamBeginCapture, hipStreamBeginCapture)
WWR_SAME_FUNCTION(gpuStreamEndCapture, hipStreamEndCapture)
WWR_SAME_FUNCTION(gpuEventCreate, hipEventCreate)
WWR_SAME_FUNCTION(gpuEventCreateWithFlags, hipEventCreateWithFlags)
WWR_SAME_FUNCTION(gpuEventRecord, hipEventRecord)
WWR_SAME_FUNCTION(gpuEventRecordWithFlags, hipEventRecordWithFlags)
WWR_SAME_FUNCTION(gpuEventSynchronize, hipEventSynchronize)
WWR_SAME_FUNCTION(gpuEventQuery, hipEventQuery)
WWR_SAME_FUNCTION(gpuEventDestroy, hipEventDestroy)
WWR_SAME_FUNCTION(gpuMemPoolCreate, hipMemPoolCreate)
WWR_SAME_FUNCTION(gpuMemPoolSetAttribute, hipMemPoolSetAttribute)
WWR_SAME_FUNCTION(gpuMemPoolDestroy, hipMemPoolDestroy)
WWR_SAME_FUNCTION(gpuGraphCreate, hipGraphCreate)
WWR_SAME_FUNCTION(gpuGraphDestroy, hipGraphDestroy)
WWR_SAME_FUNCTION(gpuGraphExecDestroy, hipGraphExecDestroy)
WWR_SAME_FUNCTION(gpuGraphLaunch, hipGraphLaunch)
WWR_SAME_FUNCTION(gpuGraphUpload, hipGraphUpload)
// gpuGraphInstantiate forwards to hipGraphInstantiateWithFlags here (CUDA maps
// to cudaGraphInstantiate); as a hand-written forwarding function it is not a
// single backend symbol, so WWR_SAME_FUNCTION does not apply. Proven by
// compiling and by GpuGraphExecTests.

WWR_SAME_TYPE(gpuMemcpyKind, hipMemcpyKind)
WWR_SAME_VALUE(gpuMemcpyHostToDevice, hipMemcpyHostToDevice)
WWR_SAME_VALUE(gpuMemcpyDeviceToHost, hipMemcpyDeviceToHost)
WWR_SAME_VALUE(gpuMemcpyDeviceToDevice, hipMemcpyDeviceToDevice)
// hipMalloc is an overload set (plus a template<class T> hipMalloc(T**, size_t));
// gpuMalloc must be its void** overload.
static_assert(&gpuMalloc == static_cast<hipError_t (*)(void **, std::size_t)>(&hipMalloc),
              "gpuMalloc is not hipMalloc(void**, size_t)");
WWR_LINK_CHECK(gpuMalloc)
WWR_SAME_FUNCTION(gpuFree, hipFree)
WWR_SAME_FUNCTION(gpuMemcpy, hipMemcpy)
WWR_SAME_FUNCTION(gpuMemset, hipMemset)
WWR_SAME_FUNCTION(gpuDeviceSynchronize, hipDeviceSynchronize)

WWR_SAME_VALUE(gpuErrorInvalidValue, hipErrorInvalidValue)
WWR_SAME_FUNCTION(gpuGetLastError, hipGetLastError)

WWR_SAME_VALUE(gpuMemcpyDefault, hipMemcpyDefault)
WWR_SAME_FUNCTION(gpuMallocAsync, hipMallocAsync)
WWR_SAME_FUNCTION(gpuMallocFromPoolAsync, hipMallocFromPoolAsync)
WWR_SAME_FUNCTION(gpuFreeAsync, hipFreeAsync)
WWR_SAME_FUNCTION(gpuMemcpyAsync, hipMemcpyAsync)
WWR_SAME_FUNCTION(gpuMemsetAsync, hipMemsetAsync)

// hip_runtime_api.h's flag macros are hipHostMalloc*; hipHostAlloc* are
// equal-valued aliases.
WWR_SAME_VALUE(gpuHostAllocDefault, hipHostMallocDefault)
WWR_SAME_VALUE(gpuHostAllocMapped, hipHostMallocMapped)
WWR_SAME_VALUE(gpuHostAllocWriteCombined, hipHostMallocWriteCombined)
// hipHostAlloc and hipMallocManaged are overload sets like hipMalloc; the gpu
// names must be their void** overloads.
static_assert(&gpuHostAlloc ==
                  static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(&hipHostAlloc),
              "gpuHostAlloc is not hipHostAlloc(void**, size_t, unsigned)");
WWR_LINK_CHECK(gpuHostAlloc)
WWR_SAME_FUNCTION(gpuFreeHost, hipFreeHost)

WWR_SAME_VALUE(gpuMemAttachGlobal, hipMemAttachGlobal)
WWR_SAME_VALUE(gpuMemAttachHost, hipMemAttachHost)
static_assert(&gpuMallocManaged == static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(
                                       &hipMallocManaged),
              "gpuMallocManaged is not hipMallocManaged(void**, size_t, unsigned)");
WWR_LINK_CHECK(gpuMallocManaged)

#endif

} // namespace wwr::test
