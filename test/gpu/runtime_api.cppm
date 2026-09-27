// runtime_api.cppm - Compile-time tests for gpumod.runtime_api
//
// Every exported gpu* name is the backend's own entity. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.runtime_api;

import std;
import gpumod.runtime_api;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_runtime_api;
#else
import gpumod.hip.hip_runtime_api;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

GPUMOD_SAME_TYPE(gpuError_t, cudaError_t)
GPUMOD_SAME_TYPE(gpuStream_t, cudaStream_t)
GPUMOD_SAME_TYPE(gpuStreamCaptureMode, cudaStreamCaptureMode)
GPUMOD_SAME_TYPE(gpuEvent_t, cudaEvent_t)
GPUMOD_SAME_TYPE(gpuMemPool_t, cudaMemPool_t)
GPUMOD_SAME_TYPE(gpuMemPoolProps, cudaMemPoolProps)
GPUMOD_SAME_TYPE(gpuGraph_t, cudaGraph_t)
GPUMOD_SAME_TYPE(gpuGraphExec_t, cudaGraphExec_t)
GPUMOD_SAME_TYPE(gpuDeviceProp, cudaDeviceProp)

GPUMOD_SAME_VALUE(gpuSuccess, cudaSuccess)
GPUMOD_SAME_VALUE(gpuStreamDefault, cudaStreamDefault)
GPUMOD_SAME_VALUE(gpuStreamNonBlocking, cudaStreamNonBlocking)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeGlobal, cudaStreamCaptureModeGlobal)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeThreadLocal, cudaStreamCaptureModeThreadLocal)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeRelaxed, cudaStreamCaptureModeRelaxed)
GPUMOD_SAME_VALUE(gpuEventDefault, cudaEventDefault)
GPUMOD_SAME_VALUE(gpuEventBlockingSync, cudaEventBlockingSync)
GPUMOD_SAME_VALUE(gpuEventDisableTiming, cudaEventDisableTiming)
GPUMOD_SAME_VALUE(gpuMemAllocationTypePinned, cudaMemAllocationTypePinned)
GPUMOD_SAME_VALUE(gpuMemHandleTypeNone, cudaMemHandleTypeNone)
GPUMOD_SAME_VALUE(gpuMemLocationTypeDevice, cudaMemLocationTypeDevice)
GPUMOD_SAME_VALUE(gpuMemPoolAttrReleaseThreshold, cudaMemPoolAttrReleaseThreshold)

GPUMOD_SAME_FUNCTION(gpuGetErrorName, cudaGetErrorName)
GPUMOD_SAME_FUNCTION(gpuGetErrorString, cudaGetErrorString)
GPUMOD_SAME_FUNCTION(gpuGetDevice, cudaGetDevice)
GPUMOD_SAME_FUNCTION(gpuSetDevice, cudaSetDevice)
GPUMOD_SAME_FUNCTION(gpuGetDeviceProperties, cudaGetDeviceProperties)
GPUMOD_SAME_FUNCTION(gpuStreamCreate, cudaStreamCreate)
GPUMOD_SAME_FUNCTION(gpuStreamCreateWithFlags, cudaStreamCreateWithFlags)
GPUMOD_SAME_FUNCTION(gpuStreamCreateWithPriority, cudaStreamCreateWithPriority)
GPUMOD_SAME_FUNCTION(gpuStreamGetFlags, cudaStreamGetFlags)
GPUMOD_SAME_FUNCTION(gpuStreamWaitEvent, cudaStreamWaitEvent)
GPUMOD_SAME_FUNCTION(gpuStreamSynchronize, cudaStreamSynchronize)
GPUMOD_SAME_FUNCTION(gpuStreamDestroy, cudaStreamDestroy)
GPUMOD_SAME_FUNCTION(gpuStreamBeginCapture, cudaStreamBeginCapture)
GPUMOD_SAME_FUNCTION(gpuStreamEndCapture, cudaStreamEndCapture)
GPUMOD_SAME_FUNCTION(gpuEventCreate, cudaEventCreate)
GPUMOD_SAME_FUNCTION(gpuEventCreateWithFlags, cudaEventCreateWithFlags)
GPUMOD_SAME_FUNCTION(gpuEventRecord, cudaEventRecord)
GPUMOD_SAME_FUNCTION(gpuEventRecordWithFlags, cudaEventRecordWithFlags)
GPUMOD_SAME_FUNCTION(gpuEventSynchronize, cudaEventSynchronize)
GPUMOD_SAME_FUNCTION(gpuEventQuery, cudaEventQuery)
GPUMOD_SAME_FUNCTION(gpuEventDestroy, cudaEventDestroy)
GPUMOD_SAME_FUNCTION(gpuMemPoolCreate, cudaMemPoolCreate)
GPUMOD_SAME_FUNCTION(gpuMemPoolSetAttribute, cudaMemPoolSetAttribute)
GPUMOD_SAME_FUNCTION(gpuMemPoolDestroy, cudaMemPoolDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphCreate, cudaGraphCreate)
GPUMOD_SAME_FUNCTION(gpuGraphDestroy, cudaGraphDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphExecDestroy, cudaGraphExecDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphLaunch, cudaGraphLaunch)
GPUMOD_SAME_FUNCTION(gpuGraphUpload, cudaGraphUpload)
// gpuGraphInstantiate is a hand-written forwarding function (its backend
// signatures diverge), not a plain alias, so &gpuGraphInstantiate is not any
// single backend symbol -- GPUMOD_SAME_FUNCTION does not apply. It is proven by
// compiling (the forward resolves on each backend) and by GpuGraphExecTests.

GPUMOD_SAME_TYPE(gpuMemcpyKind, cudaMemcpyKind)
GPUMOD_SAME_VALUE(gpuMemcpyHostToDevice, cudaMemcpyHostToDevice)
GPUMOD_SAME_VALUE(gpuMemcpyDeviceToHost, cudaMemcpyDeviceToHost)
GPUMOD_SAME_VALUE(gpuMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice)
GPUMOD_SAME_FUNCTION(gpuMalloc, cudaMalloc)
GPUMOD_SAME_FUNCTION(gpuFree, cudaFree)
GPUMOD_SAME_FUNCTION(gpuMemcpy, cudaMemcpy)
GPUMOD_SAME_FUNCTION(gpuMemset, cudaMemset)
GPUMOD_SAME_FUNCTION(gpuDeviceSynchronize, cudaDeviceSynchronize)

GPUMOD_SAME_VALUE(gpuErrorInvalidValue, cudaErrorInvalidValue)
GPUMOD_SAME_FUNCTION(gpuGetLastError, cudaGetLastError)

GPUMOD_SAME_VALUE(gpuMemcpyDefault, cudaMemcpyDefault)
GPUMOD_SAME_FUNCTION(gpuMallocAsync, cudaMallocAsync)
GPUMOD_SAME_FUNCTION(gpuMallocFromPoolAsync, cudaMallocFromPoolAsync)
GPUMOD_SAME_FUNCTION(gpuFreeAsync, cudaFreeAsync)
GPUMOD_SAME_FUNCTION(gpuMemcpyAsync, cudaMemcpyAsync)
GPUMOD_SAME_FUNCTION(gpuMemsetAsync, cudaMemsetAsync)

GPUMOD_SAME_VALUE(gpuHostAllocDefault, cudaHostAllocDefault)
GPUMOD_SAME_VALUE(gpuHostAllocMapped, cudaHostAllocMapped)
GPUMOD_SAME_VALUE(gpuHostAllocWriteCombined, cudaHostAllocWriteCombined)
GPUMOD_SAME_FUNCTION(gpuHostAlloc, cudaHostAlloc)
GPUMOD_SAME_FUNCTION(gpuFreeHost, cudaFreeHost)

GPUMOD_SAME_VALUE(gpuMemAttachGlobal, cudaMemAttachGlobal)
GPUMOD_SAME_VALUE(gpuMemAttachHost, cudaMemAttachHost)
GPUMOD_SAME_FUNCTION(gpuMallocManaged, cudaMallocManaged)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace gpumod::hip;

GPUMOD_SAME_TYPE(gpuError_t, hipError_t)
GPUMOD_SAME_TYPE(gpuStream_t, hipStream_t)
GPUMOD_SAME_TYPE(gpuStreamCaptureMode, hipStreamCaptureMode)
GPUMOD_SAME_TYPE(gpuEvent_t, hipEvent_t)
GPUMOD_SAME_TYPE(gpuMemPool_t, hipMemPool_t)
GPUMOD_SAME_TYPE(gpuMemPoolProps, hipMemPoolProps)
GPUMOD_SAME_TYPE(gpuGraph_t, hipGraph_t)
GPUMOD_SAME_TYPE(gpuGraphExec_t, hipGraphExec_t)
GPUMOD_SAME_TYPE(gpuDeviceProp, hipDeviceProp_tR0600)

GPUMOD_SAME_VALUE(gpuSuccess, hipSuccess)
GPUMOD_SAME_VALUE(gpuStreamDefault, hipStreamDefault)
GPUMOD_SAME_VALUE(gpuStreamNonBlocking, hipStreamNonBlocking)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeGlobal, hipStreamCaptureModeGlobal)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeThreadLocal, hipStreamCaptureModeThreadLocal)
GPUMOD_SAME_VALUE(gpuStreamCaptureModeRelaxed, hipStreamCaptureModeRelaxed)
GPUMOD_SAME_VALUE(gpuEventDefault, hipEventDefault)
GPUMOD_SAME_VALUE(gpuEventBlockingSync, hipEventBlockingSync)
GPUMOD_SAME_VALUE(gpuEventDisableTiming, hipEventDisableTiming)
GPUMOD_SAME_VALUE(gpuMemAllocationTypePinned, hipMemAllocationTypePinned)
GPUMOD_SAME_VALUE(gpuMemHandleTypeNone, hipMemHandleTypeNone)
GPUMOD_SAME_VALUE(gpuMemLocationTypeDevice, hipMemLocationTypeDevice)
GPUMOD_SAME_VALUE(gpuMemPoolAttrReleaseThreshold, hipMemPoolAttrReleaseThreshold)

GPUMOD_SAME_FUNCTION(gpuGetErrorName, hipGetErrorName)
GPUMOD_SAME_FUNCTION(gpuGetErrorString, hipGetErrorString)
GPUMOD_SAME_FUNCTION(gpuGetDevice, hipGetDevice)
GPUMOD_SAME_FUNCTION(gpuSetDevice, hipSetDevice)
GPUMOD_SAME_FUNCTION(gpuGetDeviceProperties, hipGetDevicePropertiesR0600)
GPUMOD_SAME_FUNCTION(gpuStreamCreate, hipStreamCreate)
GPUMOD_SAME_FUNCTION(gpuStreamCreateWithFlags, hipStreamCreateWithFlags)
GPUMOD_SAME_FUNCTION(gpuStreamCreateWithPriority, hipStreamCreateWithPriority)
GPUMOD_SAME_FUNCTION(gpuStreamGetFlags, hipStreamGetFlags)
GPUMOD_SAME_FUNCTION(gpuStreamWaitEvent, hipStreamWaitEvent)
GPUMOD_SAME_FUNCTION(gpuStreamSynchronize, hipStreamSynchronize)
GPUMOD_SAME_FUNCTION(gpuStreamDestroy, hipStreamDestroy)
GPUMOD_SAME_FUNCTION(gpuStreamBeginCapture, hipStreamBeginCapture)
GPUMOD_SAME_FUNCTION(gpuStreamEndCapture, hipStreamEndCapture)
GPUMOD_SAME_FUNCTION(gpuEventCreate, hipEventCreate)
GPUMOD_SAME_FUNCTION(gpuEventCreateWithFlags, hipEventCreateWithFlags)
GPUMOD_SAME_FUNCTION(gpuEventRecord, hipEventRecord)
GPUMOD_SAME_FUNCTION(gpuEventRecordWithFlags, hipEventRecordWithFlags)
GPUMOD_SAME_FUNCTION(gpuEventSynchronize, hipEventSynchronize)
GPUMOD_SAME_FUNCTION(gpuEventQuery, hipEventQuery)
GPUMOD_SAME_FUNCTION(gpuEventDestroy, hipEventDestroy)
GPUMOD_SAME_FUNCTION(gpuMemPoolCreate, hipMemPoolCreate)
GPUMOD_SAME_FUNCTION(gpuMemPoolSetAttribute, hipMemPoolSetAttribute)
GPUMOD_SAME_FUNCTION(gpuMemPoolDestroy, hipMemPoolDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphCreate, hipGraphCreate)
GPUMOD_SAME_FUNCTION(gpuGraphDestroy, hipGraphDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphExecDestroy, hipGraphExecDestroy)
GPUMOD_SAME_FUNCTION(gpuGraphLaunch, hipGraphLaunch)
GPUMOD_SAME_FUNCTION(gpuGraphUpload, hipGraphUpload)
// gpuGraphInstantiate forwards to hipGraphInstantiateWithFlags here (CUDA maps
// to cudaGraphInstantiate); as a hand-written forwarding function it is not a
// single backend symbol, so GPUMOD_SAME_FUNCTION does not apply. Proven by
// compiling and by GpuGraphExecTests.

GPUMOD_SAME_TYPE(gpuMemcpyKind, hipMemcpyKind)
GPUMOD_SAME_VALUE(gpuMemcpyHostToDevice, hipMemcpyHostToDevice)
GPUMOD_SAME_VALUE(gpuMemcpyDeviceToHost, hipMemcpyDeviceToHost)
GPUMOD_SAME_VALUE(gpuMemcpyDeviceToDevice, hipMemcpyDeviceToDevice)
// hipMalloc is an overload set (plus a template<class T> hipMalloc(T**, size_t));
// gpuMalloc must be its void** overload.
static_assert(&gpuMalloc == static_cast<hipError_t (*)(void **, std::size_t)>(&hipMalloc),
              "gpuMalloc is not hipMalloc(void**, size_t)");
GPUMOD_LINK_CHECK(gpuMalloc)
GPUMOD_SAME_FUNCTION(gpuFree, hipFree)
GPUMOD_SAME_FUNCTION(gpuMemcpy, hipMemcpy)
GPUMOD_SAME_FUNCTION(gpuMemset, hipMemset)
GPUMOD_SAME_FUNCTION(gpuDeviceSynchronize, hipDeviceSynchronize)

GPUMOD_SAME_VALUE(gpuErrorInvalidValue, hipErrorInvalidValue)
GPUMOD_SAME_FUNCTION(gpuGetLastError, hipGetLastError)

GPUMOD_SAME_VALUE(gpuMemcpyDefault, hipMemcpyDefault)
GPUMOD_SAME_FUNCTION(gpuMallocAsync, hipMallocAsync)
GPUMOD_SAME_FUNCTION(gpuMallocFromPoolAsync, hipMallocFromPoolAsync)
GPUMOD_SAME_FUNCTION(gpuFreeAsync, hipFreeAsync)
GPUMOD_SAME_FUNCTION(gpuMemcpyAsync, hipMemcpyAsync)
GPUMOD_SAME_FUNCTION(gpuMemsetAsync, hipMemsetAsync)

// hip_runtime_api.h's flag macros are hipHostMalloc*; hipHostAlloc* are
// equal-valued aliases.
GPUMOD_SAME_VALUE(gpuHostAllocDefault, hipHostMallocDefault)
GPUMOD_SAME_VALUE(gpuHostAllocMapped, hipHostMallocMapped)
GPUMOD_SAME_VALUE(gpuHostAllocWriteCombined, hipHostMallocWriteCombined)
// hipHostAlloc and hipMallocManaged are overload sets like hipMalloc; the gpu
// names must be their void** overloads.
static_assert(&gpuHostAlloc ==
                  static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(&hipHostAlloc),
              "gpuHostAlloc is not hipHostAlloc(void**, size_t, unsigned)");
GPUMOD_LINK_CHECK(gpuHostAlloc)
GPUMOD_SAME_FUNCTION(gpuFreeHost, hipFreeHost)

GPUMOD_SAME_VALUE(gpuMemAttachGlobal, hipMemAttachGlobal)
GPUMOD_SAME_VALUE(gpuMemAttachHost, hipMemAttachHost)
static_assert(&gpuMallocManaged == static_cast<hipError_t (*)(void **, std::size_t, unsigned int)>(
                                       &hipMallocManaged),
              "gpuMallocManaged is not hipMallocManaged(void**, size_t, unsigned)");
GPUMOD_LINK_CHECK(gpuMallocManaged)

#endif

} // namespace gpumod::test
