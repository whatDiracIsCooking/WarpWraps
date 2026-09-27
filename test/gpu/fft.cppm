// fft.cppm - Compile-time tests for wwr.fft
//
// Every exported gpufft* name is checked against the backend's own entity: the
// same type, the same constant (type and value), the same function (see
// gpu_check_macros.h). The base FFT surface is short enough to list in full,
// unlike test/gpu/blas.cppm and test/gpu/solver.cppm, which sample.
//
// The two direction flags carry a value check on top of the type check: cuFFT
// spells the inverse transform CUFFT_INVERSE and hipFFT spells it
// HIPFFT_BACKWARD, so GPUFFT_INVERSE maps to a differently-named constant per
// backend -- this pins that both are +1 and GPUFFT_FORWARD is -1.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.fft;

import std;
import wwr.fft;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cufft;
#else
import wwr.hip.hipfft;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(gpufftHandle, cufftHandle)
WWR_SAME_TYPE(gpufftResult_t, cufftResult_t)
WWR_SAME_TYPE(gpufftType_t, cufftType_t)
WWR_SAME_TYPE(gpufftReal, cufftReal)
WWR_SAME_TYPE(gpufftDoubleReal, cufftDoubleReal)
WWR_SAME_TYPE(gpufftComplex, cufftComplex)
WWR_SAME_TYPE(gpufftDoubleComplex, cufftDoubleComplex)

WWR_SAME_VALUE(GPUFFT_FORWARD, CUFFT_FORWARD)
WWR_SAME_VALUE(GPUFFT_INVERSE, CUFFT_INVERSE)

WWR_SAME_VALUE(GPUFFT_R2C, CUFFT_R2C)
WWR_SAME_VALUE(GPUFFT_C2R, CUFFT_C2R)
WWR_SAME_VALUE(GPUFFT_C2C, CUFFT_C2C)
WWR_SAME_VALUE(GPUFFT_D2Z, CUFFT_D2Z)
WWR_SAME_VALUE(GPUFFT_Z2D, CUFFT_Z2D)
WWR_SAME_VALUE(GPUFFT_Z2Z, CUFFT_Z2Z)

WWR_SAME_VALUE(GPUFFT_SUCCESS, CUFFT_SUCCESS)
WWR_SAME_VALUE(GPUFFT_INVALID_PLAN, CUFFT_INVALID_PLAN)
WWR_SAME_VALUE(GPUFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED)
WWR_SAME_VALUE(GPUFFT_INVALID_TYPE, CUFFT_INVALID_TYPE)
WWR_SAME_VALUE(GPUFFT_INVALID_VALUE, CUFFT_INVALID_VALUE)
WWR_SAME_VALUE(GPUFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR)
WWR_SAME_VALUE(GPUFFT_EXEC_FAILED, CUFFT_EXEC_FAILED)
WWR_SAME_VALUE(GPUFFT_SETUP_FAILED, CUFFT_SETUP_FAILED)
WWR_SAME_VALUE(GPUFFT_INVALID_SIZE, CUFFT_INVALID_SIZE)
WWR_SAME_VALUE(GPUFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA)
WWR_SAME_VALUE(GPUFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE)
WWR_SAME_VALUE(GPUFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE)
WWR_SAME_VALUE(GPUFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED)
WWR_SAME_VALUE(GPUFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED)

WWR_SAME_FUNCTION(gpufftCreate, cufftCreate)
WWR_SAME_FUNCTION(gpufftDestroy, cufftDestroy)
WWR_SAME_FUNCTION(gpufftSetStream, cufftSetStream)
WWR_SAME_FUNCTION(gpufftGetVersion, cufftGetVersion)

WWR_SAME_FUNCTION(gpufftPlan1d, cufftPlan1d)
WWR_SAME_FUNCTION(gpufftPlan2d, cufftPlan2d)
WWR_SAME_FUNCTION(gpufftPlan3d, cufftPlan3d)
WWR_SAME_FUNCTION(gpufftPlanMany, cufftPlanMany)

WWR_SAME_FUNCTION(gpufftMakePlan1d, cufftMakePlan1d)
WWR_SAME_FUNCTION(gpufftMakePlan2d, cufftMakePlan2d)
WWR_SAME_FUNCTION(gpufftMakePlan3d, cufftMakePlan3d)
WWR_SAME_FUNCTION(gpufftMakePlanMany, cufftMakePlanMany)
WWR_SAME_FUNCTION(gpufftMakePlanMany64, cufftMakePlanMany64)

WWR_SAME_FUNCTION(gpufftEstimate1d, cufftEstimate1d)
WWR_SAME_FUNCTION(gpufftEstimate2d, cufftEstimate2d)
WWR_SAME_FUNCTION(gpufftEstimate3d, cufftEstimate3d)
WWR_SAME_FUNCTION(gpufftEstimateMany, cufftEstimateMany)

WWR_SAME_FUNCTION(gpufftGetSize1d, cufftGetSize1d)
WWR_SAME_FUNCTION(gpufftGetSize2d, cufftGetSize2d)
WWR_SAME_FUNCTION(gpufftGetSize3d, cufftGetSize3d)
WWR_SAME_FUNCTION(gpufftGetSizeMany, cufftGetSizeMany)
WWR_SAME_FUNCTION(gpufftGetSizeMany64, cufftGetSizeMany64)
WWR_SAME_FUNCTION(gpufftGetSize, cufftGetSize)

WWR_SAME_FUNCTION(gpufftSetWorkArea, cufftSetWorkArea)
WWR_SAME_FUNCTION(gpufftSetAutoAllocation, cufftSetAutoAllocation)

WWR_SAME_FUNCTION(gpufftExecC2C, cufftExecC2C)
WWR_SAME_FUNCTION(gpufftExecR2C, cufftExecR2C)
WWR_SAME_FUNCTION(gpufftExecC2R, cufftExecC2R)
WWR_SAME_FUNCTION(gpufftExecZ2Z, cufftExecZ2Z)
WWR_SAME_FUNCTION(gpufftExecD2Z, cufftExecD2Z)
WWR_SAME_FUNCTION(gpufftExecZ2D, cufftExecZ2D)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gpufftHandle, hipfftHandle)
WWR_SAME_TYPE(gpufftResult_t, hipfftResult_t)
WWR_SAME_TYPE(gpufftType_t, hipfftType_t)
WWR_SAME_TYPE(gpufftReal, hipfftReal)
WWR_SAME_TYPE(gpufftDoubleReal, hipfftDoubleReal)
WWR_SAME_TYPE(gpufftComplex, hipfftComplex)
WWR_SAME_TYPE(gpufftDoubleComplex, hipfftDoubleComplex)

// hipFFT names the inverse transform BACKWARD, not INVERSE.
WWR_SAME_VALUE(GPUFFT_FORWARD, HIPFFT_FORWARD)
WWR_SAME_VALUE(GPUFFT_INVERSE, HIPFFT_BACKWARD)

WWR_SAME_VALUE(GPUFFT_R2C, HIPFFT_R2C)
WWR_SAME_VALUE(GPUFFT_C2R, HIPFFT_C2R)
WWR_SAME_VALUE(GPUFFT_C2C, HIPFFT_C2C)
WWR_SAME_VALUE(GPUFFT_D2Z, HIPFFT_D2Z)
WWR_SAME_VALUE(GPUFFT_Z2D, HIPFFT_Z2D)
WWR_SAME_VALUE(GPUFFT_Z2Z, HIPFFT_Z2Z)

WWR_SAME_VALUE(GPUFFT_SUCCESS, HIPFFT_SUCCESS)
WWR_SAME_VALUE(GPUFFT_INVALID_PLAN, HIPFFT_INVALID_PLAN)
WWR_SAME_VALUE(GPUFFT_ALLOC_FAILED, HIPFFT_ALLOC_FAILED)
WWR_SAME_VALUE(GPUFFT_INVALID_TYPE, HIPFFT_INVALID_TYPE)
WWR_SAME_VALUE(GPUFFT_INVALID_VALUE, HIPFFT_INVALID_VALUE)
WWR_SAME_VALUE(GPUFFT_INTERNAL_ERROR, HIPFFT_INTERNAL_ERROR)
WWR_SAME_VALUE(GPUFFT_EXEC_FAILED, HIPFFT_EXEC_FAILED)
WWR_SAME_VALUE(GPUFFT_SETUP_FAILED, HIPFFT_SETUP_FAILED)
WWR_SAME_VALUE(GPUFFT_INVALID_SIZE, HIPFFT_INVALID_SIZE)
WWR_SAME_VALUE(GPUFFT_UNALIGNED_DATA, HIPFFT_UNALIGNED_DATA)
WWR_SAME_VALUE(GPUFFT_INVALID_DEVICE, HIPFFT_INVALID_DEVICE)
WWR_SAME_VALUE(GPUFFT_NO_WORKSPACE, HIPFFT_NO_WORKSPACE)
WWR_SAME_VALUE(GPUFFT_NOT_IMPLEMENTED, HIPFFT_NOT_IMPLEMENTED)
WWR_SAME_VALUE(GPUFFT_NOT_SUPPORTED, HIPFFT_NOT_SUPPORTED)

WWR_SAME_FUNCTION(gpufftCreate, hipfftCreate)
WWR_SAME_FUNCTION(gpufftDestroy, hipfftDestroy)
WWR_SAME_FUNCTION(gpufftSetStream, hipfftSetStream)
WWR_SAME_FUNCTION(gpufftGetVersion, hipfftGetVersion)

WWR_SAME_FUNCTION(gpufftPlan1d, hipfftPlan1d)
WWR_SAME_FUNCTION(gpufftPlan2d, hipfftPlan2d)
WWR_SAME_FUNCTION(gpufftPlan3d, hipfftPlan3d)
WWR_SAME_FUNCTION(gpufftPlanMany, hipfftPlanMany)

WWR_SAME_FUNCTION(gpufftMakePlan1d, hipfftMakePlan1d)
WWR_SAME_FUNCTION(gpufftMakePlan2d, hipfftMakePlan2d)
WWR_SAME_FUNCTION(gpufftMakePlan3d, hipfftMakePlan3d)
WWR_SAME_FUNCTION(gpufftMakePlanMany, hipfftMakePlanMany)
WWR_SAME_FUNCTION(gpufftMakePlanMany64, hipfftMakePlanMany64)

WWR_SAME_FUNCTION(gpufftEstimate1d, hipfftEstimate1d)
WWR_SAME_FUNCTION(gpufftEstimate2d, hipfftEstimate2d)
WWR_SAME_FUNCTION(gpufftEstimate3d, hipfftEstimate3d)
WWR_SAME_FUNCTION(gpufftEstimateMany, hipfftEstimateMany)

WWR_SAME_FUNCTION(gpufftGetSize1d, hipfftGetSize1d)
WWR_SAME_FUNCTION(gpufftGetSize2d, hipfftGetSize2d)
WWR_SAME_FUNCTION(gpufftGetSize3d, hipfftGetSize3d)
WWR_SAME_FUNCTION(gpufftGetSizeMany, hipfftGetSizeMany)
WWR_SAME_FUNCTION(gpufftGetSizeMany64, hipfftGetSizeMany64)
WWR_SAME_FUNCTION(gpufftGetSize, hipfftGetSize)

WWR_SAME_FUNCTION(gpufftSetWorkArea, hipfftSetWorkArea)
WWR_SAME_FUNCTION(gpufftSetAutoAllocation, hipfftSetAutoAllocation)

WWR_SAME_FUNCTION(gpufftExecC2C, hipfftExecC2C)
WWR_SAME_FUNCTION(gpufftExecR2C, hipfftExecR2C)
WWR_SAME_FUNCTION(gpufftExecC2R, hipfftExecC2R)
WWR_SAME_FUNCTION(gpufftExecZ2Z, hipfftExecZ2Z)
WWR_SAME_FUNCTION(gpufftExecD2Z, hipfftExecD2Z)
WWR_SAME_FUNCTION(gpufftExecZ2D, hipfftExecZ2D)

#endif

} // namespace wwr::test
