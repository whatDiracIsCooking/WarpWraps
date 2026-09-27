// fft.cppm - Compile-time tests for wwr.fft
//
// Every exported wwrfft* name is checked against the backend's own entity: the
// same type, the same constant (type and value), the same function (see
// gpu_check_macros.h). The base FFT surface is short enough to list in full,
// unlike test/gpu/blas.cppm and test/gpu/solver.cppm, which sample.
//
// The two direction flags carry a value check on top of the type check: cuFFT
// spells the inverse transform CUFFT_INVERSE and hipFFT spells it
// HIPFFT_BACKWARD, so WWRFFT_INVERSE maps to a differently-named constant per
// backend -- this pins that both are +1 and WWRFFT_FORWARD is -1.

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

WWR_SAME_TYPE(wwrfftHandle, cufftHandle)
WWR_SAME_TYPE(wwrfftResult_t, cufftResult_t)
WWR_SAME_TYPE(wwrfftType_t, cufftType_t)
WWR_SAME_TYPE(wwrfftReal, cufftReal)
WWR_SAME_TYPE(wwrfftDoubleReal, cufftDoubleReal)
WWR_SAME_TYPE(wwrfftComplex, cufftComplex)
WWR_SAME_TYPE(wwrfftDoubleComplex, cufftDoubleComplex)

WWR_SAME_VALUE(WWRFFT_FORWARD, CUFFT_FORWARD)
WWR_SAME_VALUE(WWRFFT_INVERSE, CUFFT_INVERSE)

WWR_SAME_VALUE(WWRFFT_R2C, CUFFT_R2C)
WWR_SAME_VALUE(WWRFFT_C2R, CUFFT_C2R)
WWR_SAME_VALUE(WWRFFT_C2C, CUFFT_C2C)
WWR_SAME_VALUE(WWRFFT_D2Z, CUFFT_D2Z)
WWR_SAME_VALUE(WWRFFT_Z2D, CUFFT_Z2D)
WWR_SAME_VALUE(WWRFFT_Z2Z, CUFFT_Z2Z)

WWR_SAME_VALUE(WWRFFT_SUCCESS, CUFFT_SUCCESS)
WWR_SAME_VALUE(WWRFFT_INVALID_PLAN, CUFFT_INVALID_PLAN)
WWR_SAME_VALUE(WWRFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED)
WWR_SAME_VALUE(WWRFFT_INVALID_TYPE, CUFFT_INVALID_TYPE)
WWR_SAME_VALUE(WWRFFT_INVALID_VALUE, CUFFT_INVALID_VALUE)
WWR_SAME_VALUE(WWRFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRFFT_EXEC_FAILED, CUFFT_EXEC_FAILED)
WWR_SAME_VALUE(WWRFFT_SETUP_FAILED, CUFFT_SETUP_FAILED)
WWR_SAME_VALUE(WWRFFT_INVALID_SIZE, CUFFT_INVALID_SIZE)
WWR_SAME_VALUE(WWRFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA)
WWR_SAME_VALUE(WWRFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE)
WWR_SAME_VALUE(WWRFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE)
WWR_SAME_VALUE(WWRFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED)
WWR_SAME_VALUE(WWRFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED)

WWR_SAME_FUNCTION(wwrfftCreate, cufftCreate)
WWR_SAME_FUNCTION(wwrfftDestroy, cufftDestroy)
WWR_SAME_FUNCTION(wwrfftSetStream, cufftSetStream)
WWR_SAME_FUNCTION(wwrfftGetVersion, cufftGetVersion)

WWR_SAME_FUNCTION(wwrfftPlan1d, cufftPlan1d)
WWR_SAME_FUNCTION(wwrfftPlan2d, cufftPlan2d)
WWR_SAME_FUNCTION(wwrfftPlan3d, cufftPlan3d)
WWR_SAME_FUNCTION(wwrfftPlanMany, cufftPlanMany)

WWR_SAME_FUNCTION(wwrfftMakePlan1d, cufftMakePlan1d)
WWR_SAME_FUNCTION(wwrfftMakePlan2d, cufftMakePlan2d)
WWR_SAME_FUNCTION(wwrfftMakePlan3d, cufftMakePlan3d)
WWR_SAME_FUNCTION(wwrfftMakePlanMany, cufftMakePlanMany)
WWR_SAME_FUNCTION(wwrfftMakePlanMany64, cufftMakePlanMany64)

WWR_SAME_FUNCTION(wwrfftEstimate1d, cufftEstimate1d)
WWR_SAME_FUNCTION(wwrfftEstimate2d, cufftEstimate2d)
WWR_SAME_FUNCTION(wwrfftEstimate3d, cufftEstimate3d)
WWR_SAME_FUNCTION(wwrfftEstimateMany, cufftEstimateMany)

WWR_SAME_FUNCTION(wwrfftGetSize1d, cufftGetSize1d)
WWR_SAME_FUNCTION(wwrfftGetSize2d, cufftGetSize2d)
WWR_SAME_FUNCTION(wwrfftGetSize3d, cufftGetSize3d)
WWR_SAME_FUNCTION(wwrfftGetSizeMany, cufftGetSizeMany)
WWR_SAME_FUNCTION(wwrfftGetSizeMany64, cufftGetSizeMany64)
WWR_SAME_FUNCTION(wwrfftGetSize, cufftGetSize)

WWR_SAME_FUNCTION(wwrfftSetWorkArea, cufftSetWorkArea)
WWR_SAME_FUNCTION(wwrfftSetAutoAllocation, cufftSetAutoAllocation)

WWR_SAME_FUNCTION(wwrfftExecC2C, cufftExecC2C)
WWR_SAME_FUNCTION(wwrfftExecR2C, cufftExecR2C)
WWR_SAME_FUNCTION(wwrfftExecC2R, cufftExecC2R)
WWR_SAME_FUNCTION(wwrfftExecZ2Z, cufftExecZ2Z)
WWR_SAME_FUNCTION(wwrfftExecD2Z, cufftExecD2Z)
WWR_SAME_FUNCTION(wwrfftExecZ2D, cufftExecZ2D)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrfftHandle, hipfftHandle)
WWR_SAME_TYPE(wwrfftResult_t, hipfftResult_t)
WWR_SAME_TYPE(wwrfftType_t, hipfftType_t)
WWR_SAME_TYPE(wwrfftReal, hipfftReal)
WWR_SAME_TYPE(wwrfftDoubleReal, hipfftDoubleReal)
WWR_SAME_TYPE(wwrfftComplex, hipfftComplex)
WWR_SAME_TYPE(wwrfftDoubleComplex, hipfftDoubleComplex)

// hipFFT names the inverse transform BACKWARD, not INVERSE.
WWR_SAME_VALUE(WWRFFT_FORWARD, HIPFFT_FORWARD)
WWR_SAME_VALUE(WWRFFT_INVERSE, HIPFFT_BACKWARD)

WWR_SAME_VALUE(WWRFFT_R2C, HIPFFT_R2C)
WWR_SAME_VALUE(WWRFFT_C2R, HIPFFT_C2R)
WWR_SAME_VALUE(WWRFFT_C2C, HIPFFT_C2C)
WWR_SAME_VALUE(WWRFFT_D2Z, HIPFFT_D2Z)
WWR_SAME_VALUE(WWRFFT_Z2D, HIPFFT_Z2D)
WWR_SAME_VALUE(WWRFFT_Z2Z, HIPFFT_Z2Z)

WWR_SAME_VALUE(WWRFFT_SUCCESS, HIPFFT_SUCCESS)
WWR_SAME_VALUE(WWRFFT_INVALID_PLAN, HIPFFT_INVALID_PLAN)
WWR_SAME_VALUE(WWRFFT_ALLOC_FAILED, HIPFFT_ALLOC_FAILED)
WWR_SAME_VALUE(WWRFFT_INVALID_TYPE, HIPFFT_INVALID_TYPE)
WWR_SAME_VALUE(WWRFFT_INVALID_VALUE, HIPFFT_INVALID_VALUE)
WWR_SAME_VALUE(WWRFFT_INTERNAL_ERROR, HIPFFT_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRFFT_EXEC_FAILED, HIPFFT_EXEC_FAILED)
WWR_SAME_VALUE(WWRFFT_SETUP_FAILED, HIPFFT_SETUP_FAILED)
WWR_SAME_VALUE(WWRFFT_INVALID_SIZE, HIPFFT_INVALID_SIZE)
WWR_SAME_VALUE(WWRFFT_UNALIGNED_DATA, HIPFFT_UNALIGNED_DATA)
WWR_SAME_VALUE(WWRFFT_INVALID_DEVICE, HIPFFT_INVALID_DEVICE)
WWR_SAME_VALUE(WWRFFT_NO_WORKSPACE, HIPFFT_NO_WORKSPACE)
WWR_SAME_VALUE(WWRFFT_NOT_IMPLEMENTED, HIPFFT_NOT_IMPLEMENTED)
WWR_SAME_VALUE(WWRFFT_NOT_SUPPORTED, HIPFFT_NOT_SUPPORTED)

WWR_SAME_FUNCTION(wwrfftCreate, hipfftCreate)
WWR_SAME_FUNCTION(wwrfftDestroy, hipfftDestroy)
WWR_SAME_FUNCTION(wwrfftSetStream, hipfftSetStream)
WWR_SAME_FUNCTION(wwrfftGetVersion, hipfftGetVersion)

WWR_SAME_FUNCTION(wwrfftPlan1d, hipfftPlan1d)
WWR_SAME_FUNCTION(wwrfftPlan2d, hipfftPlan2d)
WWR_SAME_FUNCTION(wwrfftPlan3d, hipfftPlan3d)
WWR_SAME_FUNCTION(wwrfftPlanMany, hipfftPlanMany)

WWR_SAME_FUNCTION(wwrfftMakePlan1d, hipfftMakePlan1d)
WWR_SAME_FUNCTION(wwrfftMakePlan2d, hipfftMakePlan2d)
WWR_SAME_FUNCTION(wwrfftMakePlan3d, hipfftMakePlan3d)
WWR_SAME_FUNCTION(wwrfftMakePlanMany, hipfftMakePlanMany)
WWR_SAME_FUNCTION(wwrfftMakePlanMany64, hipfftMakePlanMany64)

WWR_SAME_FUNCTION(wwrfftEstimate1d, hipfftEstimate1d)
WWR_SAME_FUNCTION(wwrfftEstimate2d, hipfftEstimate2d)
WWR_SAME_FUNCTION(wwrfftEstimate3d, hipfftEstimate3d)
WWR_SAME_FUNCTION(wwrfftEstimateMany, hipfftEstimateMany)

WWR_SAME_FUNCTION(wwrfftGetSize1d, hipfftGetSize1d)
WWR_SAME_FUNCTION(wwrfftGetSize2d, hipfftGetSize2d)
WWR_SAME_FUNCTION(wwrfftGetSize3d, hipfftGetSize3d)
WWR_SAME_FUNCTION(wwrfftGetSizeMany, hipfftGetSizeMany)
WWR_SAME_FUNCTION(wwrfftGetSizeMany64, hipfftGetSizeMany64)
WWR_SAME_FUNCTION(wwrfftGetSize, hipfftGetSize)

WWR_SAME_FUNCTION(wwrfftSetWorkArea, hipfftSetWorkArea)
WWR_SAME_FUNCTION(wwrfftSetAutoAllocation, hipfftSetAutoAllocation)

WWR_SAME_FUNCTION(wwrfftExecC2C, hipfftExecC2C)
WWR_SAME_FUNCTION(wwrfftExecR2C, hipfftExecR2C)
WWR_SAME_FUNCTION(wwrfftExecC2R, hipfftExecC2R)
WWR_SAME_FUNCTION(wwrfftExecZ2Z, hipfftExecZ2Z)
WWR_SAME_FUNCTION(wwrfftExecD2Z, hipfftExecD2Z)
WWR_SAME_FUNCTION(wwrfftExecZ2D, hipfftExecZ2D)

#endif

} // namespace wwr::test
