// fft.cppm - Compile-time tests for gpumod.fft
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

export module gpumod.test.gpu.fft;

import std;
import gpumod.fft;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cufft;
#else
import gpumod.hip.hipfft;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

GPUMOD_SAME_TYPE(gpufftHandle, cufftHandle)
GPUMOD_SAME_TYPE(gpufftResult_t, cufftResult_t)
GPUMOD_SAME_TYPE(gpufftType_t, cufftType_t)
GPUMOD_SAME_TYPE(gpufftReal, cufftReal)
GPUMOD_SAME_TYPE(gpufftDoubleReal, cufftDoubleReal)
GPUMOD_SAME_TYPE(gpufftComplex, cufftComplex)
GPUMOD_SAME_TYPE(gpufftDoubleComplex, cufftDoubleComplex)

GPUMOD_SAME_VALUE(GPUFFT_FORWARD, CUFFT_FORWARD)
GPUMOD_SAME_VALUE(GPUFFT_INVERSE, CUFFT_INVERSE)

GPUMOD_SAME_VALUE(GPUFFT_R2C, CUFFT_R2C)
GPUMOD_SAME_VALUE(GPUFFT_C2R, CUFFT_C2R)
GPUMOD_SAME_VALUE(GPUFFT_C2C, CUFFT_C2C)
GPUMOD_SAME_VALUE(GPUFFT_D2Z, CUFFT_D2Z)
GPUMOD_SAME_VALUE(GPUFFT_Z2D, CUFFT_Z2D)
GPUMOD_SAME_VALUE(GPUFFT_Z2Z, CUFFT_Z2Z)

GPUMOD_SAME_VALUE(GPUFFT_SUCCESS, CUFFT_SUCCESS)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_PLAN, CUFFT_INVALID_PLAN)
GPUMOD_SAME_VALUE(GPUFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_TYPE, CUFFT_INVALID_TYPE)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_VALUE, CUFFT_INVALID_VALUE)
GPUMOD_SAME_VALUE(GPUFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR)
GPUMOD_SAME_VALUE(GPUFFT_EXEC_FAILED, CUFFT_EXEC_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_SETUP_FAILED, CUFFT_SETUP_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_SIZE, CUFFT_INVALID_SIZE)
GPUMOD_SAME_VALUE(GPUFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE)
GPUMOD_SAME_VALUE(GPUFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE)
GPUMOD_SAME_VALUE(GPUFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED)
GPUMOD_SAME_VALUE(GPUFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED)

GPUMOD_SAME_FUNCTION(gpufftCreate, cufftCreate)
GPUMOD_SAME_FUNCTION(gpufftDestroy, cufftDestroy)
GPUMOD_SAME_FUNCTION(gpufftSetStream, cufftSetStream)
GPUMOD_SAME_FUNCTION(gpufftGetVersion, cufftGetVersion)

GPUMOD_SAME_FUNCTION(gpufftPlan1d, cufftPlan1d)
GPUMOD_SAME_FUNCTION(gpufftPlan2d, cufftPlan2d)
GPUMOD_SAME_FUNCTION(gpufftPlan3d, cufftPlan3d)
GPUMOD_SAME_FUNCTION(gpufftPlanMany, cufftPlanMany)

GPUMOD_SAME_FUNCTION(gpufftMakePlan1d, cufftMakePlan1d)
GPUMOD_SAME_FUNCTION(gpufftMakePlan2d, cufftMakePlan2d)
GPUMOD_SAME_FUNCTION(gpufftMakePlan3d, cufftMakePlan3d)
GPUMOD_SAME_FUNCTION(gpufftMakePlanMany, cufftMakePlanMany)
GPUMOD_SAME_FUNCTION(gpufftMakePlanMany64, cufftMakePlanMany64)

GPUMOD_SAME_FUNCTION(gpufftEstimate1d, cufftEstimate1d)
GPUMOD_SAME_FUNCTION(gpufftEstimate2d, cufftEstimate2d)
GPUMOD_SAME_FUNCTION(gpufftEstimate3d, cufftEstimate3d)
GPUMOD_SAME_FUNCTION(gpufftEstimateMany, cufftEstimateMany)

GPUMOD_SAME_FUNCTION(gpufftGetSize1d, cufftGetSize1d)
GPUMOD_SAME_FUNCTION(gpufftGetSize2d, cufftGetSize2d)
GPUMOD_SAME_FUNCTION(gpufftGetSize3d, cufftGetSize3d)
GPUMOD_SAME_FUNCTION(gpufftGetSizeMany, cufftGetSizeMany)
GPUMOD_SAME_FUNCTION(gpufftGetSizeMany64, cufftGetSizeMany64)
GPUMOD_SAME_FUNCTION(gpufftGetSize, cufftGetSize)

GPUMOD_SAME_FUNCTION(gpufftSetWorkArea, cufftSetWorkArea)
GPUMOD_SAME_FUNCTION(gpufftSetAutoAllocation, cufftSetAutoAllocation)

GPUMOD_SAME_FUNCTION(gpufftExecC2C, cufftExecC2C)
GPUMOD_SAME_FUNCTION(gpufftExecR2C, cufftExecR2C)
GPUMOD_SAME_FUNCTION(gpufftExecC2R, cufftExecC2R)
GPUMOD_SAME_FUNCTION(gpufftExecZ2Z, cufftExecZ2Z)
GPUMOD_SAME_FUNCTION(gpufftExecD2Z, cufftExecD2Z)
GPUMOD_SAME_FUNCTION(gpufftExecZ2D, cufftExecZ2D)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace gpumod::hip;

GPUMOD_SAME_TYPE(gpufftHandle, hipfftHandle)
GPUMOD_SAME_TYPE(gpufftResult_t, hipfftResult_t)
GPUMOD_SAME_TYPE(gpufftType_t, hipfftType_t)
GPUMOD_SAME_TYPE(gpufftReal, hipfftReal)
GPUMOD_SAME_TYPE(gpufftDoubleReal, hipfftDoubleReal)
GPUMOD_SAME_TYPE(gpufftComplex, hipfftComplex)
GPUMOD_SAME_TYPE(gpufftDoubleComplex, hipfftDoubleComplex)

// hipFFT names the inverse transform BACKWARD, not INVERSE.
GPUMOD_SAME_VALUE(GPUFFT_FORWARD, HIPFFT_FORWARD)
GPUMOD_SAME_VALUE(GPUFFT_INVERSE, HIPFFT_BACKWARD)

GPUMOD_SAME_VALUE(GPUFFT_R2C, HIPFFT_R2C)
GPUMOD_SAME_VALUE(GPUFFT_C2R, HIPFFT_C2R)
GPUMOD_SAME_VALUE(GPUFFT_C2C, HIPFFT_C2C)
GPUMOD_SAME_VALUE(GPUFFT_D2Z, HIPFFT_D2Z)
GPUMOD_SAME_VALUE(GPUFFT_Z2D, HIPFFT_Z2D)
GPUMOD_SAME_VALUE(GPUFFT_Z2Z, HIPFFT_Z2Z)

GPUMOD_SAME_VALUE(GPUFFT_SUCCESS, HIPFFT_SUCCESS)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_PLAN, HIPFFT_INVALID_PLAN)
GPUMOD_SAME_VALUE(GPUFFT_ALLOC_FAILED, HIPFFT_ALLOC_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_TYPE, HIPFFT_INVALID_TYPE)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_VALUE, HIPFFT_INVALID_VALUE)
GPUMOD_SAME_VALUE(GPUFFT_INTERNAL_ERROR, HIPFFT_INTERNAL_ERROR)
GPUMOD_SAME_VALUE(GPUFFT_EXEC_FAILED, HIPFFT_EXEC_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_SETUP_FAILED, HIPFFT_SETUP_FAILED)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_SIZE, HIPFFT_INVALID_SIZE)
GPUMOD_SAME_VALUE(GPUFFT_UNALIGNED_DATA, HIPFFT_UNALIGNED_DATA)
GPUMOD_SAME_VALUE(GPUFFT_INVALID_DEVICE, HIPFFT_INVALID_DEVICE)
GPUMOD_SAME_VALUE(GPUFFT_NO_WORKSPACE, HIPFFT_NO_WORKSPACE)
GPUMOD_SAME_VALUE(GPUFFT_NOT_IMPLEMENTED, HIPFFT_NOT_IMPLEMENTED)
GPUMOD_SAME_VALUE(GPUFFT_NOT_SUPPORTED, HIPFFT_NOT_SUPPORTED)

GPUMOD_SAME_FUNCTION(gpufftCreate, hipfftCreate)
GPUMOD_SAME_FUNCTION(gpufftDestroy, hipfftDestroy)
GPUMOD_SAME_FUNCTION(gpufftSetStream, hipfftSetStream)
GPUMOD_SAME_FUNCTION(gpufftGetVersion, hipfftGetVersion)

GPUMOD_SAME_FUNCTION(gpufftPlan1d, hipfftPlan1d)
GPUMOD_SAME_FUNCTION(gpufftPlan2d, hipfftPlan2d)
GPUMOD_SAME_FUNCTION(gpufftPlan3d, hipfftPlan3d)
GPUMOD_SAME_FUNCTION(gpufftPlanMany, hipfftPlanMany)

GPUMOD_SAME_FUNCTION(gpufftMakePlan1d, hipfftMakePlan1d)
GPUMOD_SAME_FUNCTION(gpufftMakePlan2d, hipfftMakePlan2d)
GPUMOD_SAME_FUNCTION(gpufftMakePlan3d, hipfftMakePlan3d)
GPUMOD_SAME_FUNCTION(gpufftMakePlanMany, hipfftMakePlanMany)
GPUMOD_SAME_FUNCTION(gpufftMakePlanMany64, hipfftMakePlanMany64)

GPUMOD_SAME_FUNCTION(gpufftEstimate1d, hipfftEstimate1d)
GPUMOD_SAME_FUNCTION(gpufftEstimate2d, hipfftEstimate2d)
GPUMOD_SAME_FUNCTION(gpufftEstimate3d, hipfftEstimate3d)
GPUMOD_SAME_FUNCTION(gpufftEstimateMany, hipfftEstimateMany)

GPUMOD_SAME_FUNCTION(gpufftGetSize1d, hipfftGetSize1d)
GPUMOD_SAME_FUNCTION(gpufftGetSize2d, hipfftGetSize2d)
GPUMOD_SAME_FUNCTION(gpufftGetSize3d, hipfftGetSize3d)
GPUMOD_SAME_FUNCTION(gpufftGetSizeMany, hipfftGetSizeMany)
GPUMOD_SAME_FUNCTION(gpufftGetSizeMany64, hipfftGetSizeMany64)
GPUMOD_SAME_FUNCTION(gpufftGetSize, hipfftGetSize)

GPUMOD_SAME_FUNCTION(gpufftSetWorkArea, hipfftSetWorkArea)
GPUMOD_SAME_FUNCTION(gpufftSetAutoAllocation, hipfftSetAutoAllocation)

GPUMOD_SAME_FUNCTION(gpufftExecC2C, hipfftExecC2C)
GPUMOD_SAME_FUNCTION(gpufftExecR2C, hipfftExecR2C)
GPUMOD_SAME_FUNCTION(gpufftExecC2R, hipfftExecC2R)
GPUMOD_SAME_FUNCTION(gpufftExecZ2Z, hipfftExecZ2Z)
GPUMOD_SAME_FUNCTION(gpufftExecD2Z, hipfftExecD2Z)
GPUMOD_SAME_FUNCTION(gpufftExecZ2D, hipfftExecZ2D)

#endif

} // namespace gpumod::test
