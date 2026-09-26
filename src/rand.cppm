/**
 * @file rand.cppm
 * @brief Backend-neutral random number generation: gpurand* names for cuRAND / hipRAND
 *
 * The cuRAND / hipRAND host API, plus the device generator STATE TYPES and
 * nothing else of the device API: device functions are __device__-qualified
 * and live in rand.cuh, while the state types are plain data and the host is
 * what sizes the per-thread state array. gpurand<X> stands for curand<X> /
 * hiprand<X>, GPURAND_<X> for CURAND_<X> / HIPRAND_<X>, each written out in
 * full. See gpu_backend.h.
 *
 * Everything the two host APIs share is listed. Absent for want of a
 * counterpart -- cuRAND: curandGetProperty, curandGeneratePoissonMethod,
 * curandGenerateBinomial(Method), curandMethod_t, curandDistribution_t;
 * hipRAND: hiprandGenerateChar/Short, the *Half generators,
 * HIPRAND_STATUS_NOT_IMPLEMENTED. Neither has a status-to-string function.
 *
 * gpurandState and gpurandStateXORWOW are one type on CUDA and two on HIP, and
 * enumerator VALUES differ even where names agree. See docs/architecture.md,
 * sections 1 and 6.
 *
 * Usage:
 *   import gpumod.rand;
 *
 *   gpurandGenerator_t gen;
 *   gpurandCreateGenerator(&gen, GPURAND_RNG_PSEUDO_DEFAULT);
 */

module;

#include "gpu_backend.h"

export module gpumod.rand;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.curand;
#else
import gpumod.hip.hiprand;
import gpumod.hip.hiprand_kernel;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_TYPE(gpurandGenerator_t, curandGenerator_t, hiprandGenerator_t)
GPUMOD_TYPE(gpurandStatus_t, curandStatus_t, hiprandStatus_t)
GPUMOD_TYPE(gpurandRngType_t, curandRngType_t, hiprandRngType_t)
GPUMOD_TYPE(gpurandOrdering_t, curandOrdering_t, hiprandOrdering_t)
GPUMOD_TYPE(gpurandDirectionVectorSet_t, curandDirectionVectorSet_t, hiprandDirectionVectorSet_t)
GPUMOD_TYPE(gpurandDirectionVectors32_t, curandDirectionVectors32_t, hiprandDirectionVectors32_t)
GPUMOD_TYPE(gpurandDirectionVectors64_t, curandDirectionVectors64_t, hiprandDirectionVectors64_t)
GPUMOD_TYPE(gpurandDiscreteDistribution_t, curandDiscreteDistribution_t,
            hiprandDiscreteDistribution_t)

// ========================================================================
// Constants - status
// ========================================================================

GPUMOD_VALUE(GPURAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
GPUMOD_VALUE(GPURAND_STATUS_VERSION_MISMATCH, CURAND_STATUS_VERSION_MISMATCH,
             HIPRAND_STATUS_VERSION_MISMATCH)
GPUMOD_VALUE(GPURAND_STATUS_NOT_INITIALIZED, CURAND_STATUS_NOT_INITIALIZED,
             HIPRAND_STATUS_NOT_INITIALIZED)
GPUMOD_VALUE(GPURAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_ALLOCATION_FAILED,
             HIPRAND_STATUS_ALLOCATION_FAILED)
GPUMOD_VALUE(GPURAND_STATUS_TYPE_ERROR, CURAND_STATUS_TYPE_ERROR, HIPRAND_STATUS_TYPE_ERROR)
GPUMOD_VALUE(GPURAND_STATUS_OUT_OF_RANGE, CURAND_STATUS_OUT_OF_RANGE, HIPRAND_STATUS_OUT_OF_RANGE)
GPUMOD_VALUE(GPURAND_STATUS_LENGTH_NOT_MULTIPLE, CURAND_STATUS_LENGTH_NOT_MULTIPLE,
             HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
GPUMOD_VALUE(GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED, CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
             HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
GPUMOD_VALUE(GPURAND_STATUS_LAUNCH_FAILURE, CURAND_STATUS_LAUNCH_FAILURE,
             HIPRAND_STATUS_LAUNCH_FAILURE)
GPUMOD_VALUE(GPURAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_PREEXISTING_FAILURE,
             HIPRAND_STATUS_PREEXISTING_FAILURE)
GPUMOD_VALUE(GPURAND_STATUS_INITIALIZATION_FAILED, CURAND_STATUS_INITIALIZATION_FAILED,
             HIPRAND_STATUS_INITIALIZATION_FAILED)
GPUMOD_VALUE(GPURAND_STATUS_ARCH_MISMATCH, CURAND_STATUS_ARCH_MISMATCH,
             HIPRAND_STATUS_ARCH_MISMATCH)
GPUMOD_VALUE(GPURAND_STATUS_INTERNAL_ERROR, CURAND_STATUS_INTERNAL_ERROR,
             HIPRAND_STATUS_INTERNAL_ERROR)

// ========================================================================
// Constants - RNG type (values differ between backends, see file header)
// ========================================================================

GPUMOD_VALUE(GPURAND_RNG_TEST, CURAND_RNG_TEST, HIPRAND_RNG_TEST)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_XORWOW, HIPRAND_RNG_PSEUDO_XORWOW)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MRG32K3A, HIPRAND_RNG_PSEUDO_MRG32K3A)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MTGP32, HIPRAND_RNG_PSEUDO_MTGP32)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_MT19937, HIPRAND_RNG_PSEUDO_MT19937)
GPUMOD_VALUE(GPURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_PHILOX4_32_10,
             HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
GPUMOD_VALUE(GPURAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_DEFAULT, HIPRAND_RNG_QUASI_DEFAULT)
GPUMOD_VALUE(GPURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SOBOL32, HIPRAND_RNG_QUASI_SOBOL32)
GPUMOD_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
             HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32)
GPUMOD_VALUE(GPURAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SOBOL64)
GPUMOD_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
             HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64)

// ========================================================================
// Constants - ordering
// ========================================================================

GPUMOD_VALUE(GPURAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_BEST,
             HIPRAND_ORDERING_PSEUDO_BEST)
GPUMOD_VALUE(GPURAND_ORDERING_PSEUDO_DEFAULT, CURAND_ORDERING_PSEUDO_DEFAULT,
             HIPRAND_ORDERING_PSEUDO_DEFAULT)
GPUMOD_VALUE(GPURAND_ORDERING_PSEUDO_SEEDED, CURAND_ORDERING_PSEUDO_SEEDED,
             HIPRAND_ORDERING_PSEUDO_SEEDED)
GPUMOD_VALUE(GPURAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_LEGACY,
             HIPRAND_ORDERING_PSEUDO_LEGACY)
GPUMOD_VALUE(GPURAND_ORDERING_PSEUDO_DYNAMIC, CURAND_ORDERING_PSEUDO_DYNAMIC,
             HIPRAND_ORDERING_PSEUDO_DYNAMIC)
GPUMOD_VALUE(GPURAND_ORDERING_QUASI_DEFAULT, CURAND_ORDERING_QUASI_DEFAULT,
             HIPRAND_ORDERING_QUASI_DEFAULT)

// ========================================================================
// Constants - direction vector set
// ========================================================================

GPUMOD_VALUE(GPURAND_DIRECTION_VECTORS_32_JOEKUO6, CURAND_DIRECTION_VECTORS_32_JOEKUO6,
             HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)
GPUMOD_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
             CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
             HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
GPUMOD_VALUE(GPURAND_DIRECTION_VECTORS_64_JOEKUO6, CURAND_DIRECTION_VECTORS_64_JOEKUO6,
             HIPRAND_DIRECTION_VECTORS_64_JOEKUO6)
GPUMOD_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
             CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
             HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

// ========================================================================
// Generator management and configuration
// ========================================================================

GPUMOD_FUNCTION(gpurandCreateGenerator, curandCreateGenerator, hiprandCreateGenerator)
GPUMOD_FUNCTION(gpurandCreateGeneratorHost, curandCreateGeneratorHost, hiprandCreateGeneratorHost)
GPUMOD_FUNCTION(gpurandDestroyGenerator, curandDestroyGenerator, hiprandDestroyGenerator)
GPUMOD_FUNCTION(gpurandGetVersion, curandGetVersion, hiprandGetVersion)
GPUMOD_FUNCTION(gpurandSetStream, curandSetStream, hiprandSetStream)
GPUMOD_FUNCTION(gpurandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed,
                hiprandSetPseudoRandomGeneratorSeed)
GPUMOD_FUNCTION(gpurandSetGeneratorOffset, curandSetGeneratorOffset, hiprandSetGeneratorOffset)
GPUMOD_FUNCTION(gpurandSetGeneratorOrdering, curandSetGeneratorOrdering,
                hiprandSetGeneratorOrdering)
GPUMOD_FUNCTION(gpurandSetQuasiRandomGeneratorDimensions, curandSetQuasiRandomGeneratorDimensions,
                hiprandSetQuasiRandomGeneratorDimensions)
GPUMOD_FUNCTION(gpurandGenerateSeeds, curandGenerateSeeds, hiprandGenerateSeeds)

// ========================================================================
// Generation
// ========================================================================

GPUMOD_FUNCTION(gpurandGenerate, curandGenerate, hiprandGenerate)
GPUMOD_FUNCTION(gpurandGenerateLongLong, curandGenerateLongLong, hiprandGenerateLongLong)
GPUMOD_FUNCTION(gpurandGenerateUniform, curandGenerateUniform, hiprandGenerateUniform)
GPUMOD_FUNCTION(gpurandGenerateUniformDouble, curandGenerateUniformDouble,
                hiprandGenerateUniformDouble)
GPUMOD_FUNCTION(gpurandGenerateNormal, curandGenerateNormal, hiprandGenerateNormal)
GPUMOD_FUNCTION(gpurandGenerateNormalDouble, curandGenerateNormalDouble,
                hiprandGenerateNormalDouble)
GPUMOD_FUNCTION(gpurandGenerateLogNormal, curandGenerateLogNormal, hiprandGenerateLogNormal)
GPUMOD_FUNCTION(gpurandGenerateLogNormalDouble, curandGenerateLogNormalDouble,
                hiprandGenerateLogNormalDouble)
GPUMOD_FUNCTION(gpurandGeneratePoisson, curandGeneratePoisson, hiprandGeneratePoisson)

// ========================================================================
// Discrete distributions
// ========================================================================

GPUMOD_FUNCTION(gpurandCreatePoissonDistribution, curandCreatePoissonDistribution,
                hiprandCreatePoissonDistribution)
GPUMOD_FUNCTION(gpurandDestroyDistribution, curandDestroyDistribution, hiprandDestroyDistribution)

// ========================================================================
// Quasirandom direction vectors and scramble constants
// ========================================================================

GPUMOD_FUNCTION(gpurandGetDirectionVectors32, curandGetDirectionVectors32,
                hiprandGetDirectionVectors32)
GPUMOD_FUNCTION(gpurandGetDirectionVectors64, curandGetDirectionVectors64,
                hiprandGetDirectionVectors64)

// The scramble constants are a read-only table owned by the library.
// hipRAND hands them out as const (const unsigned int**); cuRAND does not
// (unsigned int**). These keep hipRAND's const-correct signature on both
// backends. On CUDA the const_cast only changes how the out-pointer is
// written; the table itself is never written through it.
#if defined(GPUMOD_GPU_BACKEND_CUDA)
inline gpurandStatus_t gpurandGetScrambleConstants32(const unsigned int **constants) {
  return ::gpumod::cuda::curandGetScrambleConstants32(const_cast<unsigned int **>(constants));
}
inline gpurandStatus_t gpurandGetScrambleConstants64(const unsigned long long **constants) {
  return ::gpumod::cuda::curandGetScrambleConstants64(const_cast<unsigned long long **>(constants));
}
#else
GPUMOD_FUNCTION(gpurandGetScrambleConstants32, curandGetScrambleConstants32,
                hiprandGetScrambleConstants32)
GPUMOD_FUNCTION(gpurandGetScrambleConstants64, curandGetScrambleConstants64,
                hiprandGetScrambleConstants64)
#endif

// ========================================================================
// Device generator state types (curand_kernel.h / hiprand_kernel.h)
//
// Types only -- the device functions that consume them live in rand.cuh,
// for the reason in this file's header. What the host needs these for is
// sizing and allocating the per-thread state array:
//
//   DeviceBuffer<gpurandState> states(count, device);  // device: shared_ptr<DeviceHandle>
//
// gpurandState is the default pseudorandom state on both backends, and on
// both backends that default is xorwow-backed. It is NOT, however, the same
// type as gpurandStateXORWOW on both:
//
//   CUDA:  curandState IS curandStateXORWOW      (one type, two names)
//   HIP:   hiprandState and hiprandStateXORWOW are two DISTINCT structs,
//          separately emitted by hiprand_kernel_rocm.h's
//          DEFINE_HIPRAND_STATE macro over the same rocrand_state_xorwow
//          base -- same layout and size, different C++ types
//
// Verified on ROCm 7.2.4; test/hip/hiprand_kernel.cppm pins it. So a
// gpurandState* and a gpurandStateXORWOW* are interchangeable on a CUDA
// build and a compile error on a HIP one. Pick one spelling and keep it: an
// API that takes gpurandState must be fed gpurandState.
//
// The sizes differ between backends too (different generator internals), so
// a state array is never a host-side fixed byte count -- always sizeof the
// type.
// ========================================================================

// Pseudorandom generators
GPUMOD_TYPE(gpurandStateXORWOW, curandStateXORWOW, hiprandStateXORWOW)
GPUMOD_TYPE(gpurandStateXORWOW_t, curandStateXORWOW_t, hiprandStateXORWOW_t)
GPUMOD_TYPE(gpurandStateMRG32k3a, curandStateMRG32k3a, hiprandStateMRG32k3a)
GPUMOD_TYPE(gpurandStateMRG32k3a_t, curandStateMRG32k3a_t, hiprandStateMRG32k3a_t)
GPUMOD_TYPE(gpurandStateMtgp32, curandStateMtgp32, hiprandStateMtgp32)
GPUMOD_TYPE(gpurandStateMtgp32_t, curandStateMtgp32_t, hiprandStateMtgp32_t)
GPUMOD_TYPE(gpurandStatePhilox4_32_10, curandStatePhilox4_32_10, hiprandStatePhilox4_32_10)
GPUMOD_TYPE(gpurandStatePhilox4_32_10_t, curandStatePhilox4_32_10_t, hiprandStatePhilox4_32_10_t)

// Quasirandom generators
GPUMOD_TYPE(gpurandStateSobol32, curandStateSobol32, hiprandStateSobol32)
GPUMOD_TYPE(gpurandStateSobol32_t, curandStateSobol32_t, hiprandStateSobol32_t)
GPUMOD_TYPE(gpurandStateScrambledSobol32, curandStateScrambledSobol32, hiprandStateScrambledSobol32)
GPUMOD_TYPE(gpurandStateScrambledSobol32_t, curandStateScrambledSobol32_t,
            hiprandStateScrambledSobol32_t)
GPUMOD_TYPE(gpurandStateSobol64, curandStateSobol64, hiprandStateSobol64)
GPUMOD_TYPE(gpurandStateSobol64_t, curandStateSobol64_t, hiprandStateSobol64_t)
GPUMOD_TYPE(gpurandStateScrambledSobol64, curandStateScrambledSobol64, hiprandStateScrambledSobol64)
GPUMOD_TYPE(gpurandStateScrambledSobol64_t, curandStateScrambledSobol64_t,
            hiprandStateScrambledSobol64_t)

// Default state -- see the section comment above
GPUMOD_TYPE(gpurandState, curandState, hiprandState)
GPUMOD_TYPE(gpurandState_t, curandState_t, hiprandState_t)

} // namespace gpumod
