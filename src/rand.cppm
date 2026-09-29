/**
 * @file rand.cppm
 * @brief Backend-neutral random number generation: wwrrand* names for cuRAND / hipRAND
 *
 * The cuRAND / hipRAND host API, plus the device generator STATE TYPES and
 * nothing else of the device API: device functions are __device__-qualified
 * and live in rand.cuh, while the state types are plain data and the host is
 * what sizes the per-thread state array. wwrrand<X> stands for curand<X> /
 * hiprand<X>, WWRRAND_<X> for CURAND_<X> / HIPRAND_<X>, each written out in
 * full. See backend.h.
 *
 * Everything the two host APIs share is listed. Absent for want of a
 * counterpart -- cuRAND: curandGetProperty, curandGeneratePoissonMethod,
 * curandGenerateBinomial(Method), curandMethod_t, curandDistribution_t;
 * hipRAND: hiprandGenerateChar/Short, the *Half generators,
 * HIPRAND_STATUS_NOT_IMPLEMENTED. Neither has a status-to-string function.
 *
 * The compile-time CURAND_VERSION / HIPRAND_VERSION macro is not aliased: its
 * value is backend-specific, and the runtime wwrrandGetVersion query is the
 * portable equivalent.
 *
 * wwrrandState and wwrrandStateXORWOW are one type on CUDA and two on HIP, and
 * enumerator VALUES differ even where names agree. See docs/architecture.md,
 * sections 1 and 6.
 *
 * Usage:
 *   import wwr.rand;
 *
 *   wwrrandGenerator_t gen;
 *   wwrrandCreateGenerator(&gen, WWRRAND_RNG_PSEUDO_DEFAULT);
 */

module;

#include "backend.h"

export module wwr.rand;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.curand;
#else
import wwr.hip.hiprand;
import wwr.hip.hiprand_kernel;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrrandGenerator_t, curandGenerator_t, hiprandGenerator_t)
WWR_TYPE(wwrrandStatus_t, curandStatus_t, hiprandStatus_t)
WWR_TYPE(wwrrandRngType_t, curandRngType_t, hiprandRngType_t)
WWR_TYPE(wwrrandOrdering_t, curandOrdering_t, hiprandOrdering_t)
WWR_TYPE(wwrrandDirectionVectorSet_t, curandDirectionVectorSet_t, hiprandDirectionVectorSet_t)
WWR_TYPE(wwrrandDirectionVectors32_t, curandDirectionVectors32_t, hiprandDirectionVectors32_t)
WWR_TYPE(wwrrandDirectionVectors64_t, curandDirectionVectors64_t, hiprandDirectionVectors64_t)
WWR_TYPE(wwrrandDiscreteDistribution_t, curandDiscreteDistribution_t,
            hiprandDiscreteDistribution_t)

// ========================================================================
// Constants - status
// ========================================================================

WWR_VALUE(WWRRAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
WWR_VALUE(WWRRAND_STATUS_VERSION_MISMATCH, CURAND_STATUS_VERSION_MISMATCH,
             HIPRAND_STATUS_VERSION_MISMATCH)
WWR_VALUE(WWRRAND_STATUS_NOT_INITIALIZED, CURAND_STATUS_NOT_INITIALIZED,
             HIPRAND_STATUS_NOT_INITIALIZED)
WWR_VALUE(WWRRAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_ALLOCATION_FAILED,
             HIPRAND_STATUS_ALLOCATION_FAILED)
WWR_VALUE(WWRRAND_STATUS_TYPE_ERROR, CURAND_STATUS_TYPE_ERROR, HIPRAND_STATUS_TYPE_ERROR)
WWR_VALUE(WWRRAND_STATUS_OUT_OF_RANGE, CURAND_STATUS_OUT_OF_RANGE, HIPRAND_STATUS_OUT_OF_RANGE)
WWR_VALUE(WWRRAND_STATUS_LENGTH_NOT_MULTIPLE, CURAND_STATUS_LENGTH_NOT_MULTIPLE,
             HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_VALUE(WWRRAND_STATUS_DOUBLE_PRECISION_REQUIRED, CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
             HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_VALUE(WWRRAND_STATUS_LAUNCH_FAILURE, CURAND_STATUS_LAUNCH_FAILURE,
             HIPRAND_STATUS_LAUNCH_FAILURE)
WWR_VALUE(WWRRAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_PREEXISTING_FAILURE,
             HIPRAND_STATUS_PREEXISTING_FAILURE)
WWR_VALUE(WWRRAND_STATUS_INITIALIZATION_FAILED, CURAND_STATUS_INITIALIZATION_FAILED,
             HIPRAND_STATUS_INITIALIZATION_FAILED)
WWR_VALUE(WWRRAND_STATUS_ARCH_MISMATCH, CURAND_STATUS_ARCH_MISMATCH,
             HIPRAND_STATUS_ARCH_MISMATCH)
WWR_VALUE(WWRRAND_STATUS_INTERNAL_ERROR, CURAND_STATUS_INTERNAL_ERROR,
             HIPRAND_STATUS_INTERNAL_ERROR)

// ========================================================================
// Constants - RNG type (values differ between backends, see file header)
// ========================================================================

WWR_VALUE(WWRRAND_RNG_TEST, CURAND_RNG_TEST, HIPRAND_RNG_TEST)
WWR_VALUE(WWRRAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)
WWR_VALUE(WWRRAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_XORWOW, HIPRAND_RNG_PSEUDO_XORWOW)
WWR_VALUE(WWRRAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MRG32K3A, HIPRAND_RNG_PSEUDO_MRG32K3A)
WWR_VALUE(WWRRAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MTGP32, HIPRAND_RNG_PSEUDO_MTGP32)
WWR_VALUE(WWRRAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_MT19937, HIPRAND_RNG_PSEUDO_MT19937)
WWR_VALUE(WWRRAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_PHILOX4_32_10,
             HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_VALUE(WWRRAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_DEFAULT, HIPRAND_RNG_QUASI_DEFAULT)
WWR_VALUE(WWRRAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SOBOL32, HIPRAND_RNG_QUASI_SOBOL32)
WWR_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
             HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_VALUE(WWRRAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SOBOL64)
WWR_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
             HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64)

// ========================================================================
// Constants - ordering
// ========================================================================

WWR_VALUE(WWRRAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_BEST,
             HIPRAND_ORDERING_PSEUDO_BEST)
WWR_VALUE(WWRRAND_ORDERING_PSEUDO_DEFAULT, CURAND_ORDERING_PSEUDO_DEFAULT,
             HIPRAND_ORDERING_PSEUDO_DEFAULT)
WWR_VALUE(WWRRAND_ORDERING_PSEUDO_SEEDED, CURAND_ORDERING_PSEUDO_SEEDED,
             HIPRAND_ORDERING_PSEUDO_SEEDED)
WWR_VALUE(WWRRAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_LEGACY,
             HIPRAND_ORDERING_PSEUDO_LEGACY)
WWR_VALUE(WWRRAND_ORDERING_PSEUDO_DYNAMIC, CURAND_ORDERING_PSEUDO_DYNAMIC,
             HIPRAND_ORDERING_PSEUDO_DYNAMIC)
WWR_VALUE(WWRRAND_ORDERING_QUASI_DEFAULT, CURAND_ORDERING_QUASI_DEFAULT,
             HIPRAND_ORDERING_QUASI_DEFAULT)

// ========================================================================
// Constants - direction vector set
// ========================================================================

WWR_VALUE(WWRRAND_DIRECTION_VECTORS_32_JOEKUO6, CURAND_DIRECTION_VECTORS_32_JOEKUO6,
             HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
             CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
             HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_VALUE(WWRRAND_DIRECTION_VECTORS_64_JOEKUO6, CURAND_DIRECTION_VECTORS_64_JOEKUO6,
             HIPRAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
             CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
             HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

// ========================================================================
// Generator management and configuration
// ========================================================================

WWR_FUNCTION(wwrrandCreateGenerator, curandCreateGenerator, hiprandCreateGenerator)
WWR_FUNCTION(wwrrandCreateGeneratorHost, curandCreateGeneratorHost, hiprandCreateGeneratorHost)
WWR_FUNCTION(wwrrandDestroyGenerator, curandDestroyGenerator, hiprandDestroyGenerator)
WWR_FUNCTION(wwrrandGetVersion, curandGetVersion, hiprandGetVersion)
WWR_FUNCTION(wwrrandSetStream, curandSetStream, hiprandSetStream)
WWR_FUNCTION(wwrrandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed,
                hiprandSetPseudoRandomGeneratorSeed)
WWR_FUNCTION(wwrrandSetGeneratorOffset, curandSetGeneratorOffset, hiprandSetGeneratorOffset)
WWR_FUNCTION(wwrrandSetGeneratorOrdering, curandSetGeneratorOrdering,
                hiprandSetGeneratorOrdering)
WWR_FUNCTION(wwrrandSetQuasiRandomGeneratorDimensions, curandSetQuasiRandomGeneratorDimensions,
                hiprandSetQuasiRandomGeneratorDimensions)
WWR_FUNCTION(wwrrandGenerateSeeds, curandGenerateSeeds, hiprandGenerateSeeds)

// ========================================================================
// Generation
// ========================================================================

WWR_FUNCTION(wwrrandGenerate, curandGenerate, hiprandGenerate)
WWR_FUNCTION(wwrrandGenerateLongLong, curandGenerateLongLong, hiprandGenerateLongLong)
WWR_FUNCTION(wwrrandGenerateUniform, curandGenerateUniform, hiprandGenerateUniform)
WWR_FUNCTION(wwrrandGenerateUniformDouble, curandGenerateUniformDouble,
                hiprandGenerateUniformDouble)
WWR_FUNCTION(wwrrandGenerateNormal, curandGenerateNormal, hiprandGenerateNormal)
WWR_FUNCTION(wwrrandGenerateNormalDouble, curandGenerateNormalDouble,
                hiprandGenerateNormalDouble)
WWR_FUNCTION(wwrrandGenerateLogNormal, curandGenerateLogNormal, hiprandGenerateLogNormal)
WWR_FUNCTION(wwrrandGenerateLogNormalDouble, curandGenerateLogNormalDouble,
                hiprandGenerateLogNormalDouble)
WWR_FUNCTION(wwrrandGeneratePoisson, curandGeneratePoisson, hiprandGeneratePoisson)

// ========================================================================
// Discrete distributions
// ========================================================================

WWR_FUNCTION(wwrrandCreatePoissonDistribution, curandCreatePoissonDistribution,
                hiprandCreatePoissonDistribution)
WWR_FUNCTION(wwrrandDestroyDistribution, curandDestroyDistribution, hiprandDestroyDistribution)

// ========================================================================
// Quasirandom direction vectors and scramble constants
// ========================================================================

WWR_FUNCTION(wwrrandGetDirectionVectors32, curandGetDirectionVectors32,
                hiprandGetDirectionVectors32)
WWR_FUNCTION(wwrrandGetDirectionVectors64, curandGetDirectionVectors64,
                hiprandGetDirectionVectors64)

// The scramble constants are a read-only table owned by the library.
// hipRAND hands them out as const (const unsigned int**); cuRAND does not
// (unsigned int**). These keep hipRAND's const-correct signature on both
// backends. On CUDA the const_cast only changes how the out-pointer is
// written; the table itself is never written through it.
#if defined(WWR_GPU_BACKEND_CUDA)
inline wwrrandStatus_t wwrrandGetScrambleConstants32(const unsigned int **constants) {
  return ::wwr::cuda::curandGetScrambleConstants32(const_cast<unsigned int **>(constants));
}
inline wwrrandStatus_t wwrrandGetScrambleConstants64(const unsigned long long **constants) {
  return ::wwr::cuda::curandGetScrambleConstants64(const_cast<unsigned long long **>(constants));
}
#else
WWR_FUNCTION(wwrrandGetScrambleConstants32, curandGetScrambleConstants32,
                hiprandGetScrambleConstants32)
WWR_FUNCTION(wwrrandGetScrambleConstants64, curandGetScrambleConstants64,
                hiprandGetScrambleConstants64)
#endif

// ========================================================================
// Device generator state types (curand_kernel.h / hiprand_kernel.h)
//
// Types only -- the device functions that consume them live in rand.cuh,
// for the reason in this file's header. What the host needs these for is
// sizing and allocating the per-thread state array:
//
//   DeviceBufferWrapper<wwrrandState, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>,
//                       MyDeviceHandle, AbortPolicy<wwrError_t>>
//       states(count, device);  // device: shared_ptr<MyDeviceHandle> (caller-supplied);
//                               // AbortPolicy is the caller's own -- the library ships none
//
// wwrrandState is the default pseudorandom state on both backends, and on
// both backends that default is xorwow-backed. It is NOT, however, the same
// type as wwrrandStateXORWOW on both:
//
//   CUDA:  curandState IS curandStateXORWOW      (one type, two names)
//   HIP:   hiprandState and hiprandStateXORWOW are two DISTINCT structs,
//          separately emitted by hiprand_kernel_rocm.h's
//          DEFINE_HIPRAND_STATE macro over the same rocrand_state_xorwow
//          base -- same layout and size, different C++ types
//
// Verified on ROCm 7.2.4; test/hip/hiprand_kernel.cppm pins it. So a
// wwrrandState* and a wwrrandStateXORWOW* are interchangeable on a CUDA
// build and a compile error on a HIP one. Pick one spelling and keep it: an
// API that takes wwrrandState must be fed wwrrandState.
//
// The sizes differ between backends too (different generator internals), so
// a state array is never a host-side fixed byte count -- always sizeof the
// type.
// ========================================================================

// Pseudorandom generators
WWR_TYPE(wwrrandStateXORWOW, curandStateXORWOW, hiprandStateXORWOW)
WWR_TYPE(wwrrandStateXORWOW_t, curandStateXORWOW_t, hiprandStateXORWOW_t)
WWR_TYPE(wwrrandStateMRG32k3a, curandStateMRG32k3a, hiprandStateMRG32k3a)
WWR_TYPE(wwrrandStateMRG32k3a_t, curandStateMRG32k3a_t, hiprandStateMRG32k3a_t)
WWR_TYPE(wwrrandStateMtgp32, curandStateMtgp32, hiprandStateMtgp32)
WWR_TYPE(wwrrandStateMtgp32_t, curandStateMtgp32_t, hiprandStateMtgp32_t)
WWR_TYPE(wwrrandStatePhilox4_32_10, curandStatePhilox4_32_10, hiprandStatePhilox4_32_10)
WWR_TYPE(wwrrandStatePhilox4_32_10_t, curandStatePhilox4_32_10_t, hiprandStatePhilox4_32_10_t)

// Quasirandom generators
WWR_TYPE(wwrrandStateSobol32, curandStateSobol32, hiprandStateSobol32)
WWR_TYPE(wwrrandStateSobol32_t, curandStateSobol32_t, hiprandStateSobol32_t)
WWR_TYPE(wwrrandStateScrambledSobol32, curandStateScrambledSobol32, hiprandStateScrambledSobol32)
WWR_TYPE(wwrrandStateScrambledSobol32_t, curandStateScrambledSobol32_t,
            hiprandStateScrambledSobol32_t)
WWR_TYPE(wwrrandStateSobol64, curandStateSobol64, hiprandStateSobol64)
WWR_TYPE(wwrrandStateSobol64_t, curandStateSobol64_t, hiprandStateSobol64_t)
WWR_TYPE(wwrrandStateScrambledSobol64, curandStateScrambledSobol64, hiprandStateScrambledSobol64)
WWR_TYPE(wwrrandStateScrambledSobol64_t, curandStateScrambledSobol64_t,
            hiprandStateScrambledSobol64_t)

// Default state -- see the section comment above
WWR_TYPE(wwrrandState, curandState, hiprandState)
WWR_TYPE(wwrrandState_t, curandState_t, hiprandState_t)

} // namespace wwr
