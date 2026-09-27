// rand.cppm - Compile-time tests for wwr.rand
//
// Every exported name is checked against the backend's own entity (see
// gpu_check_macros.h) -- all of them, unlike test/gpu/blas.cppm and
// solver.cppm, which sample. src/extension/init_state and
// src/extension/random_normal use the device state types and nothing else from
// here, so for the whole host API this file is still
// the only check: there is no instantiation or runtime test downstream to
// catch a wrong-but-existing backend name.
//
// The device FUNCTIONS are not checked here and cannot be -- they live in
// src/rand.cuh, which only a device-compile pass can include. They are
// exercised instead by test/wrappers/rand, which runs them on-device.
//
// wwrrandGetScrambleConstants32/64 are forwarding functions on CUDA (cuRAND
// hands the table out non-const); what matters is the signature, checked
// below for both backends.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.rand;

import std;
import wwr.rand;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.curand;
#else
import wwr.hip.hiprand;
import wwr.hip.hiprand_kernel;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrrandGenerator_t, curandGenerator_t)
WWR_SAME_TYPE(wwrrandStatus_t, curandStatus_t)
WWR_SAME_TYPE(wwrrandRngType_t, curandRngType_t)
WWR_SAME_TYPE(wwrrandOrdering_t, curandOrdering_t)
WWR_SAME_TYPE(wwrrandDirectionVectorSet_t, curandDirectionVectorSet_t)
WWR_SAME_TYPE(wwrrandDirectionVectors32_t, curandDirectionVectors32_t)
WWR_SAME_TYPE(wwrrandDirectionVectors64_t, curandDirectionVectors64_t)
WWR_SAME_TYPE(wwrrandDiscreteDistribution_t, curandDiscreteDistribution_t)

WWR_SAME_VALUE(WWRRAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRRAND_STATUS_VERSION_MISMATCH, CURAND_STATUS_VERSION_MISMATCH)
WWR_SAME_VALUE(WWRRAND_STATUS_NOT_INITIALIZED, CURAND_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(WWRRAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_ALLOCATION_FAILED)
WWR_SAME_VALUE(WWRRAND_STATUS_TYPE_ERROR, CURAND_STATUS_TYPE_ERROR)
WWR_SAME_VALUE(WWRRAND_STATUS_OUT_OF_RANGE, CURAND_STATUS_OUT_OF_RANGE)
WWR_SAME_VALUE(WWRRAND_STATUS_LENGTH_NOT_MULTIPLE, CURAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_SAME_VALUE(WWRRAND_STATUS_DOUBLE_PRECISION_REQUIRED, CURAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_SAME_VALUE(WWRRAND_STATUS_LAUNCH_FAILURE, CURAND_STATUS_LAUNCH_FAILURE)
WWR_SAME_VALUE(WWRRAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_PREEXISTING_FAILURE)
WWR_SAME_VALUE(WWRRAND_STATUS_INITIALIZATION_FAILED, CURAND_STATUS_INITIALIZATION_FAILED)
WWR_SAME_VALUE(WWRRAND_STATUS_ARCH_MISMATCH, CURAND_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(WWRRAND_STATUS_INTERNAL_ERROR, CURAND_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRRAND_RNG_TEST, CURAND_RNG_TEST)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_XORWOW)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MRG32K3A)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MTGP32)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_MT19937)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_DEFAULT)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SOBOL32)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SOBOL64)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_BEST)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_DEFAULT, CURAND_ORDERING_PSEUDO_DEFAULT)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_SEEDED, CURAND_ORDERING_PSEUDO_SEEDED)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_LEGACY)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_DYNAMIC, CURAND_ORDERING_PSEUDO_DYNAMIC)
WWR_SAME_VALUE(WWRRAND_ORDERING_QUASI_DEFAULT, CURAND_ORDERING_QUASI_DEFAULT)
WWR_SAME_VALUE(WWRRAND_DIRECTION_VECTORS_32_JOEKUO6, CURAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
                  CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_DIRECTION_VECTORS_64_JOEKUO6, CURAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
                  CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

WWR_SAME_FUNCTION(wwrrandCreateGenerator, curandCreateGenerator)
WWR_SAME_FUNCTION(wwrrandCreateGeneratorHost, curandCreateGeneratorHost)
WWR_SAME_FUNCTION(wwrrandDestroyGenerator, curandDestroyGenerator)
WWR_SAME_FUNCTION(wwrrandGetVersion, curandGetVersion)
WWR_SAME_FUNCTION(wwrrandSetStream, curandSetStream)
WWR_SAME_FUNCTION(wwrrandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed)
WWR_SAME_FUNCTION(wwrrandSetGeneratorOffset, curandSetGeneratorOffset)
WWR_SAME_FUNCTION(wwrrandSetGeneratorOrdering, curandSetGeneratorOrdering)
WWR_SAME_FUNCTION(wwrrandSetQuasiRandomGeneratorDimensions,
                     curandSetQuasiRandomGeneratorDimensions)
WWR_SAME_FUNCTION(wwrrandGenerateSeeds, curandGenerateSeeds)
WWR_SAME_FUNCTION(wwrrandGenerate, curandGenerate)
WWR_SAME_FUNCTION(wwrrandGenerateLongLong, curandGenerateLongLong)
WWR_SAME_FUNCTION(wwrrandGenerateUniform, curandGenerateUniform)
WWR_SAME_FUNCTION(wwrrandGenerateUniformDouble, curandGenerateUniformDouble)
WWR_SAME_FUNCTION(wwrrandGenerateNormal, curandGenerateNormal)
WWR_SAME_FUNCTION(wwrrandGenerateNormalDouble, curandGenerateNormalDouble)
WWR_SAME_FUNCTION(wwrrandGenerateLogNormal, curandGenerateLogNormal)
WWR_SAME_FUNCTION(wwrrandGenerateLogNormalDouble, curandGenerateLogNormalDouble)
WWR_SAME_FUNCTION(wwrrandGeneratePoisson, curandGeneratePoisson)
WWR_SAME_FUNCTION(wwrrandCreatePoissonDistribution, curandCreatePoissonDistribution)
WWR_SAME_FUNCTION(wwrrandDestroyDistribution, curandDestroyDistribution)
WWR_SAME_FUNCTION(wwrrandGetDirectionVectors32, curandGetDirectionVectors32)
WWR_SAME_FUNCTION(wwrrandGetDirectionVectors64, curandGetDirectionVectors64)

// Device generator state types (curand_kernel.h)
WWR_SAME_TYPE(wwrrandStateXORWOW, curandStateXORWOW)
WWR_SAME_TYPE(wwrrandStateXORWOW_t, curandStateXORWOW_t)
WWR_SAME_TYPE(wwrrandStateMRG32k3a, curandStateMRG32k3a)
WWR_SAME_TYPE(wwrrandStateMRG32k3a_t, curandStateMRG32k3a_t)
WWR_SAME_TYPE(wwrrandStateMtgp32, curandStateMtgp32)
WWR_SAME_TYPE(wwrrandStateMtgp32_t, curandStateMtgp32_t)
WWR_SAME_TYPE(wwrrandStatePhilox4_32_10, curandStatePhilox4_32_10)
WWR_SAME_TYPE(wwrrandStatePhilox4_32_10_t, curandStatePhilox4_32_10_t)
WWR_SAME_TYPE(wwrrandStateSobol32, curandStateSobol32)
WWR_SAME_TYPE(wwrrandStateSobol32_t, curandStateSobol32_t)
WWR_SAME_TYPE(wwrrandStateScrambledSobol32, curandStateScrambledSobol32)
WWR_SAME_TYPE(wwrrandStateScrambledSobol32_t, curandStateScrambledSobol32_t)
WWR_SAME_TYPE(wwrrandStateSobol64, curandStateSobol64)
WWR_SAME_TYPE(wwrrandStateSobol64_t, curandStateSobol64_t)
WWR_SAME_TYPE(wwrrandStateScrambledSobol64, curandStateScrambledSobol64)
WWR_SAME_TYPE(wwrrandStateScrambledSobol64_t, curandStateScrambledSobol64_t)
WWR_SAME_TYPE(wwrrandState, curandState)
WWR_SAME_TYPE(wwrrandState_t, curandState_t)

// On CUDA the default state IS the XORWOW state -- one type, two names. This
// is exactly what does NOT hold on HIP (see the HIP branch below), so it is
// asserted per backend rather than shared.
WWR_SAME_TYPE(wwrrandState, wwrrandStateXORWOW)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrrandGenerator_t, hiprandGenerator_t)
WWR_SAME_TYPE(wwrrandStatus_t, hiprandStatus_t)
WWR_SAME_TYPE(wwrrandRngType_t, hiprandRngType_t)
WWR_SAME_TYPE(wwrrandOrdering_t, hiprandOrdering_t)
WWR_SAME_TYPE(wwrrandDirectionVectorSet_t, hiprandDirectionVectorSet_t)
WWR_SAME_TYPE(wwrrandDirectionVectors32_t, hiprandDirectionVectors32_t)
WWR_SAME_TYPE(wwrrandDirectionVectors64_t, hiprandDirectionVectors64_t)
WWR_SAME_TYPE(wwrrandDiscreteDistribution_t, hiprandDiscreteDistribution_t)

WWR_SAME_VALUE(WWRRAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRRAND_STATUS_VERSION_MISMATCH, HIPRAND_STATUS_VERSION_MISMATCH)
WWR_SAME_VALUE(WWRRAND_STATUS_NOT_INITIALIZED, HIPRAND_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(WWRRAND_STATUS_ALLOCATION_FAILED, HIPRAND_STATUS_ALLOCATION_FAILED)
WWR_SAME_VALUE(WWRRAND_STATUS_TYPE_ERROR, HIPRAND_STATUS_TYPE_ERROR)
WWR_SAME_VALUE(WWRRAND_STATUS_OUT_OF_RANGE, HIPRAND_STATUS_OUT_OF_RANGE)
WWR_SAME_VALUE(WWRRAND_STATUS_LENGTH_NOT_MULTIPLE, HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_SAME_VALUE(WWRRAND_STATUS_DOUBLE_PRECISION_REQUIRED,
                  HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_SAME_VALUE(WWRRAND_STATUS_LAUNCH_FAILURE, HIPRAND_STATUS_LAUNCH_FAILURE)
WWR_SAME_VALUE(WWRRAND_STATUS_PREEXISTING_FAILURE, HIPRAND_STATUS_PREEXISTING_FAILURE)
WWR_SAME_VALUE(WWRRAND_STATUS_INITIALIZATION_FAILED, HIPRAND_STATUS_INITIALIZATION_FAILED)
WWR_SAME_VALUE(WWRRAND_STATUS_ARCH_MISMATCH, HIPRAND_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(WWRRAND_STATUS_INTERNAL_ERROR, HIPRAND_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRRAND_RNG_TEST, HIPRAND_RNG_TEST)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_XORWOW, HIPRAND_RNG_PSEUDO_XORWOW)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MRG32K3A, HIPRAND_RNG_PSEUDO_MRG32K3A)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MTGP32, HIPRAND_RNG_PSEUDO_MTGP32)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_MT19937, HIPRAND_RNG_PSEUDO_MT19937)
WWR_SAME_VALUE(WWRRAND_RNG_PSEUDO_PHILOX4_32_10, HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_DEFAULT, HIPRAND_RNG_QUASI_DEFAULT)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SOBOL32, HIPRAND_RNG_QUASI_SOBOL32)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL32, HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SOBOL64)
WWR_SAME_VALUE(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL64, HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_BEST, HIPRAND_ORDERING_PSEUDO_BEST)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_DEFAULT, HIPRAND_ORDERING_PSEUDO_DEFAULT)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_SEEDED, HIPRAND_ORDERING_PSEUDO_SEEDED)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_LEGACY, HIPRAND_ORDERING_PSEUDO_LEGACY)
WWR_SAME_VALUE(WWRRAND_ORDERING_PSEUDO_DYNAMIC, HIPRAND_ORDERING_PSEUDO_DYNAMIC)
WWR_SAME_VALUE(WWRRAND_ORDERING_QUASI_DEFAULT, HIPRAND_ORDERING_QUASI_DEFAULT)
WWR_SAME_VALUE(WWRRAND_DIRECTION_VECTORS_32_JOEKUO6, HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
                  HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_DIRECTION_VECTORS_64_JOEKUO6, HIPRAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_SAME_VALUE(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
                  HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

WWR_SAME_FUNCTION(wwrrandCreateGenerator, hiprandCreateGenerator)
WWR_SAME_FUNCTION(wwrrandCreateGeneratorHost, hiprandCreateGeneratorHost)
WWR_SAME_FUNCTION(wwrrandDestroyGenerator, hiprandDestroyGenerator)
WWR_SAME_FUNCTION(wwrrandGetVersion, hiprandGetVersion)
WWR_SAME_FUNCTION(wwrrandSetStream, hiprandSetStream)
WWR_SAME_FUNCTION(wwrrandSetPseudoRandomGeneratorSeed, hiprandSetPseudoRandomGeneratorSeed)
WWR_SAME_FUNCTION(wwrrandSetGeneratorOffset, hiprandSetGeneratorOffset)
WWR_SAME_FUNCTION(wwrrandSetGeneratorOrdering, hiprandSetGeneratorOrdering)
WWR_SAME_FUNCTION(wwrrandSetQuasiRandomGeneratorDimensions,
                     hiprandSetQuasiRandomGeneratorDimensions)
WWR_SAME_FUNCTION(wwrrandGenerateSeeds, hiprandGenerateSeeds)
WWR_SAME_FUNCTION(wwrrandGenerate, hiprandGenerate)
WWR_SAME_FUNCTION(wwrrandGenerateLongLong, hiprandGenerateLongLong)
WWR_SAME_FUNCTION(wwrrandGenerateUniform, hiprandGenerateUniform)
WWR_SAME_FUNCTION(wwrrandGenerateUniformDouble, hiprandGenerateUniformDouble)
WWR_SAME_FUNCTION(wwrrandGenerateNormal, hiprandGenerateNormal)
WWR_SAME_FUNCTION(wwrrandGenerateNormalDouble, hiprandGenerateNormalDouble)
WWR_SAME_FUNCTION(wwrrandGenerateLogNormal, hiprandGenerateLogNormal)
WWR_SAME_FUNCTION(wwrrandGenerateLogNormalDouble, hiprandGenerateLogNormalDouble)
WWR_SAME_FUNCTION(wwrrandGeneratePoisson, hiprandGeneratePoisson)
WWR_SAME_FUNCTION(wwrrandCreatePoissonDistribution, hiprandCreatePoissonDistribution)
WWR_SAME_FUNCTION(wwrrandDestroyDistribution, hiprandDestroyDistribution)
WWR_SAME_FUNCTION(wwrrandGetDirectionVectors32, hiprandGetDirectionVectors32)
WWR_SAME_FUNCTION(wwrrandGetDirectionVectors64, hiprandGetDirectionVectors64)
WWR_SAME_FUNCTION(wwrrandGetScrambleConstants32, hiprandGetScrambleConstants32)
WWR_SAME_FUNCTION(wwrrandGetScrambleConstants64, hiprandGetScrambleConstants64)

// Device generator state types (hiprand_kernel.h, via
// wwr.hip.hiprand_kernel -- a separate module here, unlike cuRAND)
WWR_SAME_TYPE(wwrrandStateXORWOW, hiprandStateXORWOW)
WWR_SAME_TYPE(wwrrandStateXORWOW_t, hiprandStateXORWOW_t)
WWR_SAME_TYPE(wwrrandStateMRG32k3a, hiprandStateMRG32k3a)
WWR_SAME_TYPE(wwrrandStateMRG32k3a_t, hiprandStateMRG32k3a_t)
WWR_SAME_TYPE(wwrrandStateMtgp32, hiprandStateMtgp32)
WWR_SAME_TYPE(wwrrandStateMtgp32_t, hiprandStateMtgp32_t)
WWR_SAME_TYPE(wwrrandStatePhilox4_32_10, hiprandStatePhilox4_32_10)
WWR_SAME_TYPE(wwrrandStatePhilox4_32_10_t, hiprandStatePhilox4_32_10_t)
WWR_SAME_TYPE(wwrrandStateSobol32, hiprandStateSobol32)
WWR_SAME_TYPE(wwrrandStateSobol32_t, hiprandStateSobol32_t)
WWR_SAME_TYPE(wwrrandStateScrambledSobol32, hiprandStateScrambledSobol32)
WWR_SAME_TYPE(wwrrandStateScrambledSobol32_t, hiprandStateScrambledSobol32_t)
WWR_SAME_TYPE(wwrrandStateSobol64, hiprandStateSobol64)
WWR_SAME_TYPE(wwrrandStateSobol64_t, hiprandStateSobol64_t)
WWR_SAME_TYPE(wwrrandStateScrambledSobol64, hiprandStateScrambledSobol64)
WWR_SAME_TYPE(wwrrandStateScrambledSobol64_t, hiprandStateScrambledSobol64_t)
WWR_SAME_TYPE(wwrrandState, hiprandState)
WWR_SAME_TYPE(wwrrandState_t, hiprandState_t)

// On HIP the default state is NOT the XORWOW state: hiprand_kernel_rocm.h's
// DEFINE_HIPRAND_STATE macro emits a fresh struct for each, both deriving
// from rocrand_state_xorwow -- same layout, distinct C++ types. The CUDA
// branch above asserts the opposite, deliberately: this is the one place
// gpu.rand's two backends differ in type IDENTITY rather than spelling, and
// src/rand.cppm's device-state section is written around it.
static_assert(!std::is_same_v<wwrrandState, wwrrandStateXORWOW>);
static_assert(sizeof(wwrrandState) == sizeof(wwrrandStateXORWOW));

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: getScrambleConstants has hipRAND's const-correct signature
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrrandGetScrambleConstants32)>,
                 wwrrandStatus_t(const unsigned int **))
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrrandGetScrambleConstants64)>,
                 wwrrandStatus_t(const unsigned long long **))

} // namespace wwr::test
