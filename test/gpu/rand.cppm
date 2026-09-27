// rand.cppm - Compile-time tests for gpumod.rand
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
// gpurandGetScrambleConstants32/64 are forwarding functions on CUDA (cuRAND
// hands the table out non-const); what matters is the signature, checked
// below for both backends.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.rand;

import std;
import gpumod.rand;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.curand;
#else
import gpumod.hip.hiprand;
import gpumod.hip.hiprand_kernel;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(gpurandGenerator_t, curandGenerator_t)
WWR_SAME_TYPE(gpurandStatus_t, curandStatus_t)
WWR_SAME_TYPE(gpurandRngType_t, curandRngType_t)
WWR_SAME_TYPE(gpurandOrdering_t, curandOrdering_t)
WWR_SAME_TYPE(gpurandDirectionVectorSet_t, curandDirectionVectorSet_t)
WWR_SAME_TYPE(gpurandDirectionVectors32_t, curandDirectionVectors32_t)
WWR_SAME_TYPE(gpurandDirectionVectors64_t, curandDirectionVectors64_t)
WWR_SAME_TYPE(gpurandDiscreteDistribution_t, curandDiscreteDistribution_t)

WWR_SAME_VALUE(GPURAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS)
WWR_SAME_VALUE(GPURAND_STATUS_VERSION_MISMATCH, CURAND_STATUS_VERSION_MISMATCH)
WWR_SAME_VALUE(GPURAND_STATUS_NOT_INITIALIZED, CURAND_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(GPURAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_ALLOCATION_FAILED)
WWR_SAME_VALUE(GPURAND_STATUS_TYPE_ERROR, CURAND_STATUS_TYPE_ERROR)
WWR_SAME_VALUE(GPURAND_STATUS_OUT_OF_RANGE, CURAND_STATUS_OUT_OF_RANGE)
WWR_SAME_VALUE(GPURAND_STATUS_LENGTH_NOT_MULTIPLE, CURAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_SAME_VALUE(GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED, CURAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_SAME_VALUE(GPURAND_STATUS_LAUNCH_FAILURE, CURAND_STATUS_LAUNCH_FAILURE)
WWR_SAME_VALUE(GPURAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_PREEXISTING_FAILURE)
WWR_SAME_VALUE(GPURAND_STATUS_INITIALIZATION_FAILED, CURAND_STATUS_INITIALIZATION_FAILED)
WWR_SAME_VALUE(GPURAND_STATUS_ARCH_MISMATCH, CURAND_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(GPURAND_STATUS_INTERNAL_ERROR, CURAND_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(GPURAND_RNG_TEST, CURAND_RNG_TEST)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_XORWOW)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MRG32K3A)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MTGP32)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_MT19937)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_DEFAULT)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SOBOL32)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SOBOL64)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_BEST)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_DEFAULT, CURAND_ORDERING_PSEUDO_DEFAULT)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_SEEDED, CURAND_ORDERING_PSEUDO_SEEDED)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_LEGACY)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_DYNAMIC, CURAND_ORDERING_PSEUDO_DYNAMIC)
WWR_SAME_VALUE(GPURAND_ORDERING_QUASI_DEFAULT, CURAND_ORDERING_QUASI_DEFAULT)
WWR_SAME_VALUE(GPURAND_DIRECTION_VECTORS_32_JOEKUO6, CURAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
                  CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(GPURAND_DIRECTION_VECTORS_64_JOEKUO6, CURAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_SAME_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
                  CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

WWR_SAME_FUNCTION(gpurandCreateGenerator, curandCreateGenerator)
WWR_SAME_FUNCTION(gpurandCreateGeneratorHost, curandCreateGeneratorHost)
WWR_SAME_FUNCTION(gpurandDestroyGenerator, curandDestroyGenerator)
WWR_SAME_FUNCTION(gpurandGetVersion, curandGetVersion)
WWR_SAME_FUNCTION(gpurandSetStream, curandSetStream)
WWR_SAME_FUNCTION(gpurandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed)
WWR_SAME_FUNCTION(gpurandSetGeneratorOffset, curandSetGeneratorOffset)
WWR_SAME_FUNCTION(gpurandSetGeneratorOrdering, curandSetGeneratorOrdering)
WWR_SAME_FUNCTION(gpurandSetQuasiRandomGeneratorDimensions,
                     curandSetQuasiRandomGeneratorDimensions)
WWR_SAME_FUNCTION(gpurandGenerateSeeds, curandGenerateSeeds)
WWR_SAME_FUNCTION(gpurandGenerate, curandGenerate)
WWR_SAME_FUNCTION(gpurandGenerateLongLong, curandGenerateLongLong)
WWR_SAME_FUNCTION(gpurandGenerateUniform, curandGenerateUniform)
WWR_SAME_FUNCTION(gpurandGenerateUniformDouble, curandGenerateUniformDouble)
WWR_SAME_FUNCTION(gpurandGenerateNormal, curandGenerateNormal)
WWR_SAME_FUNCTION(gpurandGenerateNormalDouble, curandGenerateNormalDouble)
WWR_SAME_FUNCTION(gpurandGenerateLogNormal, curandGenerateLogNormal)
WWR_SAME_FUNCTION(gpurandGenerateLogNormalDouble, curandGenerateLogNormalDouble)
WWR_SAME_FUNCTION(gpurandGeneratePoisson, curandGeneratePoisson)
WWR_SAME_FUNCTION(gpurandCreatePoissonDistribution, curandCreatePoissonDistribution)
WWR_SAME_FUNCTION(gpurandDestroyDistribution, curandDestroyDistribution)
WWR_SAME_FUNCTION(gpurandGetDirectionVectors32, curandGetDirectionVectors32)
WWR_SAME_FUNCTION(gpurandGetDirectionVectors64, curandGetDirectionVectors64)

// Device generator state types (curand_kernel.h)
WWR_SAME_TYPE(gpurandStateXORWOW, curandStateXORWOW)
WWR_SAME_TYPE(gpurandStateXORWOW_t, curandStateXORWOW_t)
WWR_SAME_TYPE(gpurandStateMRG32k3a, curandStateMRG32k3a)
WWR_SAME_TYPE(gpurandStateMRG32k3a_t, curandStateMRG32k3a_t)
WWR_SAME_TYPE(gpurandStateMtgp32, curandStateMtgp32)
WWR_SAME_TYPE(gpurandStateMtgp32_t, curandStateMtgp32_t)
WWR_SAME_TYPE(gpurandStatePhilox4_32_10, curandStatePhilox4_32_10)
WWR_SAME_TYPE(gpurandStatePhilox4_32_10_t, curandStatePhilox4_32_10_t)
WWR_SAME_TYPE(gpurandStateSobol32, curandStateSobol32)
WWR_SAME_TYPE(gpurandStateSobol32_t, curandStateSobol32_t)
WWR_SAME_TYPE(gpurandStateScrambledSobol32, curandStateScrambledSobol32)
WWR_SAME_TYPE(gpurandStateScrambledSobol32_t, curandStateScrambledSobol32_t)
WWR_SAME_TYPE(gpurandStateSobol64, curandStateSobol64)
WWR_SAME_TYPE(gpurandStateSobol64_t, curandStateSobol64_t)
WWR_SAME_TYPE(gpurandStateScrambledSobol64, curandStateScrambledSobol64)
WWR_SAME_TYPE(gpurandStateScrambledSobol64_t, curandStateScrambledSobol64_t)
WWR_SAME_TYPE(gpurandState, curandState)
WWR_SAME_TYPE(gpurandState_t, curandState_t)

// On CUDA the default state IS the XORWOW state -- one type, two names. This
// is exactly what does NOT hold on HIP (see the HIP branch below), so it is
// asserted per backend rather than shared.
WWR_SAME_TYPE(gpurandState, gpurandStateXORWOW)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gpurandGenerator_t, hiprandGenerator_t)
WWR_SAME_TYPE(gpurandStatus_t, hiprandStatus_t)
WWR_SAME_TYPE(gpurandRngType_t, hiprandRngType_t)
WWR_SAME_TYPE(gpurandOrdering_t, hiprandOrdering_t)
WWR_SAME_TYPE(gpurandDirectionVectorSet_t, hiprandDirectionVectorSet_t)
WWR_SAME_TYPE(gpurandDirectionVectors32_t, hiprandDirectionVectors32_t)
WWR_SAME_TYPE(gpurandDirectionVectors64_t, hiprandDirectionVectors64_t)
WWR_SAME_TYPE(gpurandDiscreteDistribution_t, hiprandDiscreteDistribution_t)

WWR_SAME_VALUE(GPURAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
WWR_SAME_VALUE(GPURAND_STATUS_VERSION_MISMATCH, HIPRAND_STATUS_VERSION_MISMATCH)
WWR_SAME_VALUE(GPURAND_STATUS_NOT_INITIALIZED, HIPRAND_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(GPURAND_STATUS_ALLOCATION_FAILED, HIPRAND_STATUS_ALLOCATION_FAILED)
WWR_SAME_VALUE(GPURAND_STATUS_TYPE_ERROR, HIPRAND_STATUS_TYPE_ERROR)
WWR_SAME_VALUE(GPURAND_STATUS_OUT_OF_RANGE, HIPRAND_STATUS_OUT_OF_RANGE)
WWR_SAME_VALUE(GPURAND_STATUS_LENGTH_NOT_MULTIPLE, HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_SAME_VALUE(GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
                  HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_SAME_VALUE(GPURAND_STATUS_LAUNCH_FAILURE, HIPRAND_STATUS_LAUNCH_FAILURE)
WWR_SAME_VALUE(GPURAND_STATUS_PREEXISTING_FAILURE, HIPRAND_STATUS_PREEXISTING_FAILURE)
WWR_SAME_VALUE(GPURAND_STATUS_INITIALIZATION_FAILED, HIPRAND_STATUS_INITIALIZATION_FAILED)
WWR_SAME_VALUE(GPURAND_STATUS_ARCH_MISMATCH, HIPRAND_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(GPURAND_STATUS_INTERNAL_ERROR, HIPRAND_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(GPURAND_RNG_TEST, HIPRAND_RNG_TEST)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_XORWOW, HIPRAND_RNG_PSEUDO_XORWOW)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MRG32K3A, HIPRAND_RNG_PSEUDO_MRG32K3A)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MTGP32, HIPRAND_RNG_PSEUDO_MTGP32)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_MT19937, HIPRAND_RNG_PSEUDO_MT19937)
WWR_SAME_VALUE(GPURAND_RNG_PSEUDO_PHILOX4_32_10, HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_DEFAULT, HIPRAND_RNG_QUASI_DEFAULT)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SOBOL32, HIPRAND_RNG_QUASI_SOBOL32)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32, HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SOBOL64)
WWR_SAME_VALUE(GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64, HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_BEST, HIPRAND_ORDERING_PSEUDO_BEST)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_DEFAULT, HIPRAND_ORDERING_PSEUDO_DEFAULT)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_SEEDED, HIPRAND_ORDERING_PSEUDO_SEEDED)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_LEGACY, HIPRAND_ORDERING_PSEUDO_LEGACY)
WWR_SAME_VALUE(GPURAND_ORDERING_PSEUDO_DYNAMIC, HIPRAND_ORDERING_PSEUDO_DYNAMIC)
WWR_SAME_VALUE(GPURAND_ORDERING_QUASI_DEFAULT, HIPRAND_ORDERING_QUASI_DEFAULT)
WWR_SAME_VALUE(GPURAND_DIRECTION_VECTORS_32_JOEKUO6, HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
                  HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_SAME_VALUE(GPURAND_DIRECTION_VECTORS_64_JOEKUO6, HIPRAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_SAME_VALUE(GPURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
                  HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

WWR_SAME_FUNCTION(gpurandCreateGenerator, hiprandCreateGenerator)
WWR_SAME_FUNCTION(gpurandCreateGeneratorHost, hiprandCreateGeneratorHost)
WWR_SAME_FUNCTION(gpurandDestroyGenerator, hiprandDestroyGenerator)
WWR_SAME_FUNCTION(gpurandGetVersion, hiprandGetVersion)
WWR_SAME_FUNCTION(gpurandSetStream, hiprandSetStream)
WWR_SAME_FUNCTION(gpurandSetPseudoRandomGeneratorSeed, hiprandSetPseudoRandomGeneratorSeed)
WWR_SAME_FUNCTION(gpurandSetGeneratorOffset, hiprandSetGeneratorOffset)
WWR_SAME_FUNCTION(gpurandSetGeneratorOrdering, hiprandSetGeneratorOrdering)
WWR_SAME_FUNCTION(gpurandSetQuasiRandomGeneratorDimensions,
                     hiprandSetQuasiRandomGeneratorDimensions)
WWR_SAME_FUNCTION(gpurandGenerateSeeds, hiprandGenerateSeeds)
WWR_SAME_FUNCTION(gpurandGenerate, hiprandGenerate)
WWR_SAME_FUNCTION(gpurandGenerateLongLong, hiprandGenerateLongLong)
WWR_SAME_FUNCTION(gpurandGenerateUniform, hiprandGenerateUniform)
WWR_SAME_FUNCTION(gpurandGenerateUniformDouble, hiprandGenerateUniformDouble)
WWR_SAME_FUNCTION(gpurandGenerateNormal, hiprandGenerateNormal)
WWR_SAME_FUNCTION(gpurandGenerateNormalDouble, hiprandGenerateNormalDouble)
WWR_SAME_FUNCTION(gpurandGenerateLogNormal, hiprandGenerateLogNormal)
WWR_SAME_FUNCTION(gpurandGenerateLogNormalDouble, hiprandGenerateLogNormalDouble)
WWR_SAME_FUNCTION(gpurandGeneratePoisson, hiprandGeneratePoisson)
WWR_SAME_FUNCTION(gpurandCreatePoissonDistribution, hiprandCreatePoissonDistribution)
WWR_SAME_FUNCTION(gpurandDestroyDistribution, hiprandDestroyDistribution)
WWR_SAME_FUNCTION(gpurandGetDirectionVectors32, hiprandGetDirectionVectors32)
WWR_SAME_FUNCTION(gpurandGetDirectionVectors64, hiprandGetDirectionVectors64)
WWR_SAME_FUNCTION(gpurandGetScrambleConstants32, hiprandGetScrambleConstants32)
WWR_SAME_FUNCTION(gpurandGetScrambleConstants64, hiprandGetScrambleConstants64)

// Device generator state types (hiprand_kernel.h, via
// gpumod.hip.hiprand_kernel -- a separate module here, unlike cuRAND)
WWR_SAME_TYPE(gpurandStateXORWOW, hiprandStateXORWOW)
WWR_SAME_TYPE(gpurandStateXORWOW_t, hiprandStateXORWOW_t)
WWR_SAME_TYPE(gpurandStateMRG32k3a, hiprandStateMRG32k3a)
WWR_SAME_TYPE(gpurandStateMRG32k3a_t, hiprandStateMRG32k3a_t)
WWR_SAME_TYPE(gpurandStateMtgp32, hiprandStateMtgp32)
WWR_SAME_TYPE(gpurandStateMtgp32_t, hiprandStateMtgp32_t)
WWR_SAME_TYPE(gpurandStatePhilox4_32_10, hiprandStatePhilox4_32_10)
WWR_SAME_TYPE(gpurandStatePhilox4_32_10_t, hiprandStatePhilox4_32_10_t)
WWR_SAME_TYPE(gpurandStateSobol32, hiprandStateSobol32)
WWR_SAME_TYPE(gpurandStateSobol32_t, hiprandStateSobol32_t)
WWR_SAME_TYPE(gpurandStateScrambledSobol32, hiprandStateScrambledSobol32)
WWR_SAME_TYPE(gpurandStateScrambledSobol32_t, hiprandStateScrambledSobol32_t)
WWR_SAME_TYPE(gpurandStateSobol64, hiprandStateSobol64)
WWR_SAME_TYPE(gpurandStateSobol64_t, hiprandStateSobol64_t)
WWR_SAME_TYPE(gpurandStateScrambledSobol64, hiprandStateScrambledSobol64)
WWR_SAME_TYPE(gpurandStateScrambledSobol64_t, hiprandStateScrambledSobol64_t)
WWR_SAME_TYPE(gpurandState, hiprandState)
WWR_SAME_TYPE(gpurandState_t, hiprandState_t)

// On HIP the default state is NOT the XORWOW state: hiprand_kernel_rocm.h's
// DEFINE_HIPRAND_STATE macro emits a fresh struct for each, both deriving
// from rocrand_state_xorwow -- same layout, distinct C++ types. The CUDA
// branch above asserts the opposite, deliberately: this is the one place
// gpu.rand's two backends differ in type IDENTITY rather than spelling, and
// src/rand.cppm's device-state section is written around it.
static_assert(!std::is_same_v<gpurandState, gpurandStateXORWOW>);
static_assert(sizeof(gpurandState) == sizeof(gpurandStateXORWOW));

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: getScrambleConstants has hipRAND's const-correct signature
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpurandGetScrambleConstants32)>,
                 gpurandStatus_t(const unsigned int **))
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpurandGetScrambleConstants64)>,
                 gpurandStatus_t(const unsigned long long **))

} // namespace wwr::test
