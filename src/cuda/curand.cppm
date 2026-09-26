/**
 * @file curand.cppm
 * @brief cuRAND API module wrapper for gpumod project
 *
 * This module wraps the cuRAND API for use in C++20/C++23 module-based code.
 * It exports types, constants, and host-side functions for cuRAND library management
 * and random number generation.
 *
 * Usage:
 *   import gpumod.cuda.curand;
 *
 * Note: This module exports cuRAND host API functions for random number generation.
 */

module;

#include <curand.h>
#include <curand_kernel.h>

export module gpumod.cuda.curand;

// ========================================================================
// Export all cuRAND types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Core Types
// ========================================================================
using ::curandGenerator_t;
using ::curandStatus_t;

// Status enum values
using ::CURAND_STATUS_ALLOCATION_FAILED;
using ::CURAND_STATUS_ARCH_MISMATCH;
using ::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
using ::CURAND_STATUS_INITIALIZATION_FAILED;
using ::CURAND_STATUS_INTERNAL_ERROR;
using ::CURAND_STATUS_LAUNCH_FAILURE;
using ::CURAND_STATUS_LENGTH_NOT_MULTIPLE;
using ::CURAND_STATUS_NOT_INITIALIZED;
using ::CURAND_STATUS_OUT_OF_RANGE;
using ::CURAND_STATUS_PREEXISTING_FAILURE;
using ::CURAND_STATUS_SUCCESS;
using ::CURAND_STATUS_TYPE_ERROR;
using ::CURAND_STATUS_VERSION_MISMATCH;

// ========================================================================
// Enumerations - RNG Types
// ========================================================================
using ::curandRngType_t;
// curandRngType_t enum values
using ::CURAND_RNG_PSEUDO_DEFAULT;
using ::CURAND_RNG_PSEUDO_MRG32K3A;
using ::CURAND_RNG_PSEUDO_MT19937;
using ::CURAND_RNG_PSEUDO_MTGP32;
using ::CURAND_RNG_PSEUDO_PHILOX4_32_10;
using ::CURAND_RNG_PSEUDO_XORWOW;
using ::CURAND_RNG_QUASI_DEFAULT;
using ::CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
using ::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
using ::CURAND_RNG_QUASI_SOBOL32;
using ::CURAND_RNG_QUASI_SOBOL64;
using ::CURAND_RNG_TEST;

// ========================================================================
// Enumerations - Ordering
// ========================================================================
using ::curandOrdering_t;
// curandOrdering_t enum values
using ::CURAND_ORDERING_PSEUDO_BEST;
using ::CURAND_ORDERING_PSEUDO_DEFAULT;
using ::CURAND_ORDERING_PSEUDO_DYNAMIC;
using ::CURAND_ORDERING_PSEUDO_LEGACY;
using ::CURAND_ORDERING_PSEUDO_SEEDED;
using ::CURAND_ORDERING_QUASI_DEFAULT;

// ========================================================================
// Enumerations - Direction Vector Set
// ========================================================================
using ::curandDirectionVectorSet_t;
// curandDirectionVectorSet_t enum values
using ::CURAND_DIRECTION_VECTORS_32_JOEKUO6;
using ::CURAND_DIRECTION_VECTORS_64_JOEKUO6;
using ::CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
using ::CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;

// ========================================================================
// Direction Vector Types
// ========================================================================
using ::curandDirectionVectors32_t;
using ::curandDirectionVectors64_t;

// ========================================================================
// Distribution Types
// ========================================================================
using ::curandDiscreteDistribution_t;
using ::curandDistribution_st;
using ::curandDistribution_t;

// ========================================================================
// Method Enumeration
// ========================================================================
using ::curandMethod_t;
// curandMethod_t enum values
using ::CURAND_3RD;
using ::CURAND_BINARY_SEARCH;
using ::CURAND_CHOOSE_BEST;
using ::CURAND_DEFINITION;
using ::CURAND_DEVICE_API;
using ::CURAND_DISCRETE_GAUSS;
using ::CURAND_FAST_REJECTION;
using ::CURAND_HITR;
using ::CURAND_ITR;
using ::CURAND_KNUTH;
using ::CURAND_M1;
using ::CURAND_M2;
using ::CURAND_POISSON;
using ::CURAND_REJECTION;

// ========================================================================
// Generator Management Functions
// ========================================================================
using ::curandCreateGenerator;
using ::curandCreateGeneratorHost;
using ::curandDestroyGenerator;
using ::curandGetProperty;
using ::curandGetVersion;

// ========================================================================
// Generator Configuration Functions
// ========================================================================
using ::curandGenerateSeeds;
using ::curandSetGeneratorOffset;
using ::curandSetGeneratorOrdering;
using ::curandSetPseudoRandomGeneratorSeed;
using ::curandSetQuasiRandomGeneratorDimensions;
using ::curandSetStream;

// ========================================================================
// Generation Functions - Uniform Integer
// ========================================================================
using ::curandGenerate;
using ::curandGenerateLongLong;

// ========================================================================
// Generation Functions - Uniform Floating Point
// ========================================================================
using ::curandGenerateUniform;
using ::curandGenerateUniformDouble;

// ========================================================================
// Generation Functions - Normal Distribution
// ========================================================================
using ::curandGenerateNormal;
using ::curandGenerateNormalDouble;

// ========================================================================
// Generation Functions - Log-Normal Distribution
// ========================================================================
using ::curandGenerateLogNormal;
using ::curandGenerateLogNormalDouble;

// ========================================================================
// Generation Functions - Poisson Distribution
// ========================================================================
using ::curandGeneratePoisson;
using ::curandGeneratePoissonMethod;

// ========================================================================
// Generation Functions - Binomial Distribution
// ========================================================================
using ::curandGenerateBinomial;
using ::curandGenerateBinomialMethod;

// ========================================================================
// Discrete Distribution Management
// ========================================================================
using ::curandCreatePoissonDistribution;
using ::curandDestroyDistribution;

// ========================================================================
// Direction Vectors and Scramble Constants
// ========================================================================
using ::curandGetDirectionVectors32;
using ::curandGetDirectionVectors64;
using ::curandGetScrambleConstants32;
using ::curandGetScrambleConstants64;

// ========================================================================
// Device API Types (from curand_kernel.h)
// ========================================================================
// Device state types
using ::curandStateMRG32k3a;
using ::curandStateMRG32k3a_t;
using ::curandStateMtgp32;
using ::curandStateMtgp32_t;
using ::curandStatePhilox4_32_10;
using ::curandStatePhilox4_32_10_t;
using ::curandStateScrambledSobol32;
using ::curandStateScrambledSobol32_t;
using ::curandStateScrambledSobol64;
using ::curandStateScrambledSobol64_t;
using ::curandStateSobol32;
using ::curandStateSobol32_t;
using ::curandStateSobol64;
using ::curandStateSobol64_t;
using ::curandStateXORWOW;
using ::curandStateXORWOW_t;

// Default state type aliases
using ::curandState;
using ::curandState_t;

} // namespace gpumod::cuda
