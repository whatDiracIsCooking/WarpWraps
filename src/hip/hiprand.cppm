/**
 * @file hiprand.cppm
 * @brief hipRAND API module wrapper for gpumod project
 *
 * Wraps hiprand/hiprand.h -- the hipRAND *host* API only. CUDA counterpart:
 * wwr.cuda.curand, which additionally wraps curand_kernel.h for the
 * device-side generator state types. hipRAND's counterpart to that lives in
 * its own module, wwr.hip.hiprand_kernel, so host-API consumers do not pay
 * for the rocRAND device-generator implementation it drags in.
 *
 * hiprandDirectionVectors32_t / 64_t are plain C array typedefs on the AMD
 * platform (unsigned int[32] / unsigned long long[64], from hiprand_rocm.h),
 * not the struct typedefs cuRAND uses.
 *
 * hiprandGenerator_t / hiprandDiscreteDistribution_t are opaque pointers to
 * incomplete struct types, exported the way cuRAND's equivalents are.
 *
 * Usage:
 *   import wwr.hip.hiprand;
 */

module;

#include <hiprand/hiprand.h>

export module wwr.hip.hiprand;

import std;

export namespace wwr::hip {

// ========================================================================
// Core Types
// ========================================================================
using ::hiprandDiscreteDistribution_t;
using ::hiprandGenerator_t;
using ::hiprandStatus_t;

// Status enum values
using ::HIPRAND_STATUS_ALLOCATION_FAILED;
using ::HIPRAND_STATUS_ARCH_MISMATCH;
using ::HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
using ::HIPRAND_STATUS_INITIALIZATION_FAILED;
using ::HIPRAND_STATUS_INTERNAL_ERROR;
using ::HIPRAND_STATUS_LAUNCH_FAILURE;
using ::HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
using ::HIPRAND_STATUS_NOT_IMPLEMENTED;
using ::HIPRAND_STATUS_NOT_INITIALIZED;
using ::HIPRAND_STATUS_OUT_OF_RANGE;
using ::HIPRAND_STATUS_PREEXISTING_FAILURE;
using ::HIPRAND_STATUS_SUCCESS;
using ::HIPRAND_STATUS_TYPE_ERROR;
using ::HIPRAND_STATUS_VERSION_MISMATCH;

// ========================================================================
// Enumerations - RNG Type
// ========================================================================
using ::HIPRAND_RNG_PSEUDO_DEFAULT;
using ::HIPRAND_RNG_PSEUDO_MRG32K3A;
using ::HIPRAND_RNG_PSEUDO_MT19937;
using ::HIPRAND_RNG_PSEUDO_MTGP32;
using ::HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
using ::HIPRAND_RNG_PSEUDO_XORWOW;
using ::HIPRAND_RNG_QUASI_DEFAULT;
using ::HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
using ::HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
using ::HIPRAND_RNG_QUASI_SOBOL32;
using ::HIPRAND_RNG_QUASI_SOBOL64;
using ::HIPRAND_RNG_TEST;
using ::hiprandRngType_t;

// ========================================================================
// Enumerations - Ordering
// ========================================================================
using ::HIPRAND_ORDERING_PSEUDO_BEST;
using ::HIPRAND_ORDERING_PSEUDO_DEFAULT;
using ::HIPRAND_ORDERING_PSEUDO_DYNAMIC;
using ::HIPRAND_ORDERING_PSEUDO_LEGACY;
using ::HIPRAND_ORDERING_PSEUDO_SEEDED;
using ::HIPRAND_ORDERING_QUASI_DEFAULT;
using ::hiprandOrdering_t;

// ========================================================================
// Enumerations - Direction Vector Set
// ========================================================================
using ::HIPRAND_DIRECTION_VECTORS_32_JOEKUO6;
using ::HIPRAND_DIRECTION_VECTORS_64_JOEKUO6;
using ::HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
using ::HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
using ::hiprandDirectionVectorSet_t;

// ========================================================================
// Direction Vector Types (plain array typedefs on the AMD platform)
// ========================================================================
using ::hiprandDirectionVectors32_t;
using ::hiprandDirectionVectors64_t;

// ========================================================================
// Generator Management Functions
// ========================================================================
using ::hiprandCreateGenerator;
using ::hiprandCreateGeneratorHost;
using ::hiprandDestroyGenerator;
using ::hiprandGetVersion;

// ========================================================================
// Generator Configuration Functions
// ========================================================================
using ::hiprandGenerateSeeds;
using ::hiprandSetGeneratorOffset;
using ::hiprandSetGeneratorOrdering;
using ::hiprandSetPseudoRandomGeneratorSeed;
using ::hiprandSetQuasiRandomGeneratorDimensions;
using ::hiprandSetStream;

// ========================================================================
// Generation Functions - Uniform Integer
// ========================================================================
using ::hiprandGenerate;
using ::hiprandGenerateChar;
using ::hiprandGenerateLongLong;
using ::hiprandGenerateShort;

// ========================================================================
// Generation Functions - Uniform Floating Point
// ========================================================================
using ::hiprandGenerateUniform;
using ::hiprandGenerateUniformDouble;
using ::hiprandGenerateUniformHalf;

// ========================================================================
// Generation Functions - Normal Distribution
// ========================================================================
using ::hiprandGenerateNormal;
using ::hiprandGenerateNormalDouble;
using ::hiprandGenerateNormalHalf;

// ========================================================================
// Generation Functions - Log-Normal Distribution
// ========================================================================
using ::hiprandGenerateLogNormal;
using ::hiprandGenerateLogNormalDouble;
using ::hiprandGenerateLogNormalHalf;

// ========================================================================
// Generation Functions - Poisson Distribution
// ========================================================================
using ::hiprandGeneratePoisson;

// ========================================================================
// Discrete Distribution Management
// ========================================================================
using ::hiprandCreatePoissonDistribution;
using ::hiprandDestroyDistribution;

// ========================================================================
// Direction Vectors and Scramble Constants
// ========================================================================
using ::hiprandGetDirectionVectors32;
using ::hiprandGetDirectionVectors64;
using ::hiprandGetScrambleConstants32;
using ::hiprandGetScrambleConstants64;

} // namespace wwr::hip
