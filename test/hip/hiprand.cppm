// hiprand.cppm - Compile-time tests for gpumod.hip.hiprand

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hiprand;

import std;
import gpumod.hip.hiprand;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hiprand
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hiprandStatus_t>);
static_assert(std::is_enum_v<hiprandRngType_t>);
static_assert(std::is_enum_v<hiprandOrdering_t>);
static_assert(std::is_enum_v<hiprandDirectionVectorSet_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiprandStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPRAND_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(HIPRAND_STATUS_VERSION_MISMATCH) == 100);
static_assert(static_cast<int>(HIPRAND_STATUS_NOT_INITIALIZED) == 101);
static_assert(static_cast<int>(HIPRAND_STATUS_ALLOCATION_FAILED) == 102);
static_assert(static_cast<int>(HIPRAND_STATUS_TYPE_ERROR) == 103);
static_assert(static_cast<int>(HIPRAND_STATUS_OUT_OF_RANGE) == 104);
static_assert(static_cast<int>(HIPRAND_STATUS_LENGTH_NOT_MULTIPLE) == 105);
static_assert(static_cast<int>(HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED) == 106);
static_assert(static_cast<int>(HIPRAND_STATUS_LAUNCH_FAILURE) == 201);
static_assert(static_cast<int>(HIPRAND_STATUS_PREEXISTING_FAILURE) == 202);
static_assert(static_cast<int>(HIPRAND_STATUS_INITIALIZATION_FAILED) == 203);
static_assert(static_cast<int>(HIPRAND_STATUS_ARCH_MISMATCH) == 204);
static_assert(static_cast<int>(HIPRAND_STATUS_INTERNAL_ERROR) == 999);
static_assert(static_cast<int>(HIPRAND_STATUS_NOT_IMPLEMENTED) == 1000);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiprandRngType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPRAND_RNG_TEST) == 0);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_DEFAULT) == 400);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_XORWOW) == 401);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_MRG32K3A) == 402);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_MTGP32) == 403);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_MT19937) == 404);
static_assert(static_cast<int>(HIPRAND_RNG_PSEUDO_PHILOX4_32_10) == 405);
static_assert(static_cast<int>(HIPRAND_RNG_QUASI_DEFAULT) == 500);
static_assert(static_cast<int>(HIPRAND_RNG_QUASI_SOBOL32) == 501);
static_assert(static_cast<int>(HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32) == 502);
static_assert(static_cast<int>(HIPRAND_RNG_QUASI_SOBOL64) == 503);
static_assert(static_cast<int>(HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64) == 504);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiprandOrdering_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPRAND_ORDERING_PSEUDO_BEST) == 100);
static_assert(static_cast<int>(HIPRAND_ORDERING_PSEUDO_DEFAULT) == 101);
static_assert(static_cast<int>(HIPRAND_ORDERING_PSEUDO_SEEDED) == 102);
static_assert(static_cast<int>(HIPRAND_ORDERING_PSEUDO_LEGACY) == 103);
static_assert(static_cast<int>(HIPRAND_ORDERING_PSEUDO_DYNAMIC) == 104);
static_assert(static_cast<int>(HIPRAND_ORDERING_QUASI_DEFAULT) == 201);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiprandDirectionVectorSet_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPRAND_DIRECTION_VECTORS_32_JOEKUO6) == 101);
static_assert(static_cast<int>(HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6) == 102);
static_assert(static_cast<int>(HIPRAND_DIRECTION_VECTORS_64_JOEKUO6) == 103);
static_assert(static_cast<int>(HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6) == 104);

// ────────────────────────────────────────────────────────────────────────
// Direction vector types: plain C arrays on the AMD platform
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<hiprandDirectionVectors32_t, unsigned int[32]>);
static_assert(std::is_same_v<hiprandDirectionVectors64_t, unsigned long long[64]>);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<hiprandGenerator_t>);
static_assert(std::is_pointer_v<hiprandDiscreteDistribution_t>);

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: generator management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hiprandCreateGenerator)
GPUMOD_LINK_CHECK(hiprandCreateGeneratorHost)
GPUMOD_LINK_CHECK(hiprandDestroyGenerator)
GPUMOD_LINK_CHECK(hiprandGetVersion)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: generator configuration
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hiprandSetStream)
GPUMOD_LINK_CHECK(hiprandSetPseudoRandomGeneratorSeed)
GPUMOD_LINK_CHECK(hiprandSetGeneratorOffset)
GPUMOD_LINK_CHECK(hiprandSetGeneratorOrdering)
GPUMOD_LINK_CHECK(hiprandSetQuasiRandomGeneratorDimensions)
GPUMOD_LINK_CHECK(hiprandGenerateSeeds)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: generation functions
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hiprandGenerate)
GPUMOD_LINK_CHECK(hiprandGenerateChar)
GPUMOD_LINK_CHECK(hiprandGenerateShort)
GPUMOD_LINK_CHECK(hiprandGenerateLongLong)
GPUMOD_LINK_CHECK(hiprandGenerateUniform)
GPUMOD_LINK_CHECK(hiprandGenerateUniformDouble)
GPUMOD_LINK_CHECK(hiprandGenerateUniformHalf)
GPUMOD_LINK_CHECK(hiprandGenerateNormal)
GPUMOD_LINK_CHECK(hiprandGenerateNormalDouble)
GPUMOD_LINK_CHECK(hiprandGenerateNormalHalf)
GPUMOD_LINK_CHECK(hiprandGenerateLogNormal)
GPUMOD_LINK_CHECK(hiprandGenerateLogNormalDouble)
GPUMOD_LINK_CHECK(hiprandGenerateLogNormalHalf)
GPUMOD_LINK_CHECK(hiprandGeneratePoisson)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: discrete distribution management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hiprandCreatePoissonDistribution)
GPUMOD_LINK_CHECK(hiprandDestroyDistribution)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: direction vectors and scramble constants
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hiprandGetDirectionVectors32)
GPUMOD_LINK_CHECK(hiprandGetDirectionVectors64)
GPUMOD_LINK_CHECK(hiprandGetScrambleConstants32)
GPUMOD_LINK_CHECK(hiprandGetScrambleConstants64)

} // namespace gpumod::hip::test
