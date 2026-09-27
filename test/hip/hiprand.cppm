// hiprand.cppm - Compile-time tests for wwr.hip.hiprand

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hiprand;

import std;
import wwr.hip.hiprand;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hiprand
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

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
// WWR_LINK_CHECK: generator management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiprandCreateGenerator)
WWR_LINK_CHECK(hiprandCreateGeneratorHost)
WWR_LINK_CHECK(hiprandDestroyGenerator)
WWR_LINK_CHECK(hiprandGetVersion)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: generator configuration
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiprandSetStream)
WWR_LINK_CHECK(hiprandSetPseudoRandomGeneratorSeed)
WWR_LINK_CHECK(hiprandSetGeneratorOffset)
WWR_LINK_CHECK(hiprandSetGeneratorOrdering)
WWR_LINK_CHECK(hiprandSetQuasiRandomGeneratorDimensions)
WWR_LINK_CHECK(hiprandGenerateSeeds)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: generation functions
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiprandGenerate)
WWR_LINK_CHECK(hiprandGenerateChar)
WWR_LINK_CHECK(hiprandGenerateShort)
WWR_LINK_CHECK(hiprandGenerateLongLong)
WWR_LINK_CHECK(hiprandGenerateUniform)
WWR_LINK_CHECK(hiprandGenerateUniformDouble)
WWR_LINK_CHECK(hiprandGenerateUniformHalf)
WWR_LINK_CHECK(hiprandGenerateNormal)
WWR_LINK_CHECK(hiprandGenerateNormalDouble)
WWR_LINK_CHECK(hiprandGenerateNormalHalf)
WWR_LINK_CHECK(hiprandGenerateLogNormal)
WWR_LINK_CHECK(hiprandGenerateLogNormalDouble)
WWR_LINK_CHECK(hiprandGenerateLogNormalHalf)
WWR_LINK_CHECK(hiprandGeneratePoisson)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: discrete distribution management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiprandCreatePoissonDistribution)
WWR_LINK_CHECK(hiprandDestroyDistribution)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: direction vectors and scramble constants
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiprandGetDirectionVectors32)
WWR_LINK_CHECK(hiprandGetDirectionVectors64)
WWR_LINK_CHECK(hiprandGetScrambleConstants32)
WWR_LINK_CHECK(hiprandGetScrambleConstants64)

} // namespace wwr::hip::test
