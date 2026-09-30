// curand.cppm - Compile-time tests for wwr.cuda.curand

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.curand;

import std;
import wwr.cuda.curand;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.curand
//
// Link-check-focused (#117): forces the linker to resolve every re-exported
// function symbol, so a re-export the libcurand.so does not actually export fails the
// build. A symbol the header declares but the library does not define uses
// WWR_DECLARED_CHECK instead, naming the version it was found missing from.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(curandCreateGenerator)
WWR_LINK_CHECK(curandCreateGeneratorHost)
WWR_LINK_CHECK(curandDestroyGenerator)
WWR_LINK_CHECK(curandGetProperty)
WWR_LINK_CHECK(curandGetVersion)
WWR_LINK_CHECK(curandGenerateSeeds)
WWR_LINK_CHECK(curandSetGeneratorOffset)
WWR_LINK_CHECK(curandSetGeneratorOrdering)
WWR_LINK_CHECK(curandSetPseudoRandomGeneratorSeed)
WWR_LINK_CHECK(curandSetQuasiRandomGeneratorDimensions)
WWR_LINK_CHECK(curandSetStream)
WWR_LINK_CHECK(curandGenerate)
WWR_LINK_CHECK(curandGenerateLongLong)
WWR_LINK_CHECK(curandGenerateUniform)
WWR_LINK_CHECK(curandGenerateUniformDouble)
WWR_LINK_CHECK(curandGenerateNormal)
WWR_LINK_CHECK(curandGenerateNormalDouble)
WWR_LINK_CHECK(curandGenerateLogNormal)
WWR_LINK_CHECK(curandGenerateLogNormalDouble)
WWR_LINK_CHECK(curandGeneratePoisson)
WWR_LINK_CHECK(curandGeneratePoissonMethod)
WWR_LINK_CHECK(curandGenerateBinomial)
WWR_LINK_CHECK(curandGenerateBinomialMethod)
WWR_LINK_CHECK(curandCreatePoissonDistribution)
WWR_LINK_CHECK(curandDestroyDistribution)
WWR_LINK_CHECK(curandGetDirectionVectors32)
WWR_LINK_CHECK(curandGetDirectionVectors64)
WWR_LINK_CHECK(curandGetScrambleConstants32)
WWR_LINK_CHECK(curandGetScrambleConstants64)

} // namespace wwr::cuda::test
