// hip_complex.cppm - Compile-time tests for gpumod.hip.hip_complex

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hip_complex;

import std;
import gpumod.hip.hip_complex;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hip_complex
//
// Like gpumod.cuda.cuComplex, this module is NOT a pure re-export:
// amd_hip_complex.h's functions have `static inline` linkage, so the wrapper
// provides thin forwarding bodies. Runtime correctness of the forwarded
// arithmetic is out of scope for this compile-time-only directory (parity
// with cuComplex.cppm's test).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ──────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<hipFloatComplex>);
static_assert(std::is_trivially_copyable_v<hipDoubleComplex>);
static_assert(std::is_standard_layout_v<hipFloatComplex>);
static_assert(std::is_standard_layout_v<hipDoubleComplex>);
static_assert(std::is_same_v<hipComplex, hipFloatComplex>);
static_assert(sizeof(hipFloatComplex) == 2 * sizeof(float));
static_assert(sizeof(hipDoubleComplex) == 2 * sizeof(double));

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ──────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(make_hipFloatComplex)
GPUMOD_LINK_CHECK(make_hipDoubleComplex)
GPUMOD_LINK_CHECK(make_hipComplex)
GPUMOD_LINK_CHECK(hipCrealf)
GPUMOD_LINK_CHECK(hipCimagf)
GPUMOD_LINK_CHECK(hipCreal)
GPUMOD_LINK_CHECK(hipCimag)
GPUMOD_LINK_CHECK(hipCaddf)
GPUMOD_LINK_CHECK(hipCsubf)
GPUMOD_LINK_CHECK(hipCmulf)
GPUMOD_LINK_CHECK(hipCdivf)
GPUMOD_LINK_CHECK(hipCsqabsf)
GPUMOD_LINK_CHECK(hipCabsf)
GPUMOD_LINK_CHECK(hipConjf)
GPUMOD_LINK_CHECK(hipCadd)
GPUMOD_LINK_CHECK(hipCsub)
GPUMOD_LINK_CHECK(hipCmul)
GPUMOD_LINK_CHECK(hipCdiv)
GPUMOD_LINK_CHECK(hipCsqabs)
GPUMOD_LINK_CHECK(hipCabs)
GPUMOD_LINK_CHECK(hipConj)
GPUMOD_LINK_CHECK(hipComplexDoubleToFloat)
GPUMOD_LINK_CHECK(hipComplexFloatToDouble)
GPUMOD_LINK_CHECK(hipCfmaf)
GPUMOD_LINK_CHECK(hipCfma)

} // namespace gpumod::hip::test
