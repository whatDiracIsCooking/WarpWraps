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

namespace wwr::hip::test {

using namespace wwr::hip;

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

WWR_LINK_CHECK(make_hipFloatComplex)
WWR_LINK_CHECK(make_hipDoubleComplex)
WWR_LINK_CHECK(make_hipComplex)
WWR_LINK_CHECK(hipCrealf)
WWR_LINK_CHECK(hipCimagf)
WWR_LINK_CHECK(hipCreal)
WWR_LINK_CHECK(hipCimag)
WWR_LINK_CHECK(hipCaddf)
WWR_LINK_CHECK(hipCsubf)
WWR_LINK_CHECK(hipCmulf)
WWR_LINK_CHECK(hipCdivf)
WWR_LINK_CHECK(hipCsqabsf)
WWR_LINK_CHECK(hipCabsf)
WWR_LINK_CHECK(hipConjf)
WWR_LINK_CHECK(hipCadd)
WWR_LINK_CHECK(hipCsub)
WWR_LINK_CHECK(hipCmul)
WWR_LINK_CHECK(hipCdiv)
WWR_LINK_CHECK(hipCsqabs)
WWR_LINK_CHECK(hipCabs)
WWR_LINK_CHECK(hipConj)
WWR_LINK_CHECK(hipComplexDoubleToFloat)
WWR_LINK_CHECK(hipComplexFloatToDouble)
WWR_LINK_CHECK(hipCfmaf)
WWR_LINK_CHECK(hipCfma)

} // namespace wwr::hip::test
