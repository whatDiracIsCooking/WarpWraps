// cuComplex.cppm - Compile-time tests for gpumod.cuda.cuComplex

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cuComplex;

import std;
import gpumod.cuda.cuComplex;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests for gpumod.cuda.cuComplex
//
// Unlike the rest of src/cuda, this module is NOT a pure re-export: cuComplex.h
// functions have `static inline` linkage, so the wrapper provides thin
// forwarding bodies with actual arithmetic in them. Type-trait and link-time
// checks alone would miss a mistake in that forwarding logic (e.g. a swapped
// operand or a wrong sign), and that logic is not tested: checking it means
// calling non-constexpr host functions, which this compile-time-only
// directory cannot do.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<cuFloatComplex>);
static_assert(std::is_trivially_copyable_v<cuDoubleComplex>);
static_assert(std::is_standard_layout_v<cuFloatComplex>);
static_assert(std::is_standard_layout_v<cuDoubleComplex>);
static_assert(std::is_same_v<cuComplex, cuFloatComplex>);
static_assert(sizeof(cuFloatComplex) == 2 * sizeof(float));
static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double));

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(make_cuFloatComplex)
WWR_LINK_CHECK(make_cuDoubleComplex)
WWR_LINK_CHECK(make_cuComplex)
WWR_LINK_CHECK(cuCrealf)
WWR_LINK_CHECK(cuCimagf)
WWR_LINK_CHECK(cuCreal)
WWR_LINK_CHECK(cuCimag)
WWR_LINK_CHECK(cuCaddf)
WWR_LINK_CHECK(cuCsubf)
WWR_LINK_CHECK(cuCmulf)
WWR_LINK_CHECK(cuCdivf)
WWR_LINK_CHECK(cuCabsf)
WWR_LINK_CHECK(cuConjf)
WWR_LINK_CHECK(cuCadd)
WWR_LINK_CHECK(cuCsub)
WWR_LINK_CHECK(cuCmul)
WWR_LINK_CHECK(cuCdiv)
WWR_LINK_CHECK(cuCabs)
WWR_LINK_CHECK(cuConj)
WWR_LINK_CHECK(cuComplexDoubleToFloat)
WWR_LINK_CHECK(cuComplexFloatToDouble)

} // namespace wwr::cuda::test
