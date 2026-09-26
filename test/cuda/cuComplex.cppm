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

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

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

GPUMOD_LINK_CHECK(make_cuFloatComplex)
GPUMOD_LINK_CHECK(make_cuDoubleComplex)
GPUMOD_LINK_CHECK(make_cuComplex)
GPUMOD_LINK_CHECK(cuCrealf)
GPUMOD_LINK_CHECK(cuCimagf)
GPUMOD_LINK_CHECK(cuCreal)
GPUMOD_LINK_CHECK(cuCimag)
GPUMOD_LINK_CHECK(cuCaddf)
GPUMOD_LINK_CHECK(cuCsubf)
GPUMOD_LINK_CHECK(cuCmulf)
GPUMOD_LINK_CHECK(cuCdivf)
GPUMOD_LINK_CHECK(cuCabsf)
GPUMOD_LINK_CHECK(cuConjf)
GPUMOD_LINK_CHECK(cuCadd)
GPUMOD_LINK_CHECK(cuCsub)
GPUMOD_LINK_CHECK(cuCmul)
GPUMOD_LINK_CHECK(cuCdiv)
GPUMOD_LINK_CHECK(cuCabs)
GPUMOD_LINK_CHECK(cuConj)
GPUMOD_LINK_CHECK(cuComplexDoubleToFloat)
GPUMOD_LINK_CHECK(cuComplexFloatToDouble)

} // namespace gpumod::cuda::test
