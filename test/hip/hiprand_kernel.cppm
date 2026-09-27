// hiprand_kernel.cppm - Compile-time tests for gpumod.hip.hiprand_kernel

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hiprand_kernel;

import std;
import gpumod.hip.hiprand_kernel;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hiprand_kernel
//
// No LINK_CHECKs: this module exports types only, never a function. See
// src/hip/hiprand_kernel.cppm for why the device functions are absent (they
// are __device__-qualified and a module unit is host code).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Every state type is a complete class type
//
// Completeness is the point of this module: the HOST sizes and allocates the
// per-thread state array, so `sizeof` has to work on plain host code. An
// incomplete type would still satisfy `using ::hiprandState;` and fail only
// at the call site.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<hiprandState>);
static_assert(sizeof(hiprandState) > 0);
static_assert(std::is_class_v<hiprandStateXORWOW>);
static_assert(sizeof(hiprandStateXORWOW) > 0);
static_assert(std::is_class_v<hiprandStateMRG32k3a>);
static_assert(sizeof(hiprandStateMRG32k3a) > 0);
static_assert(std::is_class_v<hiprandStateMtgp32>);
static_assert(sizeof(hiprandStateMtgp32) > 0);
static_assert(std::is_class_v<hiprandStatePhilox4_32_10>);
static_assert(sizeof(hiprandStatePhilox4_32_10) > 0);
static_assert(std::is_class_v<hiprandStateSobol32>);
static_assert(sizeof(hiprandStateSobol32) > 0);
static_assert(std::is_class_v<hiprandStateScrambledSobol32>);
static_assert(sizeof(hiprandStateScrambledSobol32) > 0);
static_assert(std::is_class_v<hiprandStateSobol64>);
static_assert(sizeof(hiprandStateSobol64) > 0);
static_assert(std::is_class_v<hiprandStateScrambledSobol64>);
static_assert(sizeof(hiprandStateScrambledSobol64) > 0);

// ────────────────────────────────────────────────────────────────────────
// The `_t` spelling is the same type as the struct, for every generator
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<hiprandState_t, hiprandState>);
static_assert(std::is_same_v<hiprandStateXORWOW_t, hiprandStateXORWOW>);
static_assert(std::is_same_v<hiprandStateMRG32k3a_t, hiprandStateMRG32k3a>);
static_assert(std::is_same_v<hiprandStateMtgp32_t, hiprandStateMtgp32>);
static_assert(std::is_same_v<hiprandStatePhilox4_32_10_t, hiprandStatePhilox4_32_10>);
static_assert(std::is_same_v<hiprandStateSobol32_t, hiprandStateSobol32>);
static_assert(std::is_same_v<hiprandStateScrambledSobol32_t, hiprandStateScrambledSobol32>);
static_assert(std::is_same_v<hiprandStateSobol64_t, hiprandStateSobol64>);
static_assert(std::is_same_v<hiprandStateScrambledSobol64_t, hiprandStateScrambledSobol64>);

// ────────────────────────────────────────────────────────────────────────
// hiprandState is NOT hiprandStateXORWOW -- the one place hipRAND's device
// state types differ in SHAPE from cuRAND's, not just in spelling
//
// cuRAND makes the default state an alias (`curandState` IS
// `curandStateXORWOW`). hipRAND's DEFINE_HIPRAND_STATE macro emits a fresh
// struct per generator, so its two xorwow-backed states are distinct types
// that merely share a base. Pinned here deliberately: if a future ROCm
// release collapses them into an alias, THIS assert fails and says so, which
// is what lets src/rand.cppm and src/rand.cuh keep treating
// gpurandState as its own type rather than quietly assuming otherwise.
// ────────────────────────────────────────────────────────────────────────

static_assert(!std::is_same_v<hiprandState, hiprandStateXORWOW>);
static_assert(sizeof(hiprandState) == sizeof(hiprandStateXORWOW));
static_assert(std::is_base_of_v<hiprandState::base, hiprandState>);
static_assert(std::is_same_v<hiprandState::base, hiprandStateXORWOW::base>);

} // namespace wwr::hip::test
