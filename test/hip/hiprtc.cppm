// hiprtc.cppm - Compile-time tests for gpumod.hip.hiprtc

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hiprtc;

import std;
import gpumod.hip.hiprtc;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hiprtc
//
// Mirrors test/cuda/nvrtc.cppm. The module is a pure re-export (using
// declarations). We verify at compile-time that:
//   1. The result enum type satisfies std::is_enum_v
//   2. Key enumerator values match the HIPRTC-specified integer values
//   3. The hiprtcProgram / hiprtcLinkState handles are pointer types
//   4. Every re-exported function resolves to a linkable external symbol
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type check
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hiprtcResult>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiprtcResult
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPRTC_SUCCESS) == 0);
static_assert(static_cast<int>(HIPRTC_ERROR_OUT_OF_MEMORY) == 1);
static_assert(static_cast<int>(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE) == 2);
static_assert(static_cast<int>(HIPRTC_ERROR_INVALID_INPUT) == 3);
static_assert(static_cast<int>(HIPRTC_ERROR_INVALID_PROGRAM) == 4);
static_assert(static_cast<int>(HIPRTC_ERROR_INVALID_OPTION) == 5);
static_assert(static_cast<int>(HIPRTC_ERROR_COMPILATION) == 6);
static_assert(static_cast<int>(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE) == 7);
static_assert(static_cast<int>(HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION) == 8);
static_assert(static_cast<int>(HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION) == 9);
static_assert(static_cast<int>(HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID) == 10);
static_assert(static_cast<int>(HIPRTC_ERROR_INTERNAL_ERROR) == 11);
static_assert(static_cast<int>(HIPRTC_ERROR_LINKING) == 100);

// ────────────────────────────────────────────────────────────────────────
// Handle type checks
// hiprtcProgram is typedef'd as a pointer to struct _hiprtcProgram;
// hiprtcLinkState is typedef'd as a pointer to struct ihiprtcLinkState.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<hiprtcProgram>);
static_assert(std::is_pointer_v<hiprtcLinkState>);

// ────────────────────────────────────────────────────────────────────────
// Enum type check: hipJitOption / hipJitInputType (from hip/linker_types.h,
// included directly by hiprtc.h -- see src/hip/hiprtc.cppm's doc comment)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipJitOption>);
static_assert(std::is_enum_v<hipJitInputType>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Error handling
GPUMOD_LINK_CHECK(hiprtcGetErrorString)

// General information query
GPUMOD_LINK_CHECK(hiprtcVersion)

// Program lifecycle
GPUMOD_LINK_CHECK(hiprtcCreateProgram)
GPUMOD_LINK_CHECK(hiprtcDestroyProgram)

// Compilation
GPUMOD_LINK_CHECK(hiprtcCompileProgram)

// Code / bitcode retrieval
GPUMOD_LINK_CHECK(hiprtcGetCodeSize)
GPUMOD_LINK_CHECK(hiprtcGetCode)
GPUMOD_LINK_CHECK(hiprtcGetBitcodeSize)
GPUMOD_LINK_CHECK(hiprtcGetBitcode)

// Compilation log retrieval
GPUMOD_LINK_CHECK(hiprtcGetProgramLogSize)
GPUMOD_LINK_CHECK(hiprtcGetProgramLog)

// Name expression (symbol mangling)
GPUMOD_LINK_CHECK(hiprtcAddNameExpression)
GPUMOD_LINK_CHECK(hiprtcGetLoweredName)

// Linking
GPUMOD_LINK_CHECK(hiprtcLinkCreate)
GPUMOD_LINK_CHECK(hiprtcLinkAddFile)
GPUMOD_LINK_CHECK(hiprtcLinkAddData)
GPUMOD_LINK_CHECK(hiprtcLinkComplete)
GPUMOD_LINK_CHECK(hiprtcLinkDestroy)

} // namespace gpumod::hip::test
