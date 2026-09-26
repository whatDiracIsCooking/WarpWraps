// nvrtc.cppm - Compile-time tests for gpumod.cuda.nvrtc

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.nvrtc;

import std;
import gpumod.cuda.nvrtc;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.nvrtc
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. The result enum type satisfies std::is_enum_v
//   2. Key enumerator values match the NVRTC-specified integer values
//   3. The nvrtcProgram handle is a pointer type
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type check
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvrtcResult>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvrtcResult
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVRTC_SUCCESS) == 0);
static_assert(static_cast<int>(NVRTC_ERROR_OUT_OF_MEMORY) == 1);
static_assert(static_cast<int>(NVRTC_ERROR_PROGRAM_CREATION_FAILURE) == 2);
static_assert(static_cast<int>(NVRTC_ERROR_INVALID_INPUT) == 3);
static_assert(static_cast<int>(NVRTC_ERROR_INVALID_PROGRAM) == 4);
static_assert(static_cast<int>(NVRTC_ERROR_INVALID_OPTION) == 5);
static_assert(static_cast<int>(NVRTC_ERROR_COMPILATION) == 6);
static_assert(static_cast<int>(NVRTC_ERROR_BUILTIN_OPERATION_FAILURE) == 7);
static_assert(static_cast<int>(NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION) == 8);
static_assert(static_cast<int>(NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION) == 9);
static_assert(static_cast<int>(NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID) == 10);
static_assert(static_cast<int>(NVRTC_ERROR_INTERNAL_ERROR) == 11);
static_assert(static_cast<int>(NVRTC_ERROR_TIME_FILE_WRITE_FAILED) == 12);
static_assert(static_cast<int>(NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED) == 13);
static_assert(static_cast<int>(NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED) == 14);
static_assert(static_cast<int>(NVRTC_ERROR_PCH_CREATE) == 15);
static_assert(static_cast<int>(NVRTC_ERROR_CANCELLED) == 16);
static_assert(static_cast<int>(NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED) == 17);

// ────────────────────────────────────────────────────────────────────────
// Handle type check
// nvrtcProgram is typedef'd as a pointer to struct _nvrtcProgram
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<nvrtcProgram>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Error handling
GPUMOD_LINK_CHECK(nvrtcGetErrorString)

// General information query
GPUMOD_LINK_CHECK(nvrtcVersion)
GPUMOD_LINK_CHECK(nvrtcGetNumSupportedArchs)
GPUMOD_LINK_CHECK(nvrtcGetSupportedArchs)

// Program lifecycle
GPUMOD_LINK_CHECK(nvrtcCreateProgram)
GPUMOD_LINK_CHECK(nvrtcDestroyProgram)

// Compilation
GPUMOD_LINK_CHECK(nvrtcCompileProgram)

// PTX retrieval
GPUMOD_LINK_CHECK(nvrtcGetPTXSize)
GPUMOD_LINK_CHECK(nvrtcGetPTX)

// CUBIN retrieval
GPUMOD_LINK_CHECK(nvrtcGetCUBINSize)
GPUMOD_LINK_CHECK(nvrtcGetCUBIN)

// LTO IR retrieval
GPUMOD_LINK_CHECK(nvrtcGetLTOIRSize)
GPUMOD_LINK_CHECK(nvrtcGetLTOIR)

// OptiX IR retrieval
GPUMOD_LINK_CHECK(nvrtcGetOptiXIRSize)
GPUMOD_LINK_CHECK(nvrtcGetOptiXIR)

// Compilation log retrieval
GPUMOD_LINK_CHECK(nvrtcGetProgramLogSize)
GPUMOD_LINK_CHECK(nvrtcGetProgramLog)

// Name expression (symbol mangling)
GPUMOD_LINK_CHECK(nvrtcAddNameExpression)
GPUMOD_LINK_CHECK(nvrtcGetLoweredName)

// Precompiled header (PCH)
GPUMOD_LINK_CHECK(nvrtcGetPCHHeapSize)
GPUMOD_LINK_CHECK(nvrtcSetPCHHeapSize)
GPUMOD_LINK_CHECK(nvrtcGetPCHCreateStatus)
GPUMOD_LINK_CHECK(nvrtcGetPCHHeapSizeRequired)

// Compilation flow control
GPUMOD_LINK_CHECK(nvrtcSetFlowCallback)

} // namespace gpumod::cuda::test
