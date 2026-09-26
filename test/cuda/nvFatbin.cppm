// nvFatbin.cppm - Compile-time tests for gpumod.cuda.nvFatbin

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.nvFatbin;

import std;
import gpumod.cuda.nvFatbin;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.nvFatbin
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. The result enum type satisfies std::is_enum_v
//   2. Every enumerator value matches the value assigned in nvFatbin.h
//   3. The nvFatbinHandle handle is a pointer type
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type check
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvFatbinResult>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvFatbinResult
// Values are assigned by sequential implicit enumeration in nvFatbin.h.
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVFATBIN_SUCCESS) == 0);
static_assert(static_cast<int>(NVFATBIN_ERROR_INTERNAL) == 1);
static_assert(static_cast<int>(NVFATBIN_ERROR_ELF_ARCH_MISMATCH) == 2);
static_assert(static_cast<int>(NVFATBIN_ERROR_ELF_SIZE_MISMATCH) == 3);
static_assert(static_cast<int>(NVFATBIN_ERROR_MISSING_PTX_VERSION) == 4);
static_assert(static_cast<int>(NVFATBIN_ERROR_NULL_POINTER) == 5);
static_assert(static_cast<int>(NVFATBIN_ERROR_COMPRESSION_FAILED) == 6);
static_assert(static_cast<int>(NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED) == 7);
static_assert(static_cast<int>(NVFATBIN_ERROR_UNRECOGNIZED_OPTION) == 8);
static_assert(static_cast<int>(NVFATBIN_ERROR_INVALID_ARCH) == 9);
static_assert(static_cast<int>(NVFATBIN_ERROR_INVALID_NVVM) == 10);
static_assert(static_cast<int>(NVFATBIN_ERROR_EMPTY_INPUT) == 11);
static_assert(static_cast<int>(NVFATBIN_ERROR_MISSING_PTX_ARCH) == 12);
static_assert(static_cast<int>(NVFATBIN_ERROR_PTX_ARCH_MISMATCH) == 13);
static_assert(static_cast<int>(NVFATBIN_ERROR_MISSING_FATBIN) == 14);
static_assert(static_cast<int>(NVFATBIN_ERROR_INVALID_INDEX) == 15);
static_assert(static_cast<int>(NVFATBIN_ERROR_IDENTIFIER_REUSE) == 16);
static_assert(static_cast<int>(NVFATBIN_ERROR_INTERNAL_PTX_OPTION) == 17);

// ────────────────────────────────────────────────────────────────────────
// Handle type check
// nvFatbinHandle is typedef'd as a pointer to struct _nvFatbinHandle
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<nvFatbinHandle>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Error handling
GPUMOD_LINK_CHECK(nvFatbinGetErrorString)

// Handle lifecycle
GPUMOD_LINK_CHECK(nvFatbinCreate)
GPUMOD_LINK_CHECK(nvFatbinDestroy)

// Content addition
GPUMOD_LINK_CHECK(nvFatbinAddPTX)
GPUMOD_LINK_CHECK(nvFatbinAddCubin)
GPUMOD_LINK_CHECK(nvFatbinAddLTOIR)
GPUMOD_LINK_CHECK(nvFatbinAddIndex)
GPUMOD_LINK_CHECK(nvFatbinAddReloc)

// Fatbinary retrieval
GPUMOD_LINK_CHECK(nvFatbinSize)
GPUMOD_LINK_CHECK(nvFatbinGet)

// Version query
GPUMOD_LINK_CHECK(nvFatbinVersion)

} // namespace gpumod::cuda::test
