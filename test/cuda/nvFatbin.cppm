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

namespace wwr::cuda::test {

using namespace wwr::cuda;

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
WWR_LINK_CHECK(nvFatbinGetErrorString)

// Handle lifecycle
WWR_LINK_CHECK(nvFatbinCreate)
WWR_LINK_CHECK(nvFatbinDestroy)

// Content addition
WWR_LINK_CHECK(nvFatbinAddPTX)
WWR_LINK_CHECK(nvFatbinAddCubin)
WWR_LINK_CHECK(nvFatbinAddLTOIR)
WWR_LINK_CHECK(nvFatbinAddIndex)
WWR_LINK_CHECK(nvFatbinAddReloc)

// Fatbinary retrieval
WWR_LINK_CHECK(nvFatbinSize)
WWR_LINK_CHECK(nvFatbinGet)

// Version query
WWR_LINK_CHECK(nvFatbinVersion)

} // namespace wwr::cuda::test
