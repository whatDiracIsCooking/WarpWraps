// nvJitLink.cppm - Compile-time tests for wwr.cuda.nvJitLink

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.nvJitLink;

import std;
import wwr.cuda.nvJitLink;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.nvJitLink
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. The result enum type satisfies std::is_enum_v
//   2. The input type enum satisfies std::is_enum_v
//   3. Key enumerator values match the nvJitLink-specified integer values
//   4. The nvJitLinkHandle is a pointer type
//   5. nvJitLinkVersion (the only non-inline extern symbol) resolves at link time
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvJitLinkResult>);
static_assert(std::is_enum_v<nvJitLinkInputType>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvJitLinkResult
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJITLINK_SUCCESS) == 0);
static_assert(static_cast<int>(NVJITLINK_ERROR_UNRECOGNIZED_OPTION) == 1);
static_assert(static_cast<int>(NVJITLINK_ERROR_MISSING_ARCH) == 2);
static_assert(static_cast<int>(NVJITLINK_ERROR_INVALID_INPUT) == 3);
static_assert(static_cast<int>(NVJITLINK_ERROR_PTX_COMPILE) == 4);
static_assert(static_cast<int>(NVJITLINK_ERROR_NVVM_COMPILE) == 5);
static_assert(static_cast<int>(NVJITLINK_ERROR_INTERNAL) == 6);
static_assert(static_cast<int>(NVJITLINK_ERROR_THREADPOOL) == 7);
static_assert(static_cast<int>(NVJITLINK_ERROR_UNRECOGNIZED_INPUT) == 8);
static_assert(static_cast<int>(NVJITLINK_ERROR_FINALIZE) == 9);
static_assert(static_cast<int>(NVJITLINK_ERROR_NULL_INPUT) == 10);
static_assert(static_cast<int>(NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS) == 11);
static_assert(static_cast<int>(NVJITLINK_ERROR_INCORRECT_INPUT_TYPE) == 12);
static_assert(static_cast<int>(NVJITLINK_ERROR_ARCH_MISMATCH) == 13);
static_assert(static_cast<int>(NVJITLINK_ERROR_OUTDATED_LIBRARY) == 14);
static_assert(static_cast<int>(NVJITLINK_ERROR_MISSING_FATBIN) == 15);
static_assert(static_cast<int>(NVJITLINK_ERROR_UNRECOGNIZED_ARCH) == 16);
static_assert(static_cast<int>(NVJITLINK_ERROR_UNSUPPORTED_ARCH) == 17);
static_assert(static_cast<int>(NVJITLINK_ERROR_LTO_NOT_ENABLED) == 18);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvJitLinkInputType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJITLINK_INPUT_NONE) == 0);
static_assert(static_cast<int>(NVJITLINK_INPUT_CUBIN) == 1);
static_assert(static_cast<int>(NVJITLINK_INPUT_PTX) == 2);
static_assert(static_cast<int>(NVJITLINK_INPUT_LTOIR) == 3);
static_assert(static_cast<int>(NVJITLINK_INPUT_FATBIN) == 4);
static_assert(static_cast<int>(NVJITLINK_INPUT_OBJECT) == 5);
static_assert(static_cast<int>(NVJITLINK_INPUT_LIBRARY) == 6);
static_assert(static_cast<int>(NVJITLINK_INPUT_INDEX) == 7);
static_assert(static_cast<int>(NVJITLINK_INPUT_ANY) == 10);

// ────────────────────────────────────────────────────────────────────────
// Handle type check
// nvJitLinkHandle is typedef'd as a pointer to struct nvJitLink
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<nvJitLinkHandle>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
//
// The public API functions (nvJitLinkCreate, nvJitLinkDestroy, etc.) are
// defined as `static inline` wrappers in the header under the
// NVJITLINK_NO_INLINE guard.  Static inline functions do not produce
// externally-linkable symbols, so WWR_LINK_CHECK cannot be applied to them.
//
// nvJitLinkVersion is the one function declared as a plain `extern` (not
// inline), making it the only symbol we can verify with WWR_LINK_CHECK.
// ────────────────────────────────────────────────────────────────────────

// Version query (the only non-inline extern in the public API)
WWR_LINK_CHECK(nvJitLinkVersion)

} // namespace wwr::cuda::test
