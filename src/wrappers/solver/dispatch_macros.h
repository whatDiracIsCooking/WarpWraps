/**
 * @file dispatch_macros.h
 * @brief Type-based dispatch for the wwrsolverDn* typed API
 *
 * The wwrsolverDn* legacy typed API spells its functions
 * wwrsolverDn<letter><basename>, so it dispatches with the shared, prefix-
 * agnostic WWR_REAL_DISPATCH / WWR_COMPLEX_DISPATCH from wrappers/common/dispatch_sdcz.h and
 * needs nothing more. No _64 variants: neither cuSOLVER's nor hipSOLVER's
 * legacy typed API has an int64_t-index sibling entry point -- the 64-bit
 * dimensions only appear in the modern (X-prefixed) API, which takes a runtime
 * wwrsolverDataType_t instead of a compile-time type suffix.
 *
 * Prerequisites (must be provided by the including file):
 * - std::is_same_v (via `import std;` or equivalent)
 * - wwrComplex, wwrDoubleComplex and the wwrsolverDn* functions
 *   (wwr.complex, wwr.solver)
 */

#pragma once

#include "wrappers/common/dispatch_sdcz.h"
