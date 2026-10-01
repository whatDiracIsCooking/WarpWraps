/**
 * @file atomic.cppm
 * @brief Backend-neutral scoped-atomic enums: wwrMemoryOrder / wwrThreadScope
 *
 * The host-module counterpart to atomic.h's device-pass-gated section, and the
 * one part of the scoped-atomic surface a module can carry. The operations
 * themselves -- wwrAtomicLoad / wwrAtomicFetchAdd / ... -- are
 * `__device__ __forceinline__`, callable from no host TU, so they stay in
 * atomic.h's device section and a device .cu reaches them by #include. What
 * crosses the host/device boundary is the pair of enums a host configurator
 * picks and a kernel consumes: `wwrThreadScope` as the scope template argument
 * and `wwrMemoryOrder` as the order runtime argument. See backend.h.
 *
 * Usage:
 *   import wwr.atomic;
 *
 *   constexpr auto scope = wwrThreadScope::device;   // host-side policy choice,
 *   constexpr auto order = wwrMemoryOrder::acq_rel;  // handed to a kernel
 *
 * This is the thin end of the `.h` + `.cppm` shape: unlike complex.cppm /
 * rand.cppm it re-exports no functions and imports no raw vendor module, because
 * the enums are plain project-owned data with no vendor entity behind them --
 * the rand.cppm state-type tail, where a plain `using` in this exported
 * namespace re-exports the global-module enum the GMF #include brought in. No
 * WWR_SELECT, no backend #if: the two enums are defined once, identically on
 * both backends, in atomic.h.
 *
 * A module over the host-side atomic *operations* (`cuda::atomic` over managed
 * memory) was declined in #123 -- a 21MB BMI from pulling <cuda/atomic> into the
 * module, and no signature-identity assertion the way every raw .cppm has. This
 * enum-only module escapes both: its GMF #includes only atomic.h, whose host
 * path carries no vendor header, so the BMI is tiny and there is nothing to
 * assert against beyond the enums being exported (test/gpu/atomic.cppm). See
 * docs/architecture.md §20.
 */

module;

// The enums (wwrMemoryOrder / wwrThreadScope), from atomic.h -- they land in the
// global module here and are re-exported below. atomic.h's device section gates
// itself out in this host compile, so nothing else of the header enters.
#include "atomic.h"

export module wwr.atomic;

export namespace wwr {

// ========================================================================
// Enums -- re-exported from atomic.h (the global-module definitions the GMF
// #include brought in), so importers of wwr.atomic see them. A plain `using`,
// not WWR_TYPE/WWR_SELECT: the backend-neutral definition already lives in
// atomic.h, and restating it would be duplication (the rand.cppm state-type
// tail, not the complex.cppm forwarder shape).
// ========================================================================

using wwr::wwrMemoryOrder;
using wwr::wwrThreadScope;

} // namespace wwr
