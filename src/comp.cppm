/**
 * @file comp.cppm
 * @brief Backend-neutral batched compression: wwrcomp* over nvCOMP / hipCOMP
 *
 * The one vendor pair where the two backends are version-skewed rather than
 * prefix-skewed: nvCOMP is 5.3, hipCOMP is a hipify of nvCOMP branch-2.2, so the
 * batched low-level signatures diverge. This layer carries option (a) of issue
 * #110 -- the measured 2.2 intersection -- for the three algorithms both
 * backends implement in both directions: LZ4, Snappy and Cascaded. (Bitcomp,
 * ANS and GDeflate are NVIDIA-proprietary schemes hipCOMP ships as headers
 * without device support, so they are reachable only through the raw vendor
 * modules, never portably.)
 *
 * Each wwrcompBatched<Algo>* function is written once against the hipCOMP 2.2
 * shape and forwards to whichever backend was selected. Reconciling the skew
 * costs three things on the CUDA path, all invisible to the neutral caller:
 *   - the split Sync/Async temp-size queries collapse to the Async (upper-bound)
 *     form; the exact Sync query is unreachable here;
 *   - compress device_statuses is passed NULL (nvCOMP documents this as "error
 *     status not reported"); hipCOMP 2.2 has no such parameter;
 *   - the nvCOMP-only decompress backend / bitshuffle options are defaulted.
 * A caller needing any of those -- or Zstd/GZIP/Deflate/CRC32, or the hardware
 * decompression engine -- uses wwr.cuda.nvcomp directly. This layer is for
 * backend-portable code that accepts the 2.2 ceiling. hipCOMP self-describes as
 * an early-access preview (every algorithm experimental, not performance-tuned),
 * so the HIP path is a portability contract, not a production guarantee.
 *
 * Opts carry only the fields both backends share: LZ4 its data_type, Snappy
 * nothing, Cascaded its full tuning (chunk_size / type / num_RLEs / num_deltas /
 * use_bp -- these survive field-for-field into nvCOMP 5.3). The neutral opts are
 * plain structs, distinct from either vendor's; the forwarders build the backend
 * struct from them.
 *
 * Enumerator VALUES are pinned in test/gpu/comp.cppm, one line per name: matching
 * names never guarantee matching values (this bit WWRRAND_RNG_* -- see
 * rand.cppm). Here they happen to agree, which the pins lock in.
 *
 * The wwrcomp* surface itself -- types, constants, neutral opts, backend option
 * builders and the LZ4/Snappy/Cascaded forwarders -- is NOT restated here: it is
 * the one fragment the two host paths share, detail/comp_names.h, pasted below
 * inside the exported `namespace wwr` exactly as wwr/comp.h (the non-module
 * #include path) pastes it. Add a name there, once, and both paths gain it. The
 * vendor headers are drawn from comp.h, the single vendor-include point (the
 * "rand.h shape"): this module binds wwr* references straight to the `::nvcomp*`
 * / `::hipcomp*` declarations with the _RAW macros and imports no raw vendor
 * module -- nvCOMP / hipCOMP are external-linkage libraries, so a reference or
 * type alias needs only the declaration, and the shims call the vendor globals
 * directly. The raw module wwr.cuda.nvcomp / wwr.hip.hipcomp stays for the full
 * surface this layer omits, off this path.
 *
 * Usage:
 *   import wwr.comp;
 *
 *   wwrcompBatchedLZ4Opts opts{WWRCOMP_TYPE_CHAR};
 *   wwrcompBatchedLZ4CompressGetTempSize(batch, max_chunk, opts, &temp_bytes);
 *   wwrcompBatchedLZ4CompressAsync(unc_ptrs, unc_bytes, max_chunk, batch,
 *       temp, temp_bytes, comp_ptrs, comp_bytes, opts, stream);
 */

module;

#include "backend.h"

// The single vendor-include point for the comp layer: nvcomp/*.h / hipcomp/*.h
// plus <cstddef>, whose `::nvcomp*` / `::hipcomp*` declarations the _RAW bindings
// and the shims in detail/comp_names.h resolve against. No raw vendor module is
// imported -- nvCOMP / hipCOMP are external-linkage libraries, so a reference,
// type alias or a shim forwarding to a vendor global needs only these
// declarations. See comp.h and backend.h.
#include "comp.h"

export module wwr.comp;

export namespace wwr {

// The whole neutral compression surface -- types, constants, neutral opts, the
// backend option builders and the LZ4/Snappy/Cascaded batched forwarders -- lives
// in detail/comp_names.h, the one list both host paths share: this module and
// wwr/comp.h (the non-module #include path). The WWR_*_RAW macros from backend.h,
// WWR_SELECTED_*, std::size_t and the vendor headers from comp.h's #include are
// exactly what that fragment's header documents it needs in scope.
#include "detail/comp_names.h"

} // namespace wwr
